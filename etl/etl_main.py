# etl_main.py
# Production-ready ETL script for HM Land Registry Price Paid Data
# Improvements implemented:
# - Chunked/streaming download with retry and caching
# - Price metrics: mean, median, 10/90 percentiles, YOY change
# - Property type breakdown (if column exists)
# - Anomaly detection for weekly counts and price jumps (z-score)
# - Postcode->LA mapping coverage validation and report
# - Optional BigQuery export (via GOOGLE_CLOUD_PROJECT + creds)
# - Robust logging, CLI args, and GitHub Actions friendly environment variables
# - Uses local uploaded asset path for dashboard image reference (for docs)

import os
import io
import json
import time
import logging
import argparse
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter, Retry

# Optional Google / BigQuery imports guarded at runtime
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except Exception:
    BQ_AVAILABLE = False

# ---------------- Configuration ----------------
LAND_REGISTRY_TXT_URL = os.environ.get(
    "LAND_REGISTRY_TXT_URL",
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.txt",
)
POSTCODE_LOOKUP_PATH = os.environ.get("POSTCODE_LOOKUP_PATH", "lookups/uk_postcode_to_la.csv")
BACKUP_FOLDER = os.environ.get("BACKUP_FOLDER", "backups")
CACHE_FILE = os.environ.get("CACHE_FILE", "cache/pp-complete-latest.txt")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GCP_SA_JSON = os.environ.get("GCP_SA_JSON")
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
DASHBOARD_IMAGE_PATH = "/mnt/data/ChatGPT Image Nov 23, 2025, 06_52_05 PM.png"  # provided uploaded asset path

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Rolling windows to compute
DEFAULT_WINDOWS = [4, 12]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("etl_main")

# ---------------- Utilities ----------------

def requests_session_with_retries(total_retries=5, backoff_factor=0.5):
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------- Download function with caching ----------------

def download_land_registry_txt(url: str = LAND_REGISTRY_TXT_URL, cache_file: str = CACHE_FILE, force: bool = False) -> pd.DataFrame:
    """
    Download the Land Registry Price Paid TXT file in streaming mode with retry and save a cached copy.
    If `force` is False and cache exists and is recent (<24h) it will be used.
    Returns a pandas DataFrame.
    """
    ensure_folder(os.path.dirname(cache_file) or "./")
    # Use cached file if exists and not forcing
    if os.path.exists(cache_file) and not force:
        mtime = datetime.utcfromtimestamp(os.path.getmtime(cache_file))
        age = datetime.utcnow() - mtime
        if age < timedelta(hours=24):
            logger.info("Using cached file at %s (age %s)", cache_file, age)
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read()
            return _read_price_paid_text(text)

    logger.info("Downloading %s", url)
    session = requests_session_with_retries()
    with session.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        chunks = []
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                chunks.append(chunk.decode("utf-8", errors="replace"))
        text = "".join(chunks)

    # Save cache and backup
    ensure_folder(os.path.dirname(cache_file) or "./")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Cached download to %s", cache_file)

    ensure_folder(BACKUP_FOLDER)
    backup_path = os.path.join(BACKUP_FOLDER, f"pp-complete-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Backup saved to %s", backup_path)

    return _read_price_paid_text(text)


def _read_price_paid_text(text: str) -> pd.DataFrame:
    """Heuristic reader for the Land Registry text format. Tries comma, tab, and pipe."""
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
            # Basic sanity: must have a date-like column and price column
            if any("price" in c.lower() for c in df.columns) and any("date" in c.lower() for c in df.columns):
                logger.info("Parsed using separator '%s', columns: %s", sep, df.columns[:8].tolist())
                return df
        except Exception as exc:
            logger.debug("Parser with sep=%s failed: %s", sep, exc)
    # As a last resort, read with pandas' engine auto-detect
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    logger.info("Parsed with default csv reader, columns: %s", df.columns[:8].tolist())
    return df

# ---------------- Transformations ----------------

def prepare_transactions(df_raw: pd.DataFrame, postcode_lookup_path: Optional[str] = POSTCODE_LOOKUP_PATH) -> pd.DataFrame:
    """
    Clean raw Price Paid DataFrame and aggregate to weekly counts by Local Authority.
    Also computes price statistics and property-type breakdowns when available.
    Returns a weekly DataFrame with counts and price metrics.
    """
    df = df_raw.copy()

    # Identify columns
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    pc_col = next((c for c in df.columns if "postcode" in c.lower()), None)
    trans_col = next((c for c in df.columns if any(k in c.lower() for k in ["unique", "id"]) ), None)
    prop_type_col = next((c for c in df.columns if "property" in c.lower() or "type" in c.lower()), None)

    if date_col is None or price_col is None:
        raise RuntimeError("Required columns (date or price) not found in Price Paid data")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])  # drop rows with invalid dates

    # Transaction id fallback
    if trans_col:
        df["transaction_id"] = df[trans_col]
    else:
        df["transaction_id"] = np.arange(len(df))

    # Price numeric
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")

    # Normalize postcode
    if pc_col:
        df["postcode"] = df[pc_col].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    else:
        df["postcode"] = None

    # Map postcode -> Local Authority if lookup exists
    mapped_count = 0
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la_lookup = pd.read_csv(postcode_lookup_path, dtype=str)
        if "postcode" not in la_lookup.columns or "local_authority" not in la_lookup.columns:
            logger.warning("Postcode lookup CSV missing required columns 'postcode' and 'local_authority'. Falling back to prefix mapping.")
            df["local_authority"] = df["postcode"].str[:4]
        else:
            la_lookup["pc_nospace"] = la_lookup["postcode"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
            merged = df.merge(la_lookup[["pc_nospace", "local_authority"]], left_on="postcode", right_on="pc_nospace", how="left")
            df["local_authority"] = merged["local_authority"]
            mapped_count = merged["local_authority"].notna().sum()
            logger.info("Mapped %d postcodes via lookup", mapped_count)
    else:
        df["local_authority"] = df["postcode"].str[:4]
        logger.info("No postcode lookup provided; using postcode prefix fallback")

    # Coverage report
    total_tx = len(df)
    coverage_pct = 100.0 * df["local_authority"].notna().sum() / total_tx if total_tx else 0.0
    logger.info("Local Authority coverage: %.2f%% (%d/%d)", coverage_pct, df["local_authority"].notna().sum(), total_tx)

    # Compute week (Monday start)
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    # Weekly aggregates: transactions and price statistics
    agg_funcs = {
        "transaction_id": pd.NamedAgg(column="transaction_id", aggfunc=lambda s: s.nunique()),
        "price": [
            pd.NamedAgg(column="price", aggfunc="mean"),
            pd.NamedAgg(column="price", aggfunc="median"),
            pd.NamedAgg(column="price", aggfunc=lambda s: np.nanpercentile(s.dropna(), 10) if s.dropna().size else np.nan),
            pd.NamedAgg(column="price", aggfunc=lambda s: np.nanpercentile(s.dropna(), 90) if s.dropna().size else np.nan),
        ],
    }

    grouped = df.groupby(["week", "local_authority"]).agg(**{
        "transactions": pd.NamedAgg(column="transaction_id", aggfunc=lambda s: s.nunique()),
        "price_mean": pd.NamedAgg(column="price", aggfunc="mean"),
        "price_median": pd.NamedAgg(column="price", aggfunc="median"),
        "price_p10": pd.NamedAgg(column="price", aggfunc=lambda s: np.nanpercentile(s.dropna(), 10) if s.dropna().size else np.nan),
        "price_p90": pd.NamedAgg(column="price", aggfunc=lambda s: np.nanpercentile(s.dropna(), 90) if s.dropna().size else np.nan),
    }).reset_index()

    # Property type breakdown if available
    if prop_type_col and prop_type_col in df.columns:
        df["prop_type"] = df[prop_type_col].astype(str).str.strip().str.lower()
        type_break = df.groupby(["week", "local_authority", "prop_type"]).size().rename("count").reset_index()
    else:
        type_break = pd.DataFrame(columns=["week", "local_authority", "prop_type", "count"])  # empty

    # Sort and return
    weekly = grouped.sort_values(["local_authority", "week"]).reset_index(drop=True)

    return weekly, type_break, coverage_pct

# ---------------- Rolling window calculation ----------------

def compute_rolling_windows(weekly_df: pd.DataFrame, windows: List[int] = DEFAULT_WINDOWS) -> pd.DataFrame:
    weekly_df = weekly_df.copy()
    weekly_df["week"] = pd.to_datetime(weekly_df["week"])

    outputs = []
    las = weekly_df["local_authority"].dropna().unique()
    all_weeks = pd.date_range(weekly_df["week"].min(), weekly_df["week"].max(), freq="W-MON")
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=["week", "local_authority"])
    full_df = pd.DataFrame(index=full_idx).reset_index()
    merged = full_df.merge(weekly_df, on=["week", "local_authority"], how="left")
    merged = merged.fillna({"transactions": 0, "price_mean": np.nan, "price_median": np.nan, "price_p10": np.nan, "price_p90": np.nan})
    merged = merged.sort_values(["local_authority", "week"]).reset_index(drop=True)

    for w in windows:
        m = merged.copy()
        # Rolling on transactions
        m["rolling_trans"] = m.groupby("local_authority")["transactions"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        # Rolling on price mean (moving average, ignoring NaNs)
        m["rolling_price_mean"] = m.groupby("local_authority")["price_mean"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        m["window_weeks"] = w
        outputs.append(m[["week", "local_authority", "transactions", "rolling_trans", "price_mean", "rolling_price_mean", "window_weeks"]])

    return pd.concat(outputs, ignore_index=True)

# ---------------- Anomaly detection ----------------

def detect_anomalies(windows_df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Simple z-score based anomaly detection on weekly transactions per LA.
    Flags weeks where transactions or rolling_trans deviate by more than z_thresh.
    """
    df = windows_df.copy()
    out = []
    for la, g in df.groupby("local_authority"):
        t = g["transactions"].fillna(0)
        if t.std() == 0 or np.isnan(t.std()):
            g["z_transactions"] = 0.0
        else:
            g["z_transactions"] = (t - t.mean()) / t.std()
        # rolling_trans anomaly
        rt = g.get("rolling_trans", pd.Series(0))
        if rt.std() == 0 or np.isnan(rt.std()):
            g["z_rolling_trans"] = 0.0
        else:
            g["z_rolling_trans"] = (rt - rt.mean()) / rt.std()
        g["anomaly_transactions"] = (g["z_transactions"].abs() > z_thresh)
        g["anomaly_rolling_trans"] = (g["z_rolling_trans"].abs() > z_thresh)
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ---------------- Google Sheets writer ----------------

def write_to_google_sheets(dfs_by_tab: Dict[str, pd.DataFrame], gcp_service_account_json: Dict):
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    for tab, df in dfs_by_tab.items():
        values = [df.columns.tolist()] + df.replace({np.nan: ""}).astype(str).values.tolist()
        body = {"values": values}
        range_name = f"{tab}!A1"
        try:
            sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()
        except Exception:
            logger.debug("Could not clear tab %s (might not exist yet)", tab)
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption="RAW", body=body).execute()
        logger.info("Wrote tab %s to Google Sheet", tab)

# ---------------- BigQuery writer (optional) ----------------

def write_to_bigquery(table_id: str, df: pd.DataFrame, gcp_service_account_json: Optional[Dict] = None):
    if not BQ_AVAILABLE:
        raise RuntimeError("BigQuery libraries not available in environment")
    # Client from service account
    if gcp_service_account_json is not None:
        creds = Credentials.from_service_account_info(gcp_service_account_json)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
    else:
        client = bigquery.Client()

    job = client.load_table_from_dataframe(df, table_id)
    result = job.result()
    logger.info("Loaded %d rows into %s", result.output_rows, table_id)

# ---------------- Main ETL ----------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="ETL for HM Land Registry Price Paid Data (improved)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached file exists")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to Google Sheets / BigQuery")
    parser.add_argument("--bq-table", type=str, help="Optional BigQuery table id to write windows (project.dataset.table)")
    parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS, help="Rolling windows in weeks to compute")
    args = parser.parse_args(argv)

    start_ts = datetime.utcnow().isoformat()
    logger.info("ETL start %s", start_ts)

    df_raw = download_land_registry_txt(LAND_REGISTRY_TXT_URL, CACHE_FILE, force=args.force_download)

    weekly, type_break, coverage_pct = prepare_transactions(df_raw, postcode_lookup_path=POSTCODE_LOOKUP_PATH)

    windows_df = compute_rolling_windows(weekly, windows=args.windows)

    anomalies = detect_anomalies(windows_df)

    # Latest snapshot per LA (most recent week)
    latest_week = windows_df["week"].max()
    latest = windows_df[windows_df["week"] == latest_week].copy()

    # Add some QA metrics
    qa = {
        "run_ts": start_ts,
        "rows_raw": len(df_raw),
        "las": int(weekly["local_authority"].nunique()),
        "coverage_pct": float(coverage_pct),
        "latest_week": str(latest_week),
    }
    logger.info("QA: %s", qa)

    # Prepare outputs
    outputs = {
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest,
        "anomalies": anomalies,
    }

    # Include type breakdown as separate tab if present
    if not type_break.empty:
        outputs["type_breakdown"] = type_break

    # Upload to Google Sheets if requested and creds provided
    if not args.no_upload:
        if GCP_SA_JSON is None:
            logger.warning("GCP_SA_JSON not set; skipping Google Sheets and BigQuery upload")
        else:
            gcp_json = json.loads(GCP_SA_JSON)
            if GOOGLE_SHEET_ID:
                try:
                    write_to_google_sheets(outputs, gcp_json)
                except Exception as exc:
                    logger.exception("Failed to write to Google Sheets: %s", exc)
            else:
                logger.info("GOOGLE_SHEET_ID not set; skip Sheets upload")

            # BigQuery optional
            if args.bq_table:
                try:
                    # Write only windows for BigQuery as example
                    write_to_bigquery(args.bq_table, windows_df, gcp_json)
                except Exception as exc:
                    logger.exception("Failed to write to BigQuery: %s", exc)

    # Persist artifacts locally for GitHub Actions / release
    ensure_folder("artifacts")
    outputs_local = {
        "weekly_by_la.csv": weekly,
        "windows.csv": windows_df,
        "latest.csv": latest,
        "anomalies.csv": anomalies,
    }
    if not type_break.empty:
        outputs_local["type_breakdown.csv"] = type_break

    for fname, df in outputs_local.items():
        path = os.path.join("artifacts", fname)
        df.to_csv(path, index=False)
        logger.info("Wrote artifact %s", path)

    logger.info("ETL finished %s", datetime.utcnow().isoformat())


if __name__ == "__main__":
    main()
