"""
Land Registry Price Paid ETL
- Downloads official pp-complete.txt dataset
- Cleans & aggregates weekly transactions by Local Authority
- Computes rolling 4 & 12 week totals
- Publishes results to Google Sheets
- Designed for GitHub Actions automation
"""

# ---- Standard library imports ----
import os                            # read environment variables like GOOGLE_SHEET_ID, GCP_SA_JSON
import io                            # convert downloaded text into file-like stream for pandas
import json                          # decode GCP service account JSON
from datetime import datetime        # timestamp logging

# ---- Third-party imports ----
import requests                      # download TXT file over HTTP
import pandas as pd                 # data processing, aggregation, rolling windows
import numpy as np                  # numeric helpers (nunique, NaN handling)

# ---- Google Sheets imports ----
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---- CONSTANTS ----

# ✅ ONLY WORKING RELIABLE SOURCE — TXT format
# DO NOT replace with CSV URLs — they intermittently return HTML or empty files
LAND_REGISTRY_TXT_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.txt"
)

# Google Sheet ID is securely injected via GitHub secret
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")

# Required API scopes for editing Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


# ---- 1) DOWNLOAD TXT FILE ----
def download_land_registry_txt():
    """
    Downloads the official pp-complete.txt dataset.
    TXT format is comma-separated, same as CSV but more reliable for automation.
    Returns DataFrame or raises RuntimeError upon failure.
    """
    print(f"📥 Downloading Land Registry TXT from {LAND_REGISTRY_TXT_URL} ...")

    try:
        # request full dataset — allow long timeout because file is large
        response = requests.get(LAND_REGISTRY_TXT_URL, timeout=300)
        response.raise_for_status()  # fail if not HTTP 200

        # decode file safely — ignore weird UTF-8 artifacts instead of erroring
        text = response.content.decode("utf-8", errors="ignore")

        # ensure non-empty file
        if not text.strip():
            raise RuntimeError("Downloaded TXT file is empty.")

        # read TXT into pandas — treat as CSV because it's comma-delimited
        df = pd.read_csv(
            io.StringIO(text),
            sep=",",
            low_memory=False,
            skip_blank_lines=True
        )

        # verify parsing worked
        if df.empty or len(df.columns) == 0:
            raise RuntimeError("TXT file downloaded but contains no valid data.")

        print(f"✅ Successfully loaded {len(df):,} rows.")
        return df

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download or parse TXT dataset: {e}")


# ---- 2) CLEAN + WEEKLY AGGREGATION ----
def prepare_transactions(df_raw):
    """
    Converts raw transaction-level dataset into weekly counts per Local Authority.
    """

    df = df_raw.copy()

    # Identify date column dynamically — column names vary slightly over years
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if not date_col:
        raise RuntimeError("No date column found — cannot continue ETL.")

    # convert date strings → datetime
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # Find postcode column (for grouping)
    pc_col = next((c for c in df.columns if "postcode" in c.lower()), None)
    if pc_col:
        df["postcode"] = df[pc_col].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    else:
        df["postcode"] = None

    # Assign synthetic transaction ID if none exists — needed for counting
    df["transaction_id"] = np.arange(len(df))

    # In absence of official LA mapping, approximate using first 4 postcode chars
    df["local_authority"] = df["postcode"].str[:4]

    # Convert dates to week start (Mon)
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    # Weekly unique transaction counts per LA
    weekly = (
        df.groupby(["week", "local_authority"])
        .agg(transactions=("transaction_id", "nunique"))
        .reset_index()
        .sort_values(["local_authority", "week"])
    )

    return weekly


# ---- 3) CREATE ROLLING WINDOWS ----
def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    Produces rolling transaction totals for configurable week windows.
    Example result: 4-week & 12-week trend indicators.
    """

    weekly_df["week"] = pd.to_datetime(weekly_df["week"])
    local_areas = weekly_df["local_authority"].unique()

    # build complete weekly timeline — ensures missing weeks fill with 0
    all_weeks = pd.date_range(
        weekly_df["week"].min(), weekly_df["week"].max(), freq="W-MON"
    )

    # create full week × LA matrix
    expanded = (
        pd.MultiIndex.from_product([all_weeks, local_areas], names=["week", "local_authority"])
        .to_frame(index=False)
        .merge(weekly_df, on=["week", "local_authority"], how="left")
        .fillna({"transactions": 0})
        .sort_values(["local_authority", "week"])
    )

    outputs = []

    for w in windows:
        temp = expanded.copy()
        temp["rolling_trans"] = (
            temp.groupby("local_authority")["transactions"]
            .transform(lambda s: s.rolling(w, min_periods=1).sum())
        )
        temp["window_weeks"] = w
        outputs.append(temp)

    return pd.concat(outputs, ignore_index=True)


# ---- 4) WRITE RESULTS TO GOOGLE SHEETS ----
def write_to_google_sheets(dfs, gcp_json):
    """
    Writes multiple DataFrames to separate tabs in a Google Sheet.
    """

    creds = Credentials.from_service_account_info(gcp_json, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    for tab, df in dfs.items():
        values = [df.columns.tolist()] + df.astype(str).replace({np.nan: ""}).values.tolist()

        # clear sheet tab first to avoid leftovers
        sheet.values().clear(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=tab
        ).execute()

        # upload new data
        sheet.values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            body={"values": values}
        ).execute()


# ---- 5) MAIN ETL ORCHESTRATION ----
def main():
    print(f"🚀 ETL start {datetime.utcnow().isoformat()}")

    df_raw = download_land_registry_txt()
    weekly = prepare_transactions(df_raw)
    windows = compute_rolling_windows(weekly)
    latest = windows[windows["week"] == windows["week"].max()].copy()

    # load service account JSON from GitHub Actions secret
    gcp_sa = os.environ.get("GCP_SA_JSON")
    if not gcp_sa:
        raise RuntimeError("Missing GCP_SA_JSON — add it to GitHub Secrets.")

    gcp_json = json.loads(gcp_sa)

    if not GOOGLE_SHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEET_ID — add it to GitHub Secrets.")

    write_to_google_sheets(
        {
            "weekly_by_la": weekly,
            "rolling_windows": windows,
            "latest_snapshot": latest,
        },
        gcp_json,
    )

    print(f"✅ ETL finished {datetime.utcnow().isoformat()}")


# ---- Execute only when run directly ----
if __name__ == "__main__":
    main()
