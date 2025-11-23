# etl/etl_main.py
# Full ETL script: Land Registry Price Paid -> Google Sheets
# Supports HTTPS official download with automatic fallback to older S3 URL
# Python 3.11+ recommended
# Detailed comments included for every line

# ---------- standard library imports ----------
import os                                # interact with environment variables
import io                                # create in-memory streams for CSV reading
import json                              # parse JSON strings (service account JSON)
from datetime import datetime            # generate timestamps for logging

# ---------- third-party imports ----------
import requests                          # HTTP requests for CSV download
import pandas as pd                      # data manipulation and CSV parsing
import numpy as np                       # numeric utilities (unique counts, NaN handling)

# ---------- Google API imports ----------
from google.oauth2.service_account import Credentials   # build credentials from service account JSON
from googleapiclient.discovery import build            # build Google Sheets API service client

# ---------- Configuration ----------
# Official Land Registry CSV URL (HTTPS) – recommended
LAND_REGISTRY_CSV_URL = "https://landregistry.data.gov.uk/app/uploads/pp-complete.csv"

# Fallback S3 URL (HTTP) if the official HTTPS fails
LAND_REGISTRY_S3_FALLBACK = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv"

# Local uploaded file path (optional reference)
UPLOADED_FILE_PATH = "/mnt/data/3e7adebc-632c-4a8e-8eb7-ad753a2fb041.png"

# Google Sheet ID (set as GitHub secret or environment variable)
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")

# OAuth scopes needed for Google Sheets API
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ---------- Download function with fallback ----------
def download_land_registry():
    """
    Download the Land Registry Price Paid dataset CSV.
    - Tries official HTTPS link first.
    - Falls back to older S3 URL if HTTPS fails.
    Returns a pandas DataFrame.
    """
    urls_to_try = [LAND_REGISTRY_CSV_URL, LAND_REGISTRY_S3_FALLBACK]
    last_exception = None

    for url in urls_to_try:
        try:
            print(f"Attempting to download Land Registry CSV from {url}...")
            r = requests.get(url, timeout=300)   # generous timeout for large file
            r.raise_for_status()                 # raise exception if HTTP status is not OK
            text = r.content.decode("utf-8")     # decode bytes to string
            df = pd.read_csv(io.StringIO(text), low_memory=False)  # parse CSV into DataFrame

            # check that DataFrame has columns (avoid empty downloads)
            if df.empty or len(df.columns) == 0:
                raise RuntimeError(f"Downloaded CSV from {url} is empty or invalid.")

            print(f"✅ Successfully downloaded {len(df):,} rows from {url}")
            return df

        except Exception as e:
            print(f"⚠️ Failed to download from {url}: {e}")
            last_exception = e
            continue

    # If all URLs fail, raise last exception
    raise RuntimeError(f"All download attempts failed. Last error: {last_exception}")

# ---------- Transformation helpers ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Clean the raw CSV and aggregate transactions weekly by Local Authority.
    """
    df = df_raw.copy()  # avoid modifying original
    # detect date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError("No date column in CSV.")

    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date'])

    # detect unique transaction id or create one
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower()=='transactionuniqueidentifier'), None)
    df['transaction_id'] = df[id_col] if id_col else np.arange(len(df))

    # normalize postcode
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)
    df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper() if pc_col else None

    # map postcode to local authority
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        df = df.merge(la[['pc_nospace', 'local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        df['local_authority'] = df['postcode'].str[:4]  # fallback

    # compute week start
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # aggregate transactions by week and LA
    weekly = df.groupby(['week', 'local_authority']).agg(transactions=('transaction_id', 'nunique')).reset_index()
    weekly = weekly.sort_values(['local_authority', 'week'])

    return weekly

def compute_rolling_windows(weekly_df, windows=[4,12]):
    """
    Compute rolling sums of transactions per Local Authority for given window sizes.
    Returns concatenated DataFrame with rolling sums.
    """
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])
    outputs = []
    las = weekly_df['local_authority'].dropna().unique()
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week','local_authority'])
    full_df = pd.DataFrame(index=full_idx).reset_index()
    merged = full_df.merge(weekly_df, on=['week','local_authority'], how='left').fillna(0)
    merged = merged.sort_values(['local_authority','week'])

    for w in windows:
        m = merged.copy()
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        m['window_weeks'] = w
        outputs.append(m[['week','local_authority','rolling_trans','transactions','window_weeks']])

    return pd.concat(outputs, ignore_index=True)

# ---------- Google Sheets writer ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write multiple DataFrames to a Google Sheet using a service account.
    """
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)
    service = build('sheets','v4',credentials=creds)
    sheet = service.spreadsheets()

    for tab, df in dfs_by_tab.items():
        values = [df.columns.tolist()] + df.replace({np.nan:''}).astype(str).values.tolist()
        body = {"values": values}
        range_name = f"{tab}!A1"
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()

# ---------- Main ETL flow ----------
def main():
    """
    Main ETL:
    - download Land Registry CSV (HTTPS + fallback)
    - aggregate weekly transactions by Local Authority
    - compute 4wk and 12wk rolling windows
    - write all results to Google Sheets
    """
    run_ts = datetime.utcnow().isoformat()
    print("ETL start", run_ts)

    # Download CSV
    df_raw = download_land_registry()

    # Transform
    postcode_lookup = "lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None
    weekly = prepare_transactions(df_raw, postcode_lookup_path=postcode_lookup)
    windows_df = compute_rolling_windows(weekly, windows=[4,12])
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # Load GCP service account JSON from environment
    gcp_sa_text = os.environ.get('GCP_SA_JSON')
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Set it as GitHub secret.")
    gcp_json = json.loads(gcp_sa_text)

    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID not set.")

    # Write to Sheets
    write_to_google_sheets({
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }, gcp_json)

    print("ETL finished", datetime.utcnow().isoformat())

if __name__ == "__main__":
    main()
