# etl/etl_main.py
# Full ETL script: Land Registry Price Paid -> Google Sheets
# Uses the currently working S3 URL to download the latest CSV.
# Python 3.11+ recommended.
# Every line has detailed comments explaining its purpose.

# ---------- standard library imports ----------
import os                                # interact with environment variables and file system
import io                                # create in-memory text streams for pandas CSV reader
import json                              # parse JSON strings (service account JSON)
from datetime import datetime            # capture timestamps for logging

# ---------- third-party imports ----------
import requests                          # HTTP client for downloads
import pandas as pd                      # data manipulation, CSV reading/writing
import numpy as np                       # numeric utilities (nunique, NaN handling)

# ---------- Google API imports ----------
from google.oauth2.service_account import Credentials   # create credentials from service account JSON
from googleapiclient.discovery import build            # build Google Sheets API client

# ---------- Configuration ----------
# Stable S3 URL for the latest Price Paid CSV
LAND_REGISTRY_CSV_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
    "pp-monthly-update-new-version.csv"
)

# Google Sheets target: set this in GitHub secrets as GOOGLE_SHEET_ID
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # ID of the Google Sheet to write to (required)

# OAuth scopes required to write to Sheets and access Drive (for clearing tabs)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ---------- Download function ----------
def download_land_registry():
    """
    Download the latest Price Paid CSV from the working S3 public URL.
    Returns a pandas DataFrame containing all rows.
    """
    print(f"Attempting to download Land Registry CSV from {LAND_REGISTRY_CSV_URL} ...")
    try:
        # Make HTTP GET request with generous timeout (300s) for large files
        r = requests.get(LAND_REGISTRY_CSV_URL, timeout=300)
        r.raise_for_status()  # Raise exception for HTTP errors
        text = r.content.decode("utf-8")  # Decode bytes to text (UTF-8)
        df = pd.read_csv(io.StringIO(text), low_memory=False)  # Load CSV into pandas DataFrame
        print(f"✅ Downloaded {len(df):,} rows from Land Registry CSV.")
        return df
    except Exception as e:
        # Raise a runtime error with helpful context if download fails
        raise RuntimeError(f"Failed to download Land Registry CSV from {LAND_REGISTRY_CSV_URL}: {e}")

# ---------- Transformation: prepare transactions ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Clean the raw Price Paid DataFrame and aggregate to weekly counts by Local Authority.
    - df_raw: pandas DataFrame loaded from CSV
    - postcode_lookup_path: optional CSV path mapping postcode -> local_authority
    Returns a DataFrame with columns: week, local_authority, transactions
    """
    df = df_raw.copy()  # avoid modifying the original DataFrame

    # Identify a date column (e.g., 'date_of_transfer')
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError("No date column found in Price Paid CSV.")

    # Parse date column into pandas datetime format; invalid dates become NaT
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date'])  # drop rows with invalid dates

    # Detect or create a unique transaction identifier
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower() == 'transactionuniqueidentifier'), None)
    if id_col:
        df['transaction_id'] = df[id_col]
    else:
        df['transaction_id'] = np.arange(len(df))  # synthetic IDs

    # Normalize postcode column: remove spaces and uppercase
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    else:
        df['postcode'] = None

    # Map postcode to Local Authority using lookup CSV if available
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        df = df.merge(la[['pc_nospace', 'local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        # Fallback: use first 4 chars of postcode as rough Local Authority code
        df['local_authority'] = df['postcode'].str[:4]

    # Compute week-start date for aggregation
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Aggregate weekly unique transaction counts by Local Authority
    weekly = df.groupby(['week', 'local_authority']).agg(transactions=('transaction_id', 'nunique')).reset_index()
    weekly = weekly.sort_values(['local_authority', 'week'])  # sort for deterministic output
    return weekly

# ---------- Transformation: compute rolling windows ----------
def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    Compute rolling sum of transactions per Local Authority for given window sizes.
    Returns a DataFrame with columns: week, local_authority, rolling_trans, transactions, window_weeks
    """
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])  # ensure datetime dtype
    outputs = []

    las = weekly_df['local_authority'].dropna().unique()  # unique LAs
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week', 'local_authority'])
    full_df = pd.DataFrame(index=full_idx).reset_index()

    merged = full_df.merge(weekly_df, on=['week', 'local_authority'], how='left').fillna(0)
    merged = merged.sort_values(['local_authority', 'week'])

    for w in windows:
        m = merged.copy()
        # Rolling sum per Local Authority; min_periods=1 ensures start-of-series is not NaN
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        m['window_weeks'] = w
        outputs.append(m[['week', 'local_authority', 'rolling_trans', 'transactions', 'window_weeks']])

    return pd.concat(outputs, ignore_index=True)

# ---------- Write to Google Sheets ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write a dictionary of DataFrames to a Google Sheet.
    - dfs_by_tab: dict mapping sheet tab name -> DataFrame
    - gcp_service_account_json: parsed JSON dict of service account credentials
    """
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    for tab, df in dfs_by_tab.items():
        # Prepare values: header + rows; convert NaN to empty strings
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()
        range_name = f"{tab}!A1"
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body={"values": values}).execute()

# ---------- Main ETL flow ----------
def main():
    """
    Main ETL pipeline:
    1. Download Price Paid CSV from S3
    2. Aggregate weekly transactions per Local Authority
    3. Compute 4-week and 12-week rolling windows
    4. Write results to Google Sheets
    """
    run_ts = datetime.utcnow().isoformat()
    print("ETL start", run_ts)

    # 1) Download CSV
    df_raw = download_land_registry()

    # 2) Transform CSV -> weekly transactions
    postcode_lookup = "lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None
    weekly = prepare_transactions(df_raw, postcode_lookup_path=postcode_lookup)

    # 3) Compute rolling windows
    windows_df = compute_rolling_windows(weekly, windows=[4, 12])

    # 4) Latest snapshot
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # 5) Load GCP credentials
    gcp_sa_text = os.environ.get('GCP_SA_JSON')
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Set as GitHub secret.")
    gcp_json = json.loads(gcp_sa_text)

    # 6) Validate Sheet ID
    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID environment variable is not set.")

    # 7) Write to Google Sheets
    write_to_google_sheets({
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }, gcp_json)

    print("ETL finished", datetime.utcnow().isoformat())

# Run main when script is executed
if __name__ == "__main__":
    main()
