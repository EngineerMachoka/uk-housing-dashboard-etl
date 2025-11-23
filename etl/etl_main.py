# etl/etl_main.py
# Full ETL script: Land Registry Price Paid -> Google Sheets
# Uses the official Land Registry TXT file to avoid CSV parsing issues
# Python 3.11+ recommended
# All lines include explanatory comments

# ---------- standard library imports ----------
import os                                # interact with environment variables and file system
import io                                # create in-memory text streams for pandas CSV reader
import json                              # parse JSON strings (service account JSON)
from datetime import datetime            # create timestamps for logs

# ---------- third-party imports ----------
import requests                          # HTTP client for downloads
import pandas as pd                      # data manipulation, CSV reading/writing
import numpy as np                       # numeric utilities (nunique, NaN handling)

# ---------- Google API imports ----------
from google.oauth2.service_account import Credentials   # create credentials from service account JSON
from googleapiclient.discovery import build            # build Google Sheets API client

# ---------- Configuration ----------
# Official Land Registry TXT URL for Price Paid Data
LAND_REGISTRY_TXT_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.txt"
)

# Optional uploaded file path (from previous conversation, for reference)
UPLOADED_FILE_PATH = "/mnt/data/3e7adebc-632c-4a8e-8eb7-ad753a2fb041.png"

# Google Sheets target: set this in GitHub secrets as GOOGLE_SHEET_ID
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # ID of the Google Sheet to write to (required)

# OAuth scopes required to write to Sheets and access Drive
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ---------- Download function ----------
def download_land_registry_txt():
    """
    Download the Price Paid TXT file from Land Registry and return as pandas DataFrame.
    Handles common encoding and blank-line issues.
    """
    print(f"Attempting to download Land Registry TXT from {LAND_REGISTRY_TXT_URL} ...")
    try:
        # Perform HTTP GET with generous timeout for large file
        r = requests.get(LAND_REGISTRY_TXT_URL, timeout=300)
        r.raise_for_status()  # raise exception for HTTP errors

        # Decode bytes with utf-8 and ignore decoding errors
        text = r.content.decode("utf-8", errors="ignore")

        # Read as CSV (comma-separated), skipping blank lines
        df = pd.read_csv(io.StringIO(text), sep=",", low_memory=False, skip_blank_lines=True)

        # Log the number of rows downloaded
        print(f"✅ Downloaded {len(df):,} rows from Land Registry TXT.")
        return df

    except Exception as e:
        # Raise detailed error if download fails
        raise RuntimeError(f"Failed to download Land Registry TXT: {e}")

# ---------- Transformation: prepare transactions ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Clean raw Price Paid DataFrame and aggregate to weekly counts by Local Authority.
    - df_raw: pandas DataFrame loaded from TXT
    - postcode_lookup_path: optional path to postcode->LA lookup CSV
    Returns weekly aggregated DataFrame with columns: week, local_authority, transactions
    """
    df = df_raw.copy()  # avoid mutating original DataFrame

    # Detect the first column containing 'date'
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError("No date column found in Price Paid TXT. Cannot continue ETL.")

    # Convert the date column to pandas datetime
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date'])  # drop rows where date parsing failed

    # Detect or create a unique transaction identifier
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower() == 'transactionuniqueidentifier'), None)
    if id_col:
        df['transaction_id'] = df[id_col]  # use existing unique identifier
    else:
        df['transaction_id'] = np.arange(len(df))  # synthetic ID if missing

    # Normalize postcode field
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    else:
        df['postcode'] = None  # if missing

    # Merge with postcode->local_authority lookup if provided
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        df = df.merge(la[['pc_nospace', 'local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        # fallback: use first 4 chars of postcode as rough LA bucket
        df['local_authority'] = df['postcode'].str[:4]

    # Compute week-start for each transaction
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Aggregate unique transaction counts per week and local authority
    weekly = df.groupby(['week', 'local_authority']).agg(transactions=('transaction_id', 'nunique')).reset_index()
    weekly = weekly.sort_values(['local_authority', 'week'])

    return weekly

# ---------- Transformation: compute rolling windows ----------
def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    Compute rolling sum of transactions per Local Authority for specified window sizes.
    Returns a concatenated DataFrame with columns: week, local_authority, rolling_trans, transactions, window_weeks
    """
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])
    outputs = []

    # Get unique Local Authorities
    las = weekly_df['local_authority'].dropna().unique()

    # Build continuous week range
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')

    # Full cartesian product of weeks x LAs
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week', 'local_authority'])
    full_df = pd.DataFrame(index=full_idx).reset_index()

    # Merge actual data; fill missing transactions with 0
    merged = full_df.merge(weekly_df, on=['week', 'local_authority'], how='left').fillna(0)
    merged = merged.sort_values(['local_authority', 'week'])

    for w in windows:
        m = merged.copy()
        # Rolling sum per local authority
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        m['window_weeks'] = w
        outputs.append(m[['week', 'local_authority', 'rolling_trans', 'transactions', 'window_weeks']])

    return pd.concat(outputs, ignore_index=True)

# ---------- Write DataFrames to Google Sheets ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write multiple DataFrames to a Google Sheet.
    - dfs_by_tab: dict mapping tab name -> pandas DataFrame
    - gcp_service_account_json: parsed GCP service account JSON dict
    """
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    for tab, df in dfs_by_tab.items():
        # Prepare header + data rows
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()
        body = {"values": values}
        range_name = f"{tab}!A1"

        # Clear the tab first
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()

        # Update the tab with new values
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()

# ---------- Main ETL flow ----------
def main():
    """
    Main ETL process:
    1) Download Land Registry TXT
    2) Prepare weekly transaction counts by LA
    3) Compute rolling windows (4wk/12wk)
    4) Write results to Google Sheets
    """
    run_ts = datetime.utcnow().isoformat()
    print("ETL start", run_ts)

    # Download the TXT
    df_raw = download_land_registry_txt()

    # Transform to weekly counts
    postcode_lookup = "lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None
    weekly = prepare_transactions(df_raw, postcode_lookup_path=postcode_lookup)

    # Compute rolling windows
    windows_df = compute_rolling_windows(weekly, windows=[4, 12])

    # Latest snapshot
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # Load GCP service account JSON from environment
    gcp_sa_text = os.environ.get('GCP_SA_JSON')
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Set it as GitHub secret with service account JSON.")
    gcp_json = json.loads(gcp_sa_text)

    # Validate Google Sheet ID
    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID not set. Add as environment variable or GitHub secret.")

    # Write all DataFrames to Google Sheets
    write_to_google_sheets({
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }, gcp_json)

    print("ETL finished", datetime.utcnow().isoformat())

# Run main when executed
if __name__ == "__main__":
    main()
