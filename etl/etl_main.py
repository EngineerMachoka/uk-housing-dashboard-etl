# etl/etl_main.py
# Full ETL script for UK Land Registry Price Paid Data -> Google Sheets
# Python 3.11+ recommended
# Every line is fully commented to explain its purpose

# ---------- STANDARD LIBRARY IMPORTS ----------
import os                                 # interact with environment variables and filesystem paths
import io                                 # in-memory file operations, e.g., wrap CSV string for pandas
import json                               # parse JSON strings (e.g., GCP service account JSON)
from datetime import datetime             # generate timestamps for logging and ETL runs

# ---------- THIRD-PARTY IMPORTS ----------
import requests                           # HTTP client to download CSV files from a URL
import pandas as pd                       # data manipulation, reading CSVs, aggregation
import numpy as np                        # numerical operations, arrays, NaN handling

# ---------- GOOGLE API IMPORTS ----------
from google.oauth2.service_account import Credentials   # convert GCP service account JSON into credentials object
from googleapiclient.discovery import build            # construct Google Sheets API service client

# ---------- CONFIGURATION ----------
LAND_REGISTRY_CSV_URL = "https://landregistry.data.gov.uk/app/uploads/pp-complete.csv"
# Official HTTPS-hosted full UK Price Paid Data CSV (monthly complete file)
# Avoids S3 403 issues and does not require authentication

GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
# Google Sheet ID to write to; must be set as environment variable or GitHub secret

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",   # allows reading/writing to Sheets
    "https://www.googleapis.com/auth/drive"           # allows clearing tabs and general Drive access
]

# ---------- FUNCTION: DOWNLOAD LAND REGISTRY CSV ----------
def download_land_registry():
    """
    Downloads the full UK Price Paid dataset CSV from official HTTPS host.
    Returns:
        pandas.DataFrame containing all rows from the CSV.
    """
    # Log the start of download for visibility
    print("Downloading Land Registry CSV from official hosted URL...")

    # Send HTTP GET request to the official CSV URL
    r = requests.get(LAND_REGISTRY_CSV_URL, timeout=300)  # 5-minute timeout for large CSVs

    # Raise HTTPError if response code is not 2xx (handles 403, 404, etc.)
    r.raise_for_status()

    # Decode the content from bytes to UTF-8 string
    text = r.content.decode("utf-8")

    # Parse CSV string into pandas DataFrame
    df = pd.read_csv(io.StringIO(text), low_memory=False)

    # Log number of rows downloaded
    print(f"✅ Downloaded {len(df):,} rows from Land Registry CSV at {LAND_REGISTRY_CSV_URL}")

    # Return the parsed DataFrame
    return df

# ---------- FUNCTION: PREPARE TRANSACTIONS ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Cleans raw Price Paid DataFrame and aggregates to weekly counts per Local Authority.
    Args:
        df_raw: pandas DataFrame loaded from the CSV
        postcode_lookup_path: optional CSV mapping postcode -> local authority
    Returns:
        pandas DataFrame with columns: week, local_authority, transactions
    """
    df = df_raw.copy()  # make a copy to avoid mutating the original DataFrame

    # Detect first column containing 'date' in its name (e.g., 'date_of_transfer')
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError("No date column in Price Paid CSV. Cannot continue ETL.")

    # Convert the date column to pandas datetime; invalid parsing will produce NaT
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')

    # Drop rows where date parsing failed (NaT)
    df = df.dropna(subset=['date'])

    # Detect a unique transaction identifier column; if missing, create one
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower() == 'transactionuniqueidentifier'), None)
    if id_col:
        df['transaction_id'] = df[id_col]  # use existing unique identifier
    else:
        df['transaction_id'] = np.arange(len(df))  # create synthetic ID using row index

    # Detect a postcode column and normalize values (uppercase, remove spaces)
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    else:
        df['postcode'] = None  # create column if source has no postcode

    # Merge with optional postcode -> local authority lookup if provided
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)  # load lookup CSV
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        df = df.merge(
            la[['pc_nospace', 'local_authority']],
            left_on='postcode',
            right_on='pc_nospace',
            how='left'
        )
    else:
        # Fallback: first 4 characters of postcode as a rough local authority bucket
        df['local_authority'] = df['postcode'].str[:4]

    # Compute the week start date for each transaction
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Aggregate unique transaction counts per week and Local Authority
    weekly = df.groupby(['week', 'local_authority']).agg(
        transactions=('transaction_id', 'nunique')
    ).reset_index()

    # Sort the result for deterministic ordering
    weekly = weekly.sort_values(['local_authority', 'week'])

    return weekly  # return the weekly aggregated DataFrame

# ---------- FUNCTION: COMPUTE ROLLING WINDOWS ----------
def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    Computes rolling sum of transactions per Local Authority for specified week windows.
    Args:
        weekly_df: DataFrame with week, local_authority, transactions
        windows: list of integers representing rolling window sizes (weeks)
    Returns:
        pandas DataFrame with rolling sums and window annotations
    """
    # Ensure week column is datetime for correct indexing
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])

    outputs = []  # accumulator for results of all windows

    # List of unique Local Authorities
    las = weekly_df['local_authority'].dropna().unique()

    # Continuous week range from earliest to latest
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')

    # Create full grid (all weeks x all Local Authorities)
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week', 'local_authority'])
    full_df = pd.DataFrame(index=full_idx).reset_index()

    # Merge actual weekly data onto full grid and fill missing values with 0
    merged = full_df.merge(weekly_df, on=['week', 'local_authority'], how='left').fillna(0)

    # Ensure sort order is correct for rolling calculation
    merged = merged.sort_values(['local_authority', 'week'])

    # Compute rolling sums for each requested window
    for w in windows:
        m = merged.copy()  # copy working DataFrame for current window
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(
            lambda s: s.rolling(w, min_periods=1).sum()  # rolling sum with min_periods=1
        )
        m['window_weeks'] = w  # annotate current window size
        outputs.append(m[['week', 'local_authority', 'rolling_trans', 'transactions', 'window_weeks']])

    # Concatenate all windows into one DataFrame
    return pd.concat(outputs, ignore_index=True)

# ---------- FUNCTION: WRITE TO GOOGLE SHEETS ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Writes multiple DataFrames to separate tabs in a Google Sheet.
    Args:
        dfs_by_tab: dictionary mapping tab_name -> DataFrame
        gcp_service_account_json: parsed GCP service account JSON
    """
    # Build credentials from service account JSON
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)

    # Create Google Sheets API service client
    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()  # reference spreadsheets resource

    # Iterate through each tab and write data
    for tab, df in dfs_by_tab.items():
        # Convert DataFrame to list-of-lists including header row
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()

        body = {"values": values}  # API payload
        range_name = f"{tab}!A1"   # write starting at top-left cell of tab

        # Clear existing content on tab
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()

        # Upload new values to tab
        sheet.values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()

# ---------- MAIN ETL FLOW ----------
def main():
    """
    Full ETL workflow:
    1. Download Land Registry Price Paid CSV
    2. Aggregate weekly transactions per Local Authority
    3. Compute 4-week and 12-week rolling sums
    4. Write data to Google Sheets tabs
    """
    run_ts = datetime.utcnow().isoformat()  # capture start timestamp
    print("ETL start", run_ts)

    # Step 1: download CSV
    df_raw = download_land_registry()

    # Step 2: prepare weekly aggregated transactions
    postcode_lookup = "lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None
    weekly = prepare_transactions(df_raw, postcode_lookup_path=postcode_lookup)

    # Step 3: compute rolling windows
    windows_df = compute_rolling_windows(weekly, windows=[4, 12])

    # Step 4: extract latest week snapshot
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # Load GCP service account JSON from environment variable
    gcp_sa_text = os.environ.get('GCP_SA_JSON')
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Add it as a GitHub repository secret.")
    gcp_json = json.loads(gcp_sa_text)  # parse JSON into dictionary

    # Validate Google Sheet ID
    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID environment variable not set.")

    # Step 5: write all DataFrames to Google Sheets
    write_to_google_sheets({
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }, gcp_json)

    print("ETL finished", datetime.utcnow().isoformat())

# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    main()
