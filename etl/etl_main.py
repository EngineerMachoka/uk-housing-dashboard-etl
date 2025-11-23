# etl/etl_main.py
# Full ETL script: Land Registry Price Paid -> Google Sheets
# Uses the official HTTPS hosted CSV to avoid S3 403 problems.
# Python 3.11+ recommended.
# All lines include explanatory comments.

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
# Official, stable HTTPS URL for the full Price Paid Data CSV (monthly complete file)
LAND_REGISTRY_CSV_URL = "https://landregistry.data.gov.uk/app/uploads/pp-complete.csv"  # Hosted by Land Registry

# Local uploaded file path present in the conversation history (image). Included per developer instruction.
# NOTE: this is an image path from the conversation and is not used by the ETL download.
UPLOADED_FILE_PATH = "/mnt/data/3e7adebc-632c-4a8e-8eb7-ad753a2fb041.png"  # path to uploaded screenshot (for reference)

# Google Sheets target: set this in GitHub secrets as GOOGLE_SHEET_ID
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # ID of the Google Sheet to write to (required)

# OAuth scopes required to write to Sheets and access Drive (for clearing tabs)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ---------- Download function ----------
def download_land_registry():
    """
    Download the Land Registry 'pp-complete.csv' from the official HTTPS host and return a pandas DataFrame.
    This avoids the old S3 URL that returned 403. The function streams and decodes content into pandas.
    """
    # Informational log that download is starting
    print("Downloading Land Registry CSV from official hosted URL...")

    # Use the official HTTPS URL defined in configuration
    url = LAND_REGISTRY_CSV_URL  # the stable CSV location on landregistry.data.gov.uk

    # Perform HTTP GET with a generous timeout for large file download
    r = requests.get(url, timeout=300)  # stream=False by default; content read below
    # Raise an exception if the HTTP status is not 200 OK (or other 2xx)
    r.raise_for_status()

    # Decode bytes to text assuming UTF-8 (Land Registry CSV is UTF-8)
    text = r.content.decode("utf-8")

    # Wrap the CSV text in an in-memory text stream and parse it with pandas
    df = pd.read_csv(io.StringIO(text), low_memory=False)

    # Log the number of rows downloaded for visibility in CI logs
    print(f"✅ Downloaded {len(df):,} rows from Land Registry CSV at {url}")

    # Return the parsed DataFrame to the caller
    return df

# ---------- Transformation helpers ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Clean the raw Price Paid DataFrame and aggregate to weekly per Local Authority.
    - df_raw: pandas DataFrame loaded from the CSV
    - postcode_lookup_path: optional CSV path mapping postcode -> local_authority
    Returns a DataFrame with columns: week (datetime), local_authority, transactions
    """
    # Copy input to avoid mutating caller's object
    df = df_raw.copy()

    # Find a column that contains 'date' in its name (e.g., 'date_of_transfer')
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        # If no date column found, abort with helpful message
        raise RuntimeError("No date column in Price Paid CSV. Cannot continue ETL.")

    # Parse the date column into pandas datetime; invalid parse becomes NaT
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')

    # Drop rows that failed to parse into a date
    df = df.dropna(subset=['date'])

    # Attempt to detect an existing unique transaction identifier column; otherwise create one
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower() == 'transactionuniqueidentifier'), None)
    if id_col:
        # Use the vendor-supplied unique id if available
        df['transaction_id'] = df[id_col]
    else:
        # Create a synthetic unique id using the row index if none exists
        df['transaction_id'] = np.arange(len(df))

    # Normalize postcode field if present: remove whitespace and uppercase
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)
    if pc_col:
        # Remove spaces and uppercase to create normalized postcodes
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    else:
        # Ensure postcode column exists even if source lacks it
        df['postcode'] = None

    # If a postcode->local_authority lookup is available, use it for accurate mapping
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        # Read the lookup file (expects 'postcode' and 'local_authority' columns)
        la = pd.read_csv(postcode_lookup_path, dtype=str)
        # Normalize lookup postcodes to the same no-space uppercase format
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        # Merge the lookup into the main dataframe to obtain 'local_authority'
        df = df.merge(la[['pc_nospace', 'local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        # Fallback: use the first 4 characters of postcode as a rough proxy bucket (not precise)
        df['local_authority'] = df['postcode'].str[:4]

    # Compute the week-start timestamp for each transaction (use period start)
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Aggregate to weekly unique transaction counts per local authority
    weekly = df.groupby(['week', 'local_authority']).agg(transactions=('transaction_id', 'nunique')).reset_index()

    # Sort the result for deterministic order
    weekly = weekly.sort_values(['local_authority', 'week'])

    # Return the aggregated weekly DataFrame
    return weekly

def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    For each requested window size, compute the rolling sum of transactions per Local Authority.
    Returns a concatenated DataFrame with columns: week, local_authority, rolling_trans, transactions, window_weeks
    """
    # Ensure the 'week' column is datetime for constructing ranges and rolling calculations
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])

    # Prepare an accumulator list for each window-sized DataFrame
    outputs = []

    # Unique list of Local Authorities present in the data
    las = weekly_df['local_authority'].dropna().unique()

    # Build a continuous list of week-start dates covering the whole range of data
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')

    # Create a full cartesian product index of all weeks x all LAs so we have rows for missing weeks
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week', 'local_authority'])
    full_df = pd.DataFrame(index=full_idx).reset_index()

    # Merge the actual weekly data onto the full grid and fill missing transaction counts with 0
    merged = full_df.merge(weekly_df, on=['week', 'local_authority'], how='left').fillna(0)

    # Ensure correct sort order for groupby+rolling
    merged = merged.sort_values(['local_authority', 'week'])

    # For each requested window size compute rolling sums
    for w in windows:
        # Make a working copy for this window
        m = merged.copy()
        # Compute rolling sum per local_authority with min_periods=1 to avoid NaNs at series start
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        # Annotate which window this row represents
        m['window_weeks'] = w
        # Select the output columns and add to outputs list
        outputs.append(m[['week', 'local_authority', 'rolling_trans', 'transactions', 'window_weeks']])

    # Concatenate the per-window results into a single long DataFrame and return
    return pd.concat(outputs, ignore_index=True)

# ---------- Google Sheets writer ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write provided DataFrames (dict of tab_name -> DataFrame) to the configured Google Sheet.
    - dfs_by_tab: dict mapping sheet tab name -> pandas DataFrame
    - gcp_service_account_json: parsed JSON dict of service account credentials
    """
    # Create Credentials object from the parsed service account JSON
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)

    # Build the Sheets API service object
    service = build('sheets', 'v4', credentials=creds)

    # Shortcut to the spreadsheets() resource
    sheet = service.spreadsheets()

    # Iterate over each tab and write it
    for tab, df in dfs_by_tab.items():
        # Prepare values: header row followed by data rows (convert NaN to empty string)
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()

        # Body payload for Sheets API update
        body = {"values": values}

        # Target range: start at A1 on the given tab
        range_name = f"{tab}!A1"

        # Clear the target tab first to remove leftover rows (prevents old data persisting)
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()

        # Update the sheet with the new values
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()

# ---------- Main ETL flow ----------
def main():
    """
    Main ETL entrypoint:
    - downloads the full Price Paid CSV from Land Registry
    - aggregates weekly transactions by LA
    - computes 4-week and 12-week rolling windows
    - writes results to Google Sheets
    """
    # Capture run timestamp for logs
    run_ts = datetime.utcnow().isoformat()
    print("ETL start", run_ts)

    # 1) Download the Land Registry CSV into a pandas DataFrame
    print("Downloading Land Registry CSV...")
    df_raw = download_land_registry()  # call the download function above (will raise on any HTTP error)

    # 2) Transform raw CSV -> weekly transactions by Local Authority
    # Use a postcode->LA lookup if present at lookups/uk_postcode_to_la.csv (optional)
    postcode_lookup = "lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None
    weekly = prepare_transactions(df_raw, postcode_lookup_path=postcode_lookup)

    # 3) Compute rolling windows (4-week and 12-week by default)
    windows_df = compute_rolling_windows(weekly, windows=[4, 12])

    # 4) Prepare 'latest' snapshot - rows matching the most recent week in the windows_df
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # 5) Load GCP service account JSON from environment (GitHub secret must set this)
    gcp_sa_text = os.environ.get('GCP_SA_JSON')
    if not gcp_sa_text:
        # If GCP_SA_JSON is missing, raise with a helpful message
        raise RuntimeError("GCP_SA_JSON missing. Set it as a GitHub repository secret containing the full service account JSON text.")
    # Parse the JSON text into a Python dict
    gcp_json = json.loads(gcp_sa_text)

    # 6) Validate Google Sheet ID is set
    if not GOOGLE_SHEET_ID:
        # Helpful error if the sheet ID isn't configured
        raise RuntimeError("GOOGLE_SHEET_ID environment variable is not set. Add it to GitHub secrets or environment variables.")

    # 7) Write the computed DataFrames to Google Sheets tabs
    write_to_google_sheets({
        "weekly_by_la": weekly,    # weekly counts by local authority
        "windows": windows_df,     # rolling windows table (4wk/12wk)
        "latest": latest           # latest snapshot per LA
    }, gcp_json)

    # 8) Log completion timestamp
    print("ETL finished", datetime.utcnow().isoformat())

# Run main when executed as a script
if __name__ == "__main__":
    main()
