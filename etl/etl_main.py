# etl/etl_main.py
# Python ETL that:
# - reads UK Land Registry Price Paid data (local file preferred, remote fallback)
# - maps postcodes to Local Authority (optional lookup)
# - aggregates weekly transactions and computes rolling windows (4wk, 12wk)
# - writes aggregated results to Google Sheets via a GCP service account JSON provided in env

# ---------- imports ----------
import os                                         # operating system utilities (file paths, env vars)
import io                                         # in-memory streams (used for reading remote CSV bytes)
import json                                       # for parsing JSON service account (from env)
from datetime import datetime                     # timestamping the job
import pandas as pd                               # dataframes and CSV handling
import numpy as np                                # numeric utilities (nunique, nan handling)
import requests                                   # HTTP requests for remote CSV
# google auth and sheets API
from google.oauth2.service_account import Credentials   # create credentials from service account json
from googleapiclient.discovery import build            # build the Sheets API client

# ---------- configuration (tweak these) ----------
# remote S3 URL for Land Registry CSV (may sometimes return 403; used as fallback)
LAND_REG_CSV_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update/latest/pp-complete.csv"

# local path fallback: (1) environment variable LOCAL_LAND_REG_CSV_PATH can point here,
# (2) if not set, script will look for this path inside the repo/workspace
LOCAL_LAND_REG_CSV_DEFAULT = "data/pp-complete.csv"

# optional lookup CSV mapping UK postcodes to Local Authority (used to bucket postcodes)
POSTCODE_TO_LA_CSV = "lookups/uk_postcode_to_la.csv"

# Google Sheet ID where processed tabs will be written. Replace or set via env var.
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "YOUR_GOOGLE_SHEET_ID")

# OAuth scopes required for writing updates to Google Sheets and Drive
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ---------- helper functions ----------

def download_land_registry(local_path_hint=None, remote_url=LAND_REG_CSV_URL):
    """
    Try to load the Land Registry Price Paid CSV from a local path first (recommended).
    If the local file is absent, attempt a remote HTTP GET to the known S3 URL.
    Raise a clear error if neither option works.
    """
    # 1) Check environment variable (explicit)
    #    The env var allows GitHub Actions to tell the script where the local CSV is located.
    local_env = os.environ.get("LOCAL_LAND_REG_CSV_PATH")
    # 2) Determine the candidate local path order:
    #    a) env var, b) local_path_hint (function arg), c) default path in repo workspace
    candidates = []
    if local_env:
        candidates.append(local_env)
    if local_path_hint:
        candidates.append(local_path_hint)
    candidates.append(LOCAL_LAND_REG_CSV_DEFAULT)

    # Iterate candidates and return the first file that exists
    for p in candidates:
        if p and os.path.exists(p):
            print(f"[download_land_registry] Found local CSV at: {p} — loading from disk.")
            # read the CSV using pandas and return
            df_local = pd.read_csv(p, low_memory=False)
            return df_local

    # If we reach here, no local file found — attempt remote download
    print("[download_land_registry] No local CSV found in candidates:", candidates)
    print(f"[download_land_registry] Attempting remote download from: {remote_url}")
    try:
        r = requests.get(remote_url, stream=True, timeout=60)
        # raise_for_status will throw an HTTPError if the response is not 2xx
        r.raise_for_status()
        # decode bytes to text and read with pandas
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), low_memory=False)
        print("[download_land_registry] Successfully downloaded remote CSV.")
        return df
    except requests.exceptions.HTTPError as e:
        # remote returned non-2xx (403 in your logs). Provide detailed guidance.
        print("[download_land_registry] HTTP error while downloading remote CSV:", e)
        # raise a friendly error explaining the recommended fix (upload CSV to repo or use env var)
        raise RuntimeError(
            "Failed to download Land Registry CSV from remote URL (HTTP error). "
            "Recommended action: upload pp-complete.csv into the repo at data/pp-complete.csv "
            "or set the LOCAL_LAND_REG_CSV_PATH environment variable to a valid CSV path. "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        # catch-all for connection/timeouts/decoding errors
        print("[download_land_registry] Unexpected error downloading remote CSV:", e)
        raise

def prepare_transactions(df_raw, join_postcode_to_la=True):
    """
    Prepare and aggregate transaction-level CSV into weekly counts by Local Authority.
    Steps:
      - normalize date column
      - create transaction_id if none provided
      - clean postcode (upper-case, drop spaces)
      - optionally join to a postcode->local_authority lookup CSV
      - compute 'week' (week-start timestamp) and aggregate counts
    """
    # make a copy to avoid modifying original DataFrame in-place
    df = df_raw.copy()

    # find a date column (Price Paid has a "date" or "date_of_transfer" style column).
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise RuntimeError("[prepare_transactions] No date column detected in Land Registry CSV.")
    # parse the date column into pandas datetime
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    # drop any rows where date parsing failed
    df = df.dropna(subset=['date'])

    # attempt to identify a unique transaction id column; fallback to synthetic index if none
    id_col = None
    for c in df.columns:
        if "unique" in c.lower() or c.lower() in ("transaction_unique_identifier", "transactionid", "id"):
            id_col = c
            break
    if id_col is None:
        # create synthetic transaction id to allow unique counts
        df['transaction_id'] = np.arange(len(df))
    else:
        # copy the existing unique identifier column into 'transaction_id'
        df['transaction_id'] = df[id_col]

    # Attempt to find a postcode column and normalise it (remove whitespace and uppercase)
    pc_col = next((c for c in df.columns if "postcode" in c.lower()), None)
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    else:
        # if no postcode provided, keep a null column to avoid KeyErrors later
        df['postcode'] = None

    # If a postcode->local-authority lookup exists, join on normalized postcode
    if join_postcode_to_la and os.path.exists(POSTCODE_TO_LA_CSV):
        print("[prepare_transactions] Joining to postcode->local_authority lookup:", POSTCODE_TO_LA_CSV)
        la = pd.read_csv(POSTCODE_TO_LA_CSV, dtype=str)
        # create matching no-space uppercase version on the lookup file
        if 'postcode' in la.columns:
            la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+','',regex=True).str.upper()
        else:
            # If lookup file uses a different column name change logic appropriately
            raise RuntimeError("[prepare_transactions] Postcode lookup missing 'postcode' column.")
        # merge the lookup into df on the normalized postcode
        df = df.merge(la[['pc_nospace','local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        # fallback: use the first 4 characters of postcode as an approximate bucket
        # (rough and not recommended for final analyses)
        print("[prepare_transactions] No postcode->LA lookup found; using postcode prefix as rough bucket.")
        df['local_authority'] = df['postcode'].str[:4]

    # compute week start (period start time) to group transactions weekly
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # aggregate unique transaction counts per week x local_authority
    weekly = df.groupby(['week','local_authority']).agg(
        transactions = ('transaction_id','nunique')  # unique transactions in the week & LA
    ).reset_index()

    # ensure a stable sort order
    weekly = weekly.sort_values(['local_authority','week'])
    return weekly

def compute_windows(df_weekly, windows=[4,12,52]):
    """
    Given weekly transaction counts by LA, compute rolling sums for each window size.
    Returns a long DataFrame with columns: week, local_authority, rolling_trans, transactions, window_weeks
    """
    df = df_weekly.copy()
    outputs = []
    # ensure week column is datetime
    df['week'] = pd.to_datetime(df['week'])
    for w in windows:
        # prepare a complete index of all weeks x all local_authorities to avoid missing week rows
        all_weeks = pd.date_range(df['week'].min(), df['week'].max(), freq='W-MON')
        las = df['local_authority'].dropna().unique()
        full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week','local_authority'])
        full_df = pd.DataFrame(index=full_idx).reset_index()
        # merge the real data onto the full index, filling missing transaction counts with 0
        merged = full_df.merge(df, on=['week','local_authority'], how='left').fillna(0)
        # sort before applying rolling
        merged = merged.sort_values(['local_authority','week'])
        # compute rolling sum of transactions for window w (min_periods=1 to avoid NaN at early periods)
        merged['rolling_trans'] = merged.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        merged['window_weeks'] = w
        outputs.append(merged[['week','local_authority','rolling_trans','transactions','window_weeks']])
    # concatenate results for all windows
    return pd.concat(outputs, ignore_index=True)

def write_to_google_sheets(df_dict, gcp_service_account_json):
    """
    Write a dict of DataFrames to a Google Sheet. Keys of df_dict are the tab names.
    gcp_service_account_json: parsed JSON (dict) for service account credentials.
    """
    # create Credentials object from service account JSON (in-memory)
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)
    # build the Sheets API client
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    # iterate over each DataFrame and write it to the corresponding worksheet/tab
    for tab_name, df in df_dict.items():
        # prepare the values: header row then data rows
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()
        body = {"values": values}
        range_name = f"{tab_name}!A1"
        print(f"[write_to_google_sheets] Updating sheet tab: {tab_name} with {len(values)-1} data rows.")
        # clear the existing content of the tab (so olds rows beyond new end are removed)
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab_name).execute()
        # update the sheet starting at A1
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()

# ---------- main ETL flow ----------
def main():
    """
    Main entrypoint for the ETL. Downloads/loads source, transforms, and pushes to Google Sheets.
    """
    print("ETL start", datetime.utcnow())

    # 1) load raw Land Registry CSV (local preferred)
    df_raw = download_land_registry()

    # 2) transform to weekly transactions by LA
    weekly = prepare_transactions(df_raw, join_postcode_to_la=True)

    # 3) compute rolling windows (e.g., 4-week and 12-week)
    windows_df = compute_windows(weekly, windows=[4,12])

    # 4) prepare a "latest" snapshot (optional)
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()

    # 5) obtain GCP service account JSON from environment variable (set in GitHub Actions secrets)
    gcp_json_text = os.environ.get('GCP_SA_JSON')
    if not gcp_json_text:
        # helpful error if the env var isn't set
        raise RuntimeError("GCP_SA_JSON environment variable not found. Please set it to the service account JSON content (use GitHub secret).")
    # parse to dict
    gcp_json = json.loads(gcp_json_text)

    # 6) write the DataFrames to Google Sheets (tabs: weekly_by_la, windows, latest)
    sheets_to_write = {
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }
    write_to_google_sheets(sheets_to_write, gcp_json)
    print("ETL finished", datetime.utcnow())

# Standard Python entrypoint guard
if __name__ == "__main__":
    main()
