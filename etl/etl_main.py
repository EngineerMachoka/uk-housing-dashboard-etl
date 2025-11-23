# etl/etl_main.py
# Python 3.11+ recommended
# ETL: download HM Land Registry Price Paid data via their API (secure link),
# aggregate weekly counts and windows, and write to Google Sheets.

import os                               # interact with environment variables and file paths
import io                               # for in-memory bytes/text streams
import json                             # parsing JSON strings (GCP SA JSON etc.)
from datetime import datetime           # timestamping the run
import requests                         # HTTP requests (to HMLR API and file downloads)
import pandas as pd                     # data manipulation and CSV reading/writing
import numpy as np                      # numeric utilities
from google.oauth2.service_account import Credentials   # create credentials from SA JSON
from googleapiclient.discovery import build             # Google Sheets API client

# ---------- CONFIG ----------
# Base URL for HM Land Registry "Use land and property data" API
HMLR_API_BASE = "https://use-land-property-data.service.gov.uk/api/v1"  # base API endpoint used for all HMLR requests

# The dataset name or slug we want (Price Paid Data). We will search by 'price' or 'price paid' to find it.
# This value is used as a hint when parsing the API dataset list.
DATASET_NAME_HINT = "price"  # substring used to find the Price Paid dataset among HMLR dataset listings

# Google Sheets target (tab names will be created/overwritten)
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # must be set in GitHub secrets; fetched from environment

# Scopes required to update sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]  # OAuth scopes for Sheets/Drive

# ---------- HELPER: HM Land Registry API calls ----------

def hmlr_api_get(path, api_key, params=None):
    """
    Generic GET against the HM Land Registry API.
    - path: API path (e.g., '/datasets')
    - api_key: your HMLR API key (from account)
    - params: optional dict for query parameters
    Returns parsed JSON on success, raises on HTTP error.
    """
    url = HMLR_API_BASE.rstrip('/') + '/' + path.lstrip('/')  # build the full URL by joining base and path
    headers = {
        "Accept": "application/json",
        "x-api-key": api_key   # HMLR API uses x-api-key header (per docs)
    }
    r = requests.get(url, headers=headers, params=params, timeout=60)  # perform the HTTP GET with timeout
    r.raise_for_status()  # raise an exception for non-2xx responses so failures surface
    return r.json()  # parse and return JSON response body

def find_price_paid_dataset(api_key):
    """
    Query the HMLR API for datasets and return the dataset resource dict
    corresponding to Price Paid (search by hint). If multiple matches, pick best.
    """
    # GET /datasets (or similar) - API docs say there is a list endpoint
    datasets = hmlr_api_get("/datasets", api_key)  # request the list of available datasets from the API
    # datasets is expected to be a list or dict with 'items'; handle both
    items = datasets.get('items') if isinstance(datasets, dict) and 'items' in datasets else datasets  # normalize response shape to a list
    if not items:
        raise RuntimeError("No datasets returned by HMLR API. Check your API key and licence.")  # bail if no datasets
    # find any dataset containing the hint (case-insensitive) in title or name
    candidates = []  # will collect matching dataset resources
    for d in items:
        title = (d.get('title') or d.get('name') or "").lower()  # get title/name and lowercase for comparison
        if DATASET_NAME_HINT in title:
            candidates.append(d)  # add any dataset that contains the hint
    if not candidates:
        # fallback: return first item and hope it contains price paid resources
        # but raise a helpful error message so operator can debug
        raise RuntimeError("Price Paid dataset not found via HMLR API. Check dataset list: " + str([d.get('title') for d in items[:10]]))  # helpful diagnostic
    # pick first candidate
    return candidates[0]  # choose the first candidate as the Price Paid dataset

def get_secure_download_link(dataset, api_key):
    """
    Given a dataset resource dict (from find_price_paid_dataset), call the API to
    get secure download links (HMLR provides signed links for the actual CSV).
    The exact resource structure can vary; we search common keys.
    """
    # Many HMLR dataset entries have a 'resources' list with download objects
    resources = dataset.get('resources') or dataset.get('files') or []  # find resources/files array in dataset record
    # If resources exist and contain 'download' or 'url', pick CSV-like ones
    for r in resources:
        url = r.get('download') or r.get('url') or r.get('link')  # check common URL keys
        if url and (url.endswith('.csv') or 'pp' in (r.get('name') or '').lower() or 'price' in (r.get('title') or '').lower()):
            return url  # return the first plausible CSV URL found
    # If the resources do not include direct URLs, the API may require you to call a /datasets/{id}/download endpoint
    ds_id = dataset.get('id') or dataset.get('dataset_id') or dataset.get('slug')  # try to extract an identifier for the dataset
    if ds_id:
        # Attempt to call an endpoint that issues a secure link
        # This is best-effort: API paths can vary; try '/datasets/{id}/download' and '/datasets/{id}/resources'
        for try_path in (f"/datasets/{ds_id}/download", f"/datasets/{ds_id}/resources"):
            try:
                resp = hmlr_api_get(try_path, api_key)  # call the API for dataset-specific resources
                # if resp contains a URL or 'download' field, return it
                if isinstance(resp, dict):
                    # look for keys containing 'url' or 'link'
                    for v in resp.values():
                        if isinstance(v, str) and v.startswith('http') and '.csv' in v:
                            return v  # return any string value that looks like a CSV URL
                    # if 'items' exist with resource entries, recurse
                    if 'items' in resp and isinstance(resp['items'], list):
                        for item in resp['items']:
                            url = item.get('download') or item.get('url') or item.get('link')
                            if url and url.endswith('.csv'):
                                return url  # return CSV URL found in nested items
            except Exception:
                continue  # ignore errors here and try the next candidate path
    # If we reached here we couldn't find a download URL automatically
    raise RuntimeError("Could not resolve a secure download URL for the Price Paid dataset automatically. Inspect the dataset object: " + json.dumps(dataset)[:2000])  # raise with partial dataset JSON for debugging

# ---------- DOWNLOAD CSV using HMLR secure link ----------

def download_csv_from_url(url):
    """
    Download CSV bytes from a URL and return a pandas DataFrame.
    Uses requests streaming to protect memory for large files.
    """
    r = requests.get(url, stream=True, timeout=300)  # stream the HTTP GET to avoid loading entire response at once into memory
    r.raise_for_status()  # ensure HTTP status is OK, otherwise raise
    # For large files we could stream to disk; for simplicity, read into memory here
    text = r.content.decode('utf-8')  # decode the raw bytes to text assuming UTF-8 encoding
    df = pd.read_csv(io.StringIO(text), low_memory=False)  # parse CSV text into a pandas DataFrame
    return df  # return the parsed DataFrame

# ---------- TRANSFORMATIONS ----------

def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Given raw PPD DataFrame, clean and aggregate to weekly transactions by Local Authority.
    If postcode_lookup_path provided, join to get accurate local_authority; otherwise approximate using prefix.
    """
    df = df_raw.copy()  # copy the input DataFrame to avoid mutating caller's object
    # find date column and parse
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)  # find first column name containing 'date'
    if date_col is None:
        raise RuntimeError("No date column in PPD CSV.")  # fail early if no date column is found
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')  # parse the date column into pandas datetime, coercing errors to NaT
    df = df.dropna(subset=['date'])  # drop rows where date parsing failed
    # identify or create transaction id
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower()=='transactionuniqueidentifier'), None)  # try to find a unique id column
    if id_col:
        df['transaction_id'] = df[id_col]  # use the found unique id column as transaction_id
    else:
        df['transaction_id'] = np.arange(len(df))  # otherwise create a synthetic numeric transaction id
    # normalize postcode if present
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)  # identify postcode column if present
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+','',regex=True).str.upper()  # remove whitespace and uppercase postcode
    else:
        df['postcode'] = None  # set postcode to None when not present so downstream logic can handle it
    # join to postcode->LA lookup if available
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)  # load lookup CSV mapping postcodes to local authority
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+','',regex=True).str.upper()  # normalize lookup postcode
        df = df.merge(la[['pc_nospace','local_authority']], left_on='postcode', right_on='pc_nospace', how='left')  # merge LA into main DF
    else:
        df['local_authority'] = df['postcode'].str[:4]  # rough fallback: use first 4 chars of postcode as proxy for area
    # compute week start and aggregate
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)  # compute week-start datetime for grouping
    weekly = df.groupby(['week','local_authority']).agg(transactions=('transaction_id','nunique')).reset_index()  # aggregate unique transactions per week x LA
    weekly = weekly.sort_values(['local_authority','week'])  # sort for consistent ordering
    return weekly  # return the weekly aggregate DataFrame

def compute_rolling_windows(weekly_df, windows=[4,12]):
    """
    For each window (weeks), compute rolling sum per LA. Return concatenated table with 'window_weeks' column.
    """
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])  # ensure 'week' column is datetime for date_range and sorting
    outputs = []  # accumulator for per-window DataFrames
    las = weekly_df['local_authority'].dropna().unique()  # list of unique local authorities present
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')  # construct full continuous set of week-start dates
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week','local_authority'])  # create full grid of week x LA combinations
    full_df = pd.DataFrame(index=full_idx).reset_index()  # turn multiindex into a DataFrame with all combinations
    merged = full_df.merge(weekly_df, on=['week','local_authority'], how='left').fillna(0)  # merge actual data onto full grid, fill missing with 0
    merged = merged.sort_values(['local_authority','week'])  # sort for rolling computation
    for w in windows:
        m = merged.copy()  # work on a copy for this window to avoid modifying the master
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())  # compute rolling sum per LA
        m['window_weeks'] = w  # annotate which window size this row corresponds to
        outputs.append(m[['week','local_authority','rolling_trans','transactions','window_weeks']])  # select and append relevant columns
    return pd.concat(outputs, ignore_index=True)  # concatenate all window DataFrames into one long table

# ---------- WRITE to Google Sheets ----------

def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write dict of DataFrames to Google Sheets tabs.
    dfs_by_tab: {'tabname': dataframe}
    gcp_service_account_json: parsed JSON dict of service account credentials
    """
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)  # create Credentials object from service account JSON
    service = build('sheets', 'v4', credentials=creds)  # initialize the Sheets API client
    sheet = service.spreadsheets()  # reference to spreadsheets resource
    for tab, df in dfs_by_tab.items():  # iterate over each tab name and DataFrame in the input dict
        values = [df.columns.tolist()] + df.replace({np.nan:''}).astype(str).values.tolist()  # prepare a list of lists: header row + data rows for Sheets API
        body = {"values": values}  # request body for update
        range_name = f"{tab}!A1"  # target range start for writing (tab A1)
        # clear previous
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()  # clear the existing contents of the tab to avoid leftover rows
        # write new
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()  # write updated values into the sheet

# ---------- Main ETL flow ----------

def main():
    run_ts = datetime.utcnow().isoformat()  # capture the run timestamp as ISO string
    print("ETL start", run_ts)  # log start time

    # 1) get HM Land Registry API key from env (set as GitHub secret)
    api_key = os.environ.get('HMLR_API_KEY')  # fetch API key from environment variables
    if not api_key:
        raise RuntimeError("HMLR_API_KEY missing. Set it in repository secrets.")  # error out if API key isn't provided

    # 2) find the Price Paid dataset
    dataset = find_price_paid_dataset(api_key)  # use the helper to locate the Price Paid dataset resource
    print("Found dataset:", dataset.get('title') or dataset.get('name') or dataset.get('id'))  # log the found dataset identifier

    # 3) resolve secure download URL for CSV
    secure_url = get_secure_download_link(dataset, api_key)  # resolve a secure CSV download link from the dataset resource
    print("Resolved secure download URL (first 200 chars):", str(secure_url)[:200])  # log part of the secure URL for debugging

    # 4) download the CSV into pandas DataFrame
    df_raw = download_csv_from_url(secure_url)  # download CSV and parse into DataFrame
    print("Downloaded CSV rows:", len(df_raw))  # log number of rows downloaded

    # 5) transform: aggregate weekly by LA
    weekly = prepare_transactions(df_raw, postcode_lookup_path="lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None)  # transform raw PPD to weekly LA-level aggregates

    # 6) compute 4-week & 12-week rolling sums
    windows_df = compute_rolling_windows(weekly, windows=[4,12])  # compute rolling-window aggregates for 4 and 12 week windows

    # 7) prepare snapshot for 'latest'
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()  # extract the most recent week snapshot for the latest tab

    # 8) GCP service account JSON from env (GitHub secret)
    gcp_sa_text = os.environ.get('GCP_SA_JSON')  # retrieve the service account JSON text from environment
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Set it as GitHub secret with the contents of your service account JSON.")  # raise if not provided
    gcp_json = json.loads(gcp_sa_text)  # parse the JSON string into a Python dict

    # 9) write to Google Sheets
    write_to_google_sheets({
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }, gcp_json)  # call the writer to push three tabs to the target Google Sheet

    print("ETL finished", datetime.utcnow().isoformat())  # log completion timestamp

if __name__ == "__main__":
    main()  # execute main when run as a script
