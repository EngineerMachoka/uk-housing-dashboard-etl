# etl/etl_main.py
# Full ETL script for HM Land Registry Price Paid -> Google Sheets
# Python 3.11+ recommended
# This file contains line-by-line comments explaining what each line does.

# ---------- imports: standard library ----------
import os                                 # operating system interfaces (env vars, path checks)
import io                                 # in-memory streams (wrap text for pandas CSV reader)
import json                               # parse JSON strings (GCP service account JSON)
from datetime import datetime             # create timestamps for logging

# ---------- imports: third-party ----------
import requests                           # perform HTTP requests (API calls and CSV downloads)
import pandas as pd                       # data manipulation and CSV reading/writing
import numpy as np                        # numeric utilities (nunique, NaN handling)

# google auth and sheets client imports
from google.oauth2.service_account import Credentials   # build credentials from service account JSON
from googleapiclient.discovery import build            # construct Google Sheets API service

# ---------- CONFIG ----------
HMLR_API_BASE = "https://use-land-property-data.service.gov.uk/api/v1"  # base URL for HMLR API endpoints
DATASET_NAME_HINT = "price"  # string hint to find Price Paid dataset among HMLR datasets
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # target Google Sheet ID (set as GitHub secret)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]  # OAuth scopes needed

# ---------- HELPER: HMLR API GET ----------
def hmlr_api_get(path, api_key, params=None):
    """
    Perform a GET request to the HMLR API.
    - path: API path (e.g., '/datasets' or '/datasets/{id}/resources')
    - api_key: HMLR API key from environment/secret
    - params: optional dict of query parameters
    Returns parsed JSON on success, raises HTTPError on failure.
    """
    url = HMLR_API_BASE.rstrip('/') + '/' + path.lstrip('/')  # build full endpoint URL
    headers = {
        "Accept": "application/json",  # request JSON response
        "x-api-key": api_key           # include HMLR API key in header per API docs
    }
    r = requests.get(url, headers=headers, params=params, timeout=60)  # perform HTTP GET with timeout
    r.raise_for_status()  # if HTTP response code is not 2xx raise requests.HTTPError
    return r.json()  # parse and return JSON body

# ---------- HELPER: find the Price Paid dataset ----------
def find_price_paid_dataset(api_key):
    """
    Locate the Price Paid dataset resource using the HMLR API.
    - api_key: HMLR API key
    Returns dataset resource dict if found, raises RuntimeError otherwise.
    """
    datasets = hmlr_api_get("/datasets", api_key)  # request list of datasets from HMLR
    # normalize response shape: some endpoints return {'items': [...]}, others a list directly
    items = datasets.get('items') if isinstance(datasets, dict) and 'items' in datasets else datasets
    if not items:
        # no datasets returned -> raise helpful error so operator can debug API key/licence
        raise RuntimeError("No datasets returned by HMLR API. Check your API key and licence.")
    candidates = []  # will collect dataset resources that match hint
    for d in items:
        # get a searchable title or name (lowercased)
        title = (d.get('title') or d.get('name') or "").lower()
        if DATASET_NAME_HINT in title:
            candidates.append(d)  # add if hint appears in title/name
    if not candidates:
        # if nothing matched, raise an error and provide a short diagnostic
        raise RuntimeError("Price Paid dataset not found via HMLR API. Check dataset list: " + str([d.get('title') for d in items[:10]]))
    return candidates[0]  # return the first candidate found

# ---------- HELPER: resolve secure download link ----------
def get_secure_download_link(dataset, api_key):
    """
    Given a dataset resource dict, try to extract a secure CSV download URL.
    The dataset structure varies, so we look in common keys and try dataset-specific endpoints.
    """
    resources = dataset.get('resources') or dataset.get('files') or []  # common keys for resource listings
    for r in resources:
        # common URL keys: 'download', 'url', 'link'
        url = r.get('download') or r.get('url') or r.get('link')
        # if this resource looks like the Price Paid CSV (by name/title/url), return it
        if url and (url.endswith('.csv') or 'pp' in (r.get('name') or '').lower() or 'price' in (r.get('title') or '').lower()):
            return url
    # if not found directly, attempt dataset-specific resource endpoints using dataset id/slug
    ds_id = dataset.get('id') or dataset.get('dataset_id') or dataset.get('slug')  # extract dataset identifier
    if ds_id:
        # try two likely endpoint patterns that may return resources or signed links
        for try_path in (f"/datasets/{ds_id}/download", f"/datasets/{ds_id}/resources"):
            try:
                resp = hmlr_api_get(try_path, api_key)  # call the dataset-specific endpoint
                if isinstance(resp, dict):
                    # look for any string value that is a CSV URL among the response values
                    for v in resp.values():
                        if isinstance(v, str) and v.startswith('http') and '.csv' in v:
                            return v
                    # if resp contains 'items', inspect each item for CSV URLs
                    if 'items' in resp and isinstance(resp['items'], list):
                        for item in resp['items']:
                            url = item.get('download') or item.get('url') or item.get('link')
                            if url and url.endswith('.csv'):
                                return url
            except Exception:
                # ignore errors here; continue trying other paths
                continue
    # if no download link could be resolved, raise an informative error (slice dataset JSON for brevity)
    raise RuntimeError("Could not resolve a secure download URL for the Price Paid dataset automatically. Inspect the dataset object: " + json.dumps(dataset)[:2000])

# ---------- DOWNLOAD CSV from resolved URL ----------
def download_csv_from_url(url):
    """
    Download a CSV from the given URL and return it as a pandas DataFrame.
    Uses streaming and decodes content as UTF-8.
    """
    r = requests.get(url, stream=True, timeout=300)  # stream download with larger timeout for big files
    r.raise_for_status()  # raise HTTPError if status not OK
    text = r.content.decode('utf-8')  # decode bytes to text assuming UTF-8 encoding
    df = pd.read_csv(io.StringIO(text), low_memory=False)  # parse CSV text to pandas DataFrame
    return df  # return parsed DataFrame

# ---------- TRANSFORMATION: prepare transactions ----------
def prepare_transactions(df_raw, postcode_lookup_path=None):
    """
    Clean raw Price Paid DataFrame and aggregate to weekly counts by Local Authority.
    - df_raw: pandas DataFrame loaded from PPD CSV
    - postcode_lookup_path: optional path to postcode->LA lookup CSV
    Returns weekly aggregated DataFrame with columns: week, local_authority, transactions
    """
    df = df_raw.copy()  # copy input to avoid side-effects on original variable
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)  # find first column that contains 'date'
    if date_col is None:
        raise RuntimeError("No date column in PPD CSV.")  # cannot proceed without date
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')  # parse dates, invalid->NaT
    df = df.dropna(subset=['date'])  # drop rows where date parsing failed
    id_col = next((c for c in df.columns if 'unique' in c.lower() or c.lower()=='transactionuniqueidentifier'), None)  # attempt to find unique id
    if id_col:
        df['transaction_id'] = df[id_col]  # copy existing unique identifier into a standard column
    else:
        df['transaction_id'] = np.arange(len(df))  # create synthetic transaction ids using row numbers
    pc_col = next((c for c in df.columns if 'postcode' in c.lower()), None)  # find postcode column if present
    if pc_col:
        df['postcode'] = df[pc_col].astype(str).str.replace(r'\s+','',regex=True).str.upper()  # normalize postcode: remove spaces, uppercase
    else:
        df['postcode'] = None  # set postcode column to None if not available
    if postcode_lookup_path and os.path.exists(postcode_lookup_path):
        la = pd.read_csv(postcode_lookup_path, dtype=str)  # load postcode->LA lookup mapping
        la['pc_nospace'] = la['postcode'].astype(str).str.replace(r'\s+','',regex=True).str.upper()  # normalize lookup postcodes
        df = df.merge(la[['pc_nospace','local_authority']], left_on='postcode', right_on='pc_nospace', how='left')  # merge LA codes/names
    else:
        df['local_authority'] = df['postcode'].str[:4]  # fallback: use first 4 chars of postcode as rough bucket when no lookup available
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)  # compute week-start timestamp for each transaction
    weekly = df.groupby(['week','local_authority']).agg(transactions=('transaction_id','nunique')).reset_index()  # aggregate weekly unique transaction counts by LA
    weekly = weekly.sort_values(['local_authority','week'])  # sort results for consistent ordering
    return weekly  # return the weekly aggregation DataFrame

# ---------- TRANSFORMATION: compute rolling windows ----------
def compute_rolling_windows(weekly_df, windows=[4,12]):
    """
    Compute rolling window sums (e.g., 4-week, 12-week) per Local Authority.
    Returns a concatenated DataFrame with columns: week, local_authority, rolling_trans, transactions, window_weeks
    """
    weekly_df['week'] = pd.to_datetime(weekly_df['week'])  # ensure week is datetime dtype
    outputs = []  # list accumulator for per-window DataFrames
    las = weekly_df['local_authority'].dropna().unique()  # get unique LAs present in data
    all_weeks = pd.date_range(weekly_df['week'].min(), weekly_df['week'].max(), freq='W-MON')  # construct continuous week range
    full_idx = pd.MultiIndex.from_product([all_weeks, las], names=['week','local_authority'])  # create complete week x LA grid
    full_df = pd.DataFrame(index=full_idx).reset_index()  # turn grid into DataFrame
    merged = full_df.merge(weekly_df, on=['week','local_authority'], how='left').fillna(0)  # merge actual data and fill missing with 0
    merged = merged.sort_values(['local_authority','week'])  # sort to ensure rolling computation correctness
    for w in windows:
        m = merged.copy()  # copy merged grid for this window size
        # compute rolling sum per local_authority, using min_periods=1 to avoid NaN at start of series
        m['rolling_trans'] = m.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        m['window_weeks'] = w  # annotate which window this row represents
        outputs.append(m[['week','local_authority','rolling_trans','transactions','window_weeks']])  # select output columns
    return pd.concat(outputs, ignore_index=True)  # concatenate per-window DataFrames into one long table

# ---------- WRITE TO GOOGLE SHEETS ----------
def write_to_google_sheets(dfs_by_tab, gcp_service_account_json):
    """
    Write a dictionary of DataFrames to the specified Google Sheet.
    - dfs_by_tab: mapping of sheet tab names to pandas DataFrames
    - gcp_service_account_json: parsed JSON dict of GCP service account credentials
    """
    creds = Credentials.from_service_account_info(gcp_service_account_json, scopes=SCOPES)  # create credentials for Sheets API
    service = build('sheets', 'v4', credentials=creds)  # build Sheets API service client
    sheet = service.spreadsheets()  # reference to spreadsheets resource
    for tab, df in dfs_by_tab.items():  # iterate through each tab name and DataFrame
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()  # prepare header row + data rows for upload
        body = {"values": values}  # body payload for Sheets update
        range_name = f"{tab}!A1"  # target range starts at A1 for the given tab
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab).execute()  # clear existing content of the tab first
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()  # write new values

# ---------- MAIN ETL FLOW ----------
def main():
    run_ts = datetime.utcnow().isoformat()  # capture current UTC timestamp as ISO string
    print("ETL start", run_ts)  # log start time to console

    api_key = os.environ.get('HMLR_API_KEY')  # fetch HM Land Registry API key from environment (GitHub secret)
    if not api_key:
        raise RuntimeError("HMLR_API_KEY missing. Set it in repository secrets.")  # error out if API key is not provided

    dataset = find_price_paid_dataset(api_key)  # discover Price Paid dataset resource using HMLR API
    print("Found dataset:", dataset.get('title') or dataset.get('name') or dataset.get('id'))  # log dataset identifier for debugging

    secure_url = get_secure_download_link(dataset, api_key)  # resolve secure signed download URL for CSV
    print("Resolved secure download URL (first 200 chars):", str(secure_url)[:200])  # log a substring of the URL so logs don't leak entire link

    df_raw = download_csv_from_url(secure_url)  # download CSV from resolved secure URL and parse into DataFrame
    print("Downloaded CSV rows:", len(df_raw))  # log how many rows were downloaded

    # transform: compute weekly transactions by local authority (use postcode lookup if available in repo under lookups/)
    weekly = prepare_transactions(df_raw, postcode_lookup_path="lookups/uk_postcode_to_la.csv" if os.path.exists("lookups/uk_postcode_to_la.csv") else None)

    windows_df = compute_rolling_windows(weekly, windows=[4,12])  # compute rolling windows for 4-week and 12-week windows

    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()  # prepare a snapshot of the most recent week

    gcp_sa_text = os.environ.get('GCP_SA_JSON')  # retrieve GCP service account JSON text from environment (GitHub secret)
    if not gcp_sa_text:
        raise RuntimeError("GCP_SA_JSON missing. Set it as GitHub secret with the contents of your service account JSON.")  # error if not set
    gcp_json = json.loads(gcp_sa_text)  # parse JSON string into Python dict

    write_to_google_sheets({
        "weekly_by_la": weekly,   # tab with weekly transaction counts by LA
        "windows": windows_df,    # tab with rolling window aggregates (4wk/12wk)
        "latest": latest          # tab with latest snapshot per LA
    }, gcp_json)  # write all tabs to the target Google Sheet using provided service account JSON

    print("ETL finished", datetime.utcnow().isoformat())  # log completion time

if __name__ == "__main__":
    main()  # run main when executed as a script
