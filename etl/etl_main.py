# etl/etl_main.py
# Python 3.10+
import os
import io
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# -------- CONFIG ----------
LAND_REG_CSV_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update/latest/pp-complete.csv" 
# If that URL does not work, download the CSV and place in repo or change to gov.uk link.
POSTCODE_TO_LA_CSV = "lookups/uk_postcode_to_la.csv"  # optional file to join postcodes -> Local Authority
GOOGLE_SHEET_ID = "YOUR_GOOGLE_SHEET_ID"  # replace or set via env var
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
# --------------------------

def download_land_registry(url=LAND_REG_CSV_URL):
    print("Downloading Land Registry CSV...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), low_memory=False)
    return df

def prepare_transactions(df_raw, join_postcode_to_la=True):
    # Standard Land Registry PPD fields include: 'transaction_unique_identifier' or 'price', 'postcode', 'date_of_transfer'
    df = df_raw.copy()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise RuntimeError("No date column found")
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date'])
    # Try to find unique id
    id_col = None
    for c in df.columns:
        if "unique" in c.lower() or "id" == c.lower():
            id_col = c
            break
    if id_col is None:
        df['transaction_id'] = np.arange(len(df))
    else:
        df['transaction_id'] = df[id_col]
    # Postcode cleanup
    pc_col = next((c for c in df.columns if "postcode" in c.lower()), None)
    if pc_col:
        df['postcode'] = df[pc_col].str.replace(r'\s+', '', regex=True).str.upper()
    else:
        df['postcode'] = None

    # Aggregate weekly by Local Authority (if join provided) or by postcode centroid
    if join_postcode_to_la and os.path.exists(POSTCODE_TO_LA_CSV):
        la = pd.read_csv(POSTCODE_TO_LA_CSV, dtype=str)
        la['pc_nospace'] = la['postcode'].str.replace(r'\s+','',regex=True).str.upper()
        df = df.merge(la[['pc_nospace','local_authority']], left_on='postcode', right_on='pc_nospace', how='left')
    else:
        df['local_authority'] = df['postcode'].str[:4]  # fallback: use postcode prefix as bucket (approx)
    # Week start
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    # Count transactions
    weekly = df.groupby(['week','local_authority']).agg(
        transactions = ('transaction_id','nunique')
    ).reset_index()
    weekly = weekly.sort_values(['local_authority','week'])
    return weekly

def compute_windows(df_weekly, windows=[4,12,52]):
    df = df_weekly.copy()
    output = []
    for w in windows:
        df_w = df.copy()
        df_w['window_weeks'] = w
        # rolling sum per LA. Need to pivot to ensure continuous weeks:
        # first create full index of weeks x LA
        all_weeks = pd.date_range(df['week'].min(), df['week'].max(), freq='W-MON')
        las = df['local_authority'].dropna().unique()
        full = pd.MultiIndex.from_product([all_weeks, las], names=['week','local_authority'])
        full_df = pd.DataFrame(index=full).reset_index()
        merged = full_df.merge(df, on=['week','local_authority'], how='left').fillna(0)
        merged = merged.sort_values(['local_authority','week'])
        merged['rolling_trans'] = merged.groupby('local_authority')['transactions'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        merged['window_weeks'] = w
        output.append(merged[['week','local_authority','rolling_trans','transactions','window_weeks']])
    return pd.concat(output, ignore_index=True)

def write_to_google_sheets(df_dict, creds_json):
    # df_dict: {'sheet_tab_name': dataframe}
    creds = Credentials.from_service_account_info(creds_json, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    for tab_name, df in df_dict.items():
        # Prepare values
        values = [df.columns.tolist()] + df.replace({np.nan: ''}).astype(str).values.tolist()
        body = {"values": values}
        # clear the range then write
        range_name = f"{tab_name}!A1"
        print(f"Updating sheet tab: {tab_name} rows:{len(values)}")
        sheet.values().clear(spreadsheetId=GOOGLE_SHEET_ID, range=tab_name).execute()
        sheet.values().update(spreadsheetId=GOOGLE_SHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()

def main():
    print("ETL start", datetime.utcnow())
    df_raw = download_land_registry()
    weekly = prepare_transactions(df_raw)
    windows_df = compute_windows(weekly, windows=[4,12])
    # Prepare sheet tabs
    # - weekly by LA
    # - windows aggregated
    # - latest summary
    latest = windows_df[windows_df['week'] == windows_df['week'].max()].copy()
    # Load credentials from env var (GitHub Actions will set)
    import json
    gcp_json = json.loads(os.environ['GCP_SA_JSON'])
    # Write
    sheets = {
        "weekly_by_la": weekly,
        "windows": windows_df,
        "latest": latest
    }
    write_to_google_sheets(sheets, gcp_json)
    print("ETL finished", datetime.utcnow())

if __name__ == "__main__":
    main()
