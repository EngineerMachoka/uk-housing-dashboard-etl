"""
UK Housing ETL Pipeline — Land Registry Price Paid Data

✅ Downloads official pp-complete.txt dataset
✅ Cleans transaction records
✅ Aggregates by week & local authority
✅ Calculates rolling 4- and 12-week totals
✅ Publishes output to Google Sheets
✅ Runs automated in GitHub Actions
"""

# -------------------------
# IMPORTS
# -------------------------

import os                                # Used to read environment variables in GitHub Actions
import io                                # Converts downloaded text into in-memory file stream
import json                              # Decodes GCP service account JSON
from datetime import datetime            # Logging timestamps

import requests                           # HTTP download of dataset
import pandas as pd                      # Data processing
import numpy as np                       # Numeric operations

from google.oauth2.service_account import Credentials  # Auth for Google Sheets API
from googleapiclient.discovery import build            # Interacts with Sheets API


# -------------------------
# CONSTANTS — CONFIGURATION
# -------------------------

# ✅ ONLY DATA SOURCE — DO NOT CHANGE
LAND_REGISTRY_TXT_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.txt"
)

# ✅ Spreadsheet ID stored securely in GitHub Actions → Settings → Secrets → GOOGLE_SHEET_ID
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")

# Required OAuth permission scope — allows writing to spreadsheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


# -------------------------
# 1) DOWNLOAD DATASET
# -------------------------

def download_land_registry_txt():
    """
    Downloads the pp-complete.txt dataset directly from HM Land Registry’s S3 bucket.
    This file is comma-delimited and behaves like CSV, but is far more reliable than the
    public CSV endpoints — which sometimes return empty responses.

    Returns:
        pandas.DataFrame — full unprocessed transaction dataset

    Raises:
        RuntimeError — if download fails or file cannot be parsed
    """
    print(f"📥 Downloading Land Registry TXT from {LAND_REGISTRY_TXT_URL} ...")

    try:
        # Requests file — timeout increased because dataset is large (~7M rows)
        response = requests.get(LAND_REGISTRY_TXT_URL, timeout=300)
        response.raise_for_status()  # Immediately fail if HTTP 4XX/5XX

        # Decode as UTF-8 and ignore weird encoding characters instead of crashing
        text_data = response.content.decode("utf-8", errors="ignore")

        # Prevent silently accepting empty text responses
        if not text_data.strip():
            raise RuntimeError("Downloaded TXT file is empty.")

        # Parse TXT as CSV because it is comma-separated
        df = pd.read_csv(
            io.StringIO(text_data),
            sep=",",
            low_memory=False,        # Avoid dtype warnings for large file
            skip_blank_lines=True     # Skip random blank rows
        )

        # Validate successful parsing
        if df.empty or len(df.columns) == 0:
            raise RuntimeError("TXT file parsed but contains no usable columns.")

        print(f"✅ Successfully loaded {len(df):,} records.")
        return df

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download or parse TXT dataset: {e}")


# -------------------------
# 2) CLEAN & PREPARE DATA
# -------------------------

def prepare_transactions(df_raw):
    """
    Standardizes, cleans, and maps transaction data.
    Generates weekly transaction counts by Local Authority.

    Returns:
        pandas.DataFrame — weekly transaction totals
    """

    df = df_raw.copy()

    # Automatically find the transaction date column — names vary by year
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if not date_col:
        raise RuntimeError("Transaction date column not found.")

    # Convert date strings → datetime objects
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])  # Remove rows without valid dates

    # Identify postcode column if present
    pc_col = next((c for c in df.columns if "postcode" in c.lower()), None)

    # Standardize postcodes — uppercase + remove spaces
    if pc_col:
        df["postcode"] = (
            df[pc_col]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
        )
    else:
        df["postcode"] = None  # Continue so data still processes

    # Create unique transaction ID if dataset doesn't provide one
    df["transaction_id"] = np.arange(len(df))

    # Approximate local authority via first 4 postcode characters
    df["local_authority"] = df["postcode"].str[:4]

    # Convert date → Monday week start
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    # Weekly transaction counts per area
    weekly = (
        df.groupby(["week", "local_authority"])
        .agg(transactions=("transaction_id", "nunique"))
        .reset_index()
        .sort_values(["local_authority", "week"])
    )

    return weekly


# -------------------------
# 3) ROLLING TREND ANALYSIS
# -------------------------

def compute_rolling_windows(weekly_df, windows=[4, 12]):
    """
    Computes rolling summed transaction counts over multiple week windows.

    Args:
        weekly_df — grouped weekly dataset
        windows — list of window lengths (in weeks)

    Returns:
        pandas.DataFrame — long-format rolling trend dataset
    """

    weekly_df["week"] = pd.to_datetime(weekly_df["week"])

    # Unique list of tracked local authorities
    local_areas = weekly_df["local_authority"].unique()

    # Generate full weekly calendar — prevents missing-week gaps
    week_range = pd.date_range(
        weekly_df["week"].min(),
        weekly_df["week"].max(),
        freq="W-MON"
    )

    # Create full matrix of week × local authority
    expanded = (
        pd.MultiIndex.from_product([week_range, local_areas], names=["week", "local_authority"])
        .to_frame(index=False)
        .merge(weekly_df, on=["week", "local_authority"], how="left")
        .fillna({"transactions": 0})
        .sort_values(["local_authority", "week"])
    )

    trend_outputs = []

    # Compute each rolling window separately
    for w in windows:
        temp = expanded.copy()
        temp["rolling_transactions"] = (
            temp.groupby("local_authority")["transactions"]
            .transform(lambda s: s.rolling(w, min_periods=1).sum())
        )
        temp["window_weeks"] = w
        trend_outputs.append(temp)

    return pd.concat(trend_outputs, ignore_index=True)


# -------------------------
# 4) WRITE TO GOOGLE SHEETS
# -------------------------

def write_to_google_sheets(dfs, gcp_json):
    """
    Uploads multiple DataFrames to Google Sheets — one tab per dataset.

    Args:
        dfs — dict: {"tab_name": dataframe}
        gcp_json — decoded service account JSON
    """

    # Authenticate using credentials stored in GitHub Secret
    creds = Credentials.from_service_account_info(gcp_json, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    for tab_name, df in dfs.items():

        # Convert DataFrame → list-of-lists for Sheets API
        rows = [df.columns.tolist()] + df.astype(str).replace({np.nan: ""}).values.tolist()

        # Clear existing tab contents to avoid mismatched row counts
        sheet.values().clear(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=tab_name
        ).execute()

        # Upload new data
        sheet.values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=f"{tab_name}!A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()


# -------------------------
# 5) MAIN ETL EXECUTION
# -------------------------

def main():
    """ Master pipeline — runs entire ETL process """
    print(f"🚀 ETL started {datetime.utcnow().isoformat()}")

    # Step 1 — Download raw dataset
    df_raw = download_land_registry_txt()

    # Step 2 — Create weekly transaction dataset
    weekly = prepare_transactions(df_raw)

    # Step 3 — Create rolling trend analysis
    rolling = compute_rolling_windows(weekly)

    # Step 4 — Extract most recent week snapshot
    latest = rolling[rolling["week"] == rolling["week"].max()].copy()

    # Load Google Auth service account from GitHub Secret
    gcp_env = os.environ.get("GCP_SA_JSON")
    if not gcp_env:
        raise RuntimeError("Missing GCP_SA_JSON secret in GitHub Actions.")
    gcp_json = json.loads(gcp_env)

    if not GOOGLE_SHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEET_ID secret.")

    # Step 5 — Upload results to Sheets
    write_to_google_sheets(
        {
            "weekly_by_la": weekly,
            "rolling_windows": rolling,
            "latest_snapshot": latest,
        },
        gcp_json
    )

    print(f"✅ ETL completed {datetime.utcnow().isoformat()}")


# Run pipeline when executed directly
if __name__ == "__main__":
    main()
