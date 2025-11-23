# -------- CONFIG ----------
LAND_REG_CSV_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update/latest/pp-complete.csv"
# Fallback local path (if you add the CSV to the repo at data/pp-complete.csv)
LAND_REG_CSV_LOCAL = "data/pp-complete.csv"
POSTCODE_TO_LA_CSV = "lookups/uk_postcode_to_la.csv"  # optional file to join postcodes -> Local Authority
GOOGLE_SHEET_ID = "YOUR_GOOGLE_SHEET_ID"  # replace or set via env var
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
# --------------------------

def download_land_registry(url=LAND_REG_CSV_URL, local_path=LAND_REG_CSV_LOCAL):
    """
    Try to read local CSV first (repo). If not available, try remote download.
    """
    import os
    print("download_land_registry: checking local path:", local_path)
    if os.path.exists(local_path):
        print("Found local Land Registry CSV at", local_path, "- loading from disk.")
        df = pd.read_csv(local_path, low_memory=False)
        return df
    # else try remote download (original behaviour)
    print("Local CSV not found; attempting remote download from:", url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), low_memory=False)
    return df
