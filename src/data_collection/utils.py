import os
import pandas as pd
from datetime import datetime


# ----------------------------- LOGGER -----------------------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------- SAFE CSV READ --------------------------
def read_csv_safe(path):
    """
    Safely reads a CSV file.
    Returns an empty DataFrame if file does not exist or is invalid.
    """
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path)
            log(f"Loaded existing file: {path} ({len(df)} rows)")
            return df
        else:
            log(f"File not found or empty: {path}")
            return pd.DataFrame()
    except Exception as e:
        log(f"Error reading CSV {path}: {e}")
        return pd.DataFrame()


# -------------------------- SAFE CSV WRITE --------------------------
def write_csv_safe(df, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        log(f"Error writing CSV {path}: {e}")


# ------------------------ MERGE UNIQUE COMMENTS ------------------------
def merge_unique_comments(new_df, old_df):
    if old_df.empty:
        log("No existing comments found. Using new dataset only.")
        return new_df

    # Detect duplication key
    key = "comment_id" if "comment_id" in new_df.columns else None

    try:
        if key:
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged.drop_duplicates(subset=[key], inplace=True)
        else:
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged.drop_duplicates(subset=["video_id", "text"], inplace=True)

        log(f"Merged datasets: {len(old_df)} old + {len(new_df)} new → {len(merged)} unique rows")
        return merged.reset_index(drop=True)
    except Exception as e:
        log(f"Error merging comments: {e}")
        return new_df
