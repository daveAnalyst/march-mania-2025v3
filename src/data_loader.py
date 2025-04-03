# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm.auto import tqdm
import os
import subprocess
import json
import platform
import shutil
import pickle
import time

# --- Import from Project Modules ---
try:
    from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CACHE_FILE,
                        COMPETITION_NAME, ALL_AVAILABLE_FILES, ESSENTIAL_FILES,
                        SAMPLE_SUBMISSION_FILE, KAGGLE_JSON_PATH)
    from src.utils import logger, load_csv_data, check_kaggle_api
except ImportError as e:
     print(f"ERROR: Failed data_loader import: {e}")
     # Setup basic logger if utils failed
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
     logger = logging.getLogger('src.data_loader')
     logger.error("Imports failed, data loading may not work correctly.")


def download_competition_data(force=False):
    """Download competition data from Kaggle using the CLI."""
    if not check_kaggle_api():
        logger.error("Kaggle API setup failed. Cannot download data.")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_path = str(RAW_DATA_DIR)
    zip_file = RAW_DATA_DIR / f"{COMPETITION_NAME}.zip"

    # Only download if zip doesn't exist or force is True
    # Also check if essential files are missing even if zip exists (maybe extraction failed before)
    essential_missing = any(not (RAW_DATA_DIR / f"{f}.csv").exists() for f in ESSENTIAL_FILES)

    if force or essential_missing or not zip_file.exists() :
        if force: logger.info("Forcing data download...")
        elif essential_missing: logger.info("Essential files missing, attempting download...")
        else: logger.info(f"Zip file {zip_file} not found, attempting download...")

        logger.info(f"Downloading data for '{COMPETITION_NAME}' to {download_path}...")
        command = [
            "kaggle", "competitions", "download", "-c", COMPETITION_NAME,
            "-p", download_path, "--force" # Force overwrite existing zip
        ]
        try:
            logger.info(f"Executing: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8') # Use check=True
            logger.info("Kaggle download command successful.")
            logger.debug(f"Stdout:\n{result.stdout}")
        except FileNotFoundError:
            logger.error("ERROR: 'kaggle' command not found. Is the Kaggle CLI installed and in your system's PATH?")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle download command failed with return code {e.returncode}.")
            logger.error(f"Stderr:\n{e.stderr}")
            # logger.error(f"Stdout:\n{e.stdout}") # Stdout might be large
            logger.error("Please check the competition name, your Kaggle credentials, and network connection.")
            if zip_file.exists():
                try: zip_file.unlink(); logger.info("Deleted potentially corrupted zip file.")
                except OSError: logger.error("Could not delete corrupted zip file.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during kaggle download: {e}")
            return False
    else:
        logger.info(f"Zip file {zip_file} already exists and essential files seem present. Skipping download.")

    # --- Extraction ---
    if zip_file.exists():
        logger.info(f"Extracting {zip_file}...")
        import zipfile
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            zip_file.unlink() # Delete zip after successful extraction
            logger.info("Extraction complete.")
            time.sleep(0.5) # Brief pause
            return True
        except zipfile.BadZipFile:
            logger.error(f"ERROR: {zip_file} is corrupted. Deleting."); zip_file.unlink(); return False
        except Exception as e: logger.error(f"ERROR: Failed to extract {zip_file}: {e}"); return False
    elif any((RAW_DATA_DIR / f"{f}.csv").exists() for f in ESSENTIAL_FILES):
         logger.info("Zip file not found, but essential CSV files exist.")
         return True
    else:
         logger.error("Zip file not found after download attempt, and essential files are missing.")
         return False


def load_raw_data(reload=False, download_if_missing=True, use_cache=True):
    """Loads all NCAA tournament data files defined in config.ALL_AVAILABLE_FILES."""
    if use_cache and DATA_CACHE_FILE.exists() and not reload:
        logger.info(f"Attempting to load data from cache: {DATA_CACHE_FILE}")
        try:
            with open(DATA_CACHE_FILE, 'rb') as f: data = pickle.load(f)
            logger.info(f"Data successfully loaded from cache ({len(data)} files).")
            if validate_data_integrity(data): return data
            else: logger.warning("Cached data failed validation. Reloading.")
        except Exception as e: logger.error(f"Error loading cached data: {e}. Reloading.")

    if download_if_missing:
        logger.info("Checking for data presence and downloading if necessary...")
        download_successful = download_competition_data(force=reload) # Force download if reload=True
        if not download_successful and not any((RAW_DATA_DIR / f"{f}.csv").exists() for f in ESSENTIAL_FILES):
             logger.error("Download/Extraction failed and essential files missing. Cannot load data.")
             return {} # Return empty if download fails and files aren't there

    logger.info(f"Loading data files from CSVs in {RAW_DATA_DIR}...")
    data = {}
    files_loaded_count = 0
    files_missing_count = 0
    for file_stem in tqdm(ALL_AVAILABLE_FILES, desc="Loading CSV files", unit="file"):
        file_path = RAW_DATA_DIR / f"{file_stem}.csv"
        df = load_csv_data(file_path)
        if df is not None: data[file_stem] = df; files_loaded_count += 1
        else: files_missing_count += 1
    logger.info(f"Loaded {files_loaded_count} files. {files_missing_count} other specified files were not found.")

    if not validate_data_integrity(data):
        logger.error("Essential data is missing or corrupt after loading CSVs. Cannot proceed.")
        return {}

    if use_cache:
        logger.info(f"Caching loaded data to {DATA_CACHE_FILE}")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(DATA_CACHE_FILE, 'wb') as f: pickle.dump(data, f)
            logger.info("Data cached successfully.")
        except Exception as e: logger.error(f"Error caching data: {e}")

    return data

def load_sample_submission():
    """Loads the sample submission file specified in config."""
    # Ensure config was loaded or provide default path
    try: from config import RAW_DATA_DIR, SAMPLE_SUBMISSION_FILE
    except ImportError: logger.error("Config not loaded, cannot find sample submission."); return None
    sample_file_path = RAW_DATA_DIR / SAMPLE_SUBMISSION_FILE
    df = load_csv_data(sample_file_path, use_mem_reduce=False)
    if df is None: logger.warning(f"Sample submission '{SAMPLE_SUBMISSION_FILE}' not found.")
    return df

def validate_data_integrity(data):
    """Checks if ESSENTIAL datasets defined in config are present and non-empty."""
    # Ensure config was loaded
    try: from config import ESSENTIAL_FILES
    except ImportError: logger.error("Config not loaded, cannot validate data."); return False
    logger.info("Validating data integrity...")
    is_valid = True
    for file_stem in ESSENTIAL_FILES:
        if file_stem not in data or data[file_stem].empty:
            logger.error(f"Validation FAILED: Essential dataset '{file_stem}' is missing or empty.")
            is_valid = False
    if is_valid: logger.info("Data integrity validation PASSED.")
    else: logger.error("Please ensure essential files are available and downloaded correctly.")
    return is_valid

if __name__ == "__main__":
    logger.info("Running data loader script directly...")
    raw_data = load_raw_data(reload=True, download_if_missing=True, use_cache=True) # Force reload for test
    if raw_data: print(f"\nSuccessfully loaded {len(raw_data)} datasets.")
    else: print("\nFailed to load data.") 