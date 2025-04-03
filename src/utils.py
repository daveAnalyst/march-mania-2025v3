# src/utils.py
import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
import logging
import platform
import json
from pathlib import Path # Import Path

# --- Setup Logger ---
# Use basicConfig here to ensure logging works even if config import fails later
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('src.utils') # Use a specific logger name

# --- Try importing config vars, provide defaults if fails ---
try:
    from config import RANDOM_SEED, KAGGLE_JSON_PATH, KAGGLE_DIR
except ImportError:
    logger.error("Could not import from config.py in utils. Using default values.")
    RANDOM_SEED = 42 # Default seed
    # Define default Kaggle paths if config fails
    if platform.system() == "Windows":
        KAGGLE_DIR = Path(os.environ.get("USERPROFILE", "~")) / ".kaggle"
    else:
        KAGGLE_DIR = Path(os.environ.get("HOME", "~")) / ".kaggle"
    KAGGLE_JSON_PATH = KAGGLE_DIR / "kaggle.json"


def seed_everything(seed=RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Global random seed set to {seed}")

def reduce_mem_usage(df, verbose=True):
    """Iterate through all columns of a dataframe and modify the data type."""
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose: logger.debug(f'Mem usage start: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_object_dtype(col_type) or pd.api.types.is_datetime64_any_dtype(col_type):
            continue

        c_min = df[col].min()
        c_max = df[col].max()
        has_nans = df[col].isnull().any()

        try:
            if pd.api.types.is_integer_dtype(col_type) and not has_nans:
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_type):
                 if np.isfinite(c_min) and np.isfinite(c_max):
                     # Use float32 as minimum float type
                     if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                          # Add precision check if needed
                          df[col] = df[col].astype(np.float32)
                     else: df[col] = df[col].astype(np.float64)
                 else: df[col] = df[col].astype(np.float64) # Keep float64 if non-finite
        except Exception as e: logger.warning(f"Could not convert {col} (type {col_type}): {e}")

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.debug(f'Mem usage end: {end_mem:.2f} MB')
        decrease = (start_mem - end_mem) / max(start_mem, 1e-9) * 100
        logger.debug(f'Decreased by {decrease:.1f}%')
    return df

def load_csv_data(file_path, use_mem_reduce=True, **kwargs):
    """Loads a CSV file, optionally reducing memory usage."""
    try:
        df = pd.read_csv(file_path, **kwargs)
        if use_mem_reduce: df = reduce_mem_usage(df)
        logger.info(f"✅ Loaded {Path(file_path).name} (shape: {df.shape})")
        return df
    except FileNotFoundError: logger.warning(f"⚠️ File not found: {file_path}"); return None
    except Exception as e: logger.error(f"❌ Error loading {Path(file_path).name}: {str(e)}"); return None

def extract_seed_number(seed_str):
    """Extracts the integer part of a seed string like W01, X11a -> 1, 11"""
    if pd.isna(seed_str) or not isinstance(seed_str, str) or len(seed_str) < 3: return 25
    try: return int(seed_str[1:3])
    except ValueError: return 25 # Handle cases like 'W1', 'XY', etc.

def brier_score_tf(y_true, y_pred):
    """Custom Brier score loss function for TensorFlow/Keras."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))

def check_kaggle_api():
    """Check if Kaggle API is configured and available"""
    try: import kaggle; logger.debug("Kaggle package found.")
    except ImportError: logger.error("Kaggle package not installed."); return False

    if not KAGGLE_JSON_PATH.exists():
        logger.error(f"Kaggle API credentials not found at {KAGGLE_JSON_PATH}.")
        logger.error("Download 'kaggle.json' from Kaggle account settings.")
        return False

    try:
        with open(KAGGLE_JSON_PATH, 'r') as f: creds = json.load(f)
        if 'username' not in creds or 'key' not in creds:
            logger.error(f"Kaggle credentials file {KAGGLE_JSON_PATH} invalid."); return False
    except Exception as e: logger.error(f"Error reading {KAGGLE_JSON_PATH}: {str(e)}"); return False

    if platform.system() != "Windows":
        try:
            current_mode = KAGGLE_JSON_PATH.stat().st_mode & 0o777
            if current_mode != 0o600: os.chmod(KAGGLE_JSON_PATH, 0o600); logger.info(f"Set {KAGGLE_JSON_PATH} permissions to 600.")
        except Exception as e: logger.warning(f"Could not set permissions on kaggle.json: {str(e)}")

    logger.info("Kaggle API credentials appear valid.")
    return True 