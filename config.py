# config.py
import os
from pathlib import Path
import platform
import logging # Add logging setup here

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# Suppress overly verbose TF logging if desired
# logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- Core Paths ---
ROOT_DIR = Path(__file__).resolve().parent # Use resolve() for robustness
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
VIZ_PATH = ROOT_DIR / "visualizations"

# --- Kaggle API Path (Confirmed by user) ---
# Claude's logic correctly finds this on Windows: C:\Users\POLSTORE\.kaggle
if platform.system() == "Windows":
    KAGGLE_DIR = Path(os.environ.get("USERPROFILE", "~")) / ".kaggle"
else:
    KAGGLE_DIR = Path(os.environ.get("HOME", "~")) / ".kaggle"
KAGGLE_JSON_PATH = KAGGLE_DIR / "kaggle.json"

# --- Create Directories ---
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, VIZ_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# --- Competition & Data Constants ---
COMPETITION_NAME = "march-machine-learning-mania-2025"
CURRENT_SEASON = 2025  # The season we are predicting FOR
# Determine start year based on detailed data availability (adjust if needed)
DATA_START_YEAR_M = 2003
DATA_START_YEAR_W = 2010
# Use all available seasons up to CURRENT_SEASON - 1 for training feature generation
TRAINING_END_SEASON = CURRENT_SEASON - 1

TARGET = "WinTeam1" # Our prediction target variable name

# --- File Lists (Comprehensive) ---
# Using M/W dicts where applicable for clarity in loading/processing
# Define essential files needed for the pipeline to run
ESSENTIAL_FILES = {
    'MTeams', 'WTeams', 'MSeasons', 'WSeasons',
    'MRegularSeasonDetailedResults', 'WRegularSeasonDetailedResults', # Needed for detailed stats
    'MNCAATourneySeeds', 'WNCAATourneySeeds', # Needed for seeds
    'MRegularSeasonCompactResults', 'WRegularSeasonCompactResults', # Needed for basic stats/history
    'MNCAATourneyCompactResults', 'WNCAATourneyCompactResults' # Needed for tourney history
}
# Define all potentially available files
ALL_AVAILABLE_FILES = ESSENTIAL_FILES.union({
    'MTeamCoaches', 'WTeamCoaches', # Optional features
    'MMasseyOrdinals', 'WMasseyOrdinals', # Optional features
    # Add others like Conferences, Cities, etc., if used
})
SAMPLE_SUBMISSION_FILE = "SampleSubmissionStage1.csv" # Verify exact name!
FINAL_SUBMISSION_FILE = f"{CURRENT_SEASON}_submission_tf_v1.csv" # Version your submissions

# --- Processed File Names ---
DATA_CACHE_FILE = PROCESSED_DATA_DIR / "data_cache.pkl"
TEAM_STATS_FILE = PROCESSED_DATA_DIR / "team_stats_per_season.parquet" # Main feature store
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "training_matchups.parquet"
TEST_DATA_FILE = PROCESSED_DATA_DIR / f"{CURRENT_SEASON}_prediction_matchups.parquet"
OOF_PREDS_FILE = PROCESSED_DATA_DIR / "oof_tf_v1_predictions.csv" # Version OOF too
SCALER_FILE = MODELS_DIR / "scaler_tf_v1.joblib" # Version scaler

# --- Modeling Constants ---
RANDOM_SEED = 42
N_FOLDS = 5 # Number of folds for GroupKFold Cross-Validation
USE_GPU = False # Set True if you switch to GPU and install 'tensorflow' package

# --- TensorFlow/Keras Params (Adjustable) ---
TF_PARAMS = {
    'hidden_layers': [256, 128, 64], # Slightly deeper
    'dropout_rate': 0.4,           # Increase dropout slightly
    'learning_rate': 0.0005,       # Lower learning rate
    'batch_size': 1024,
    'epochs': 150,                 # Increase epochs, rely on early stopping
    'early_stopping_patience': 15, # Increase patience
    'l2_reg': 0.0005               # Slightly lower regularization
}