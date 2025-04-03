# src/submit.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib # To load scaler
from tqdm.auto import tqdm
import time
import os
from pathlib import Path

# --- Import from Project Modules ---
try:
    from config import (TEST_DATA_FILE, MODELS_DIR, SUBMISSIONS_DIR, SCALER_FILE,
                        FINAL_SUBMISSION_FILE, N_FOLDS, CURRENT_SEASON, PROCESSED_DATA_DIR)
    from src.utils import logger, seed_everything, reduce_mem_usage, brier_score_tf
    config_loaded = True
except ImportError as e:
     print(f"ERROR: submit.py failed import: {e}")
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
     logger = logging.getLogger('src.submit')
     logger.error("Imports failed, prediction may not work correctly.")
     config_loaded = False


# --- Main Prediction Function ---
def generate_tf_predictions(test_file=None, n_folds=None):
    """Loads trained TF models and generates predictions for the submission file."""
    if not config_loaded: logger.error("Config failed to load. Cannot run prediction."); return None
    # Use config values if arguments are None
    test_file = test_file if test_file is not None else TEST_DATA_FILE
    n_folds = n_folds if n_folds is not None else N_FOLDS

    seed_everything()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info(f"Loading test matchup data from {test_file}...")
    try: pred_df = pd.read_parquet(test_file)
    except FileNotFoundError as e: logger.error(f"FATAL: Prediction file not found at {test_file}."); raise e
    pred_df = reduce_mem_usage(pred_df, verbose=False); logger.info(f"Loaded prediction data. Shape: {pred_df.shape}")

    # Identify feature columns
    feature_names = sorted([col for col in pred_df.columns if col.endswith('_Diff')])
    if not feature_names: logger.error("No difference features found in prediction data."); return None
    logger.info(f"Using {len(feature_names)} features: {feature_names[:5]}...")
    if 'ID' not in pred_df.columns: logger.error("FATAL: 'ID' column missing."); return None
    X_test = pred_df[feature_names].copy()

    # --- Load Scaler and Scale ---
    logger.info(f"Loading scaler from {SCALER_FILE}...")
    try: scaler = joblib.load(SCALER_FILE); logger.info("Scaling test features..."); X_test_scaled = scaler.transform(X_test); logger.info("✅ Test features scaled.")
    except FileNotFoundError: logger.error(f"❌ Scaler file not found: {SCALER_FILE}."); return None
    except Exception as e: logger.error(f"❌ Error loading/applying scaler: {e}", exc_info=True); return None

    # --- Load models and predict ---
    all_fold_preds = []
    logger.info(f"Loading {n_folds} TF models and predicting...")
    for fold in tqdm(range(n_folds), desc="Predicting Folds"):
        fold_start_time = time.time(); model_path = MODELS_DIR / f"model_fold_{fold+1}.keras"; logger.debug(f"Loading model {model_path}...")
        if not model_path.exists(): logger.error(f"Model file not found: {model_path}. Skipping fold {fold+1}."); continue
        try:
            model = keras.models.load_model(model_path, custom_objects={'brier_score_tf': brier_score_tf}) # Use correct name
            fold_preds = model.predict(X_test_scaled, batch_size=8192, verbose=0).flatten() # Larger batch size
            all_fold_preds.append(fold_preds); logger.debug(f"Fold {fold+1} pred done ({time.time() - fold_start_time:.2f}s).")
            del model; keras.backend.clear_session(); gc.collect() # Clear memory after each fold prediction
        except Exception as e: logger.error(f"❌ Error fold {fold+1} ({model_path}): {e}", exc_info=True); logger.warning("Skipping fold.")

    if not all_fold_preds: logger.error("❌ No predictions generated."); return None
    if len(all_fold_preds) < n_folds: logger.warning(f"Generated preds from {len(all_fold_preds)}/{n_folds} models.")

    # --- Average and Create Submission ---
    logger.info(f"Averaging {len(all_fold_preds)} fold predictions..."); avg_preds = np.mean(all_fold_preds, axis=0)
    logger.info("Creating submission file..."); submission_df = pred_df[['ID']].copy(); submission_df['Pred'] = avg_preds
    min_clip, max_clip = 0.01, 0.99; submission_df['Pred'] = np.clip(submission_df['Pred'], min_clip, max_clip); logger.info(f"Predictions clipped to [{min_clip}, {max_clip}].")
    output_path = SUBMISSIONS_DIR / FINAL_SUBMISSION_FILE; submission_df.to_csv(output_path, index=False); total_time = time.time() - start_time
    logger.info(f"✅ Submission file generated: {output_path}"); logger.info(f"Submission shape: {submission_df.shape}"); logger.info(f"Prediction head:\n{submission_df.head()}"); logger.info(f"Prediction generation took {total_time:.2f} seconds.")
    return submission_df

# Removed if __name__ == "__main__": block 