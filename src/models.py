# src/models.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss # Use sklearn's brier for final reporting
from sklearn.preprocessing import StandardScaler
import joblib # To save the scaler
import time
import os
import gc # Garbage collector
from pathlib import Path # Import Path

# --- Import from Project Modules ---
try:
    from config import (TRAIN_DATA_FILE, MODELS_DIR, SCALER_FILE, OOF_PREDS_FILE,
                        RANDOM_SEED, TARGET, N_FOLDS, TF_PARAMS, PROCESSED_DATA_DIR)
    from src.utils import logger, seed_everything, reduce_mem_usage, brier_score_tf
    config_loaded = True
except ImportError as e:
     print(f"ERROR: models.py failed import: {e}")
     # Setup basic logger if utils failed
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
     logger = logging.getLogger('src.models')
     logger.error("Imports failed, model training may not work correctly.")
     config_loaded = False # Flag that config wasn't loaded


def build_tf_model(input_shape, params=None):
    """Builds the Keras Sequential model based on config."""
    if not config_loaded and params is None:
         logger.error("Config not loaded and no TF_PARAMS passed to build_tf_model"); return None
    if params is None: params = TF_PARAMS # Use default from config if not passed

    seed_everything(RANDOM_SEED) # Ensure consistent initialization
    model = keras.Sequential(name="MarchMadness_NN")
    model.add(keras.Input(shape=(input_shape,), name="Input")) # Input shape based on features

    # Build hidden layers based on params
    for units in params.get('hidden_layers', [128, 64]): # Provide default layers
        model.add(layers.Dense(units, activation="relu",
                               kernel_regularizer=keras.regularizers.l2(params.get('l2_reg', 0.001))))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params.get('dropout_rate', 0.3)))

    model.add(layers.Dense(1, activation="sigmoid", name="Output")) # Sigmoid for probability

    optimizer = keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))

    # Compile with Brier score loss (the custom TF version)
    model.compile(optimizer=optimizer,
                  loss=brier_score_tf, # Use brier_score_tf from utils
                  metrics=[brier_score_tf, 'AUC']) # Track Brier and AUC using the TF version
    logger.info("Keras model built successfully.")
    model.summary(print_fn=logger.info) # Log model summary
    return model

# Updated function signature to use default from config
def train_evaluate_tf(train_file=None, n_folds=None, params=None):
    """Trains and evaluates the TensorFlow Keras model using GroupKFold CV."""
    # Use config values if arguments are None
    if not config_loaded: logger.error("Config failed to load. Cannot run training without parameters."); return None, None, []
    train_file = train_file if train_file is not None else TRAIN_DATA_FILE
    n_folds = n_folds if n_folds is not None else N_FOLDS
    params = params if params is not None else TF_PARAMS

    seed_everything(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info(f"Loading training data from {train_file}...")
    try: df = pd.read_parquet(train_file)
    except FileNotFoundError as e: logger.error(f"FATAL: Training file not found at {train_file}."); raise e
    df = reduce_mem_usage(df, verbose=False)

    # Identify feature columns
    feature_names = sorted([col for col in df.columns if col.endswith('_Diff')])
    if not feature_names: logger.error("No difference features found."); return None, None, []
    logger.info(f"Using {len(feature_names)} features: {feature_names[:5]}...")

    X = df[feature_names]; y = df[TARGET]; groups = df['Season'] # Target might be int8

    # --- Feature Scaling ---
    logger.info("Scaling features..."); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE); logger.info(f"Scaler saved to {SCALER_FILE}")
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index) # Keep as DF for iloc

    oof_preds = np.zeros(len(df)); oof_brier_scores = []; oof_logloss_scores = []; fold_histories = []
    gkf = GroupKFold(n_splits=n_folds); logger.info(f"Starting {n_folds}-Fold GroupKFold Training...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
        fold_start_time = time.time(); logger.info(f"===== Fold {fold+1}/{n_folds} =====")
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]; y_train_orig, y_val_orig = y.iloc[train_idx], y.iloc[val_idx]

        # --- FIX: CAST y TO float32 FOR TENSORFLOW ---
        y_train = y_train_orig.astype(np.float32); y_val = y_val_orig.astype(np.float32); logger.debug(f"Casted y to {y_train.dtype}")

        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}"); logger.info(f"Val seasons: {sorted(df.iloc[val_idx]['Season'].unique())}")

        # Build and compile model
        model = build_tf_model(input_shape=X_train.shape[1], params=params)
        if model is None: logger.error("Model building failed."); continue # Skip fold if model fails

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_brier_score_tf', patience=params['early_stopping_patience'], restore_best_weights=True, mode='min', verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_brier_score_tf', factor=0.2, patience=5, min_lr=1e-6, mode='min', verbose=1)
        fold_model_path = MODELS_DIR / f"model_fold_{fold+1}.keras"; model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=fold_model_path, save_best_only=True, monitor='val_brier_score_tf', mode='min', verbose=0)

        logger.info(f"Training Fold {fold+1}..."); history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[early_stopping, reduce_lr, model_checkpoint], verbose=2)
        fold_histories.append(history.history)

        # Load best model
        logger.info(f"Loading best weights for fold {fold+1}...");
        if fold_model_path.exists():
            try: model = keras.models.load_model(fold_model_path, custom_objects={'brier_score_tf': brier_score_tf})
            except Exception as load_err: logger.error(f"Failed load best model {fold+1}: {load_err}"); logger.warning("Using end-of-training weights.")
        else: logger.warning(f"Checkpoint file {fold_model_path} not found. Using end-of-training weights.")

        # Predict OOF
        logger.info("Predicting OOF..."); fold_preds = model.predict(X_val, batch_size=params['batch_size']*4, verbose=0).flatten(); oof_preds[val_idx] = fold_preds

        # Evaluate fold
        brier = brier_score_loss(y_val, fold_preds); logloss = log_loss(y_val, fold_preds); oof_brier_scores.append(brier); oof_logloss_scores.append(logloss); fold_time = time.time() - fold_start_time
        logger.info(f"Fold {fold+1} Brier: {brier:.5f}, LogLoss: {logloss:.5f} ({fold_time:.2f}s)")
        keras.backend.clear_session(); gc.collect() # Clear memory

    # Overall OOF evaluation
    if len(oof_brier_scores) > 0: # Check if any folds completed
        overall_brier = brier_score_loss(y, oof_preds); overall_logloss = log_loss(y, oof_preds)
        logger.info("="*30); logger.info("Overall OOF Performance:"); logger.info(f"OOF Brier Score: {overall_brier:.5f}"); logger.info(f"OOF Log Loss: {overall_logloss:.5f}"); logger.info("="*30)
        # Save OOF predictions
        oof_df = df[['Season', 'Team1ID', 'Team2ID', TARGET]].copy(); oof_df['oof_pred'] = oof_preds
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        oof_df.to_csv(OOF_PREDS_FILE, index=False); logger.info(f"OOF predictions saved to {OOF_PREDS_FILE}")
    else:
        logger.error("No folds completed successfully. Cannot calculate OOF performance or save predictions.")
        return None, None, [] # Return indication of failure

    total_time = time.time() - start_time; logger.info(f"Total training time: {total_time:.2f} seconds.")
    return oof_preds, scaler, feature_names 