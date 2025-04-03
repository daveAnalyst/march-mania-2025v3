# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.calibration import calibration_curve # For calibration plot

# --- Import from Project Modules ---
try:
    from config import VIZ_PATH, PROCESSED_DATA_DIR, TARGET, TRAIN_DATA_FILE, OOF_PREDS_FILE # Added OOF_PREDS_FILE
    from src.utils import logger
    config_loaded = True
except ImportError as e:
     print(f"ERROR: visualize.py failed import: {e}")
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
     logger = logging.getLogger('src.visualize')
     logger.error("Imports failed, visualization may not work correctly.")
     config_loaded = False # Flag that config wasn't loaded


# Ensure viz directory exists
if config_loaded: VIZ_PATH.mkdir(parents=True, exist_ok=True)

# --- Visualization Functions ---

def plot_feature_distributions(df, features, save_path=None, title="Feature Distributions"):
    """Plot distributions of specified numerical features."""
    if not config_loaded: logger.error("Config not loaded, cannot save plots."); return
    save_path = save_path or VIZ_PATH / "feature_distributions.png"
    logger.info(f"Plotting feature distributions to {save_path}")
    num_cols = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not num_cols: logger.warning("No numeric features found to plot distributions."); return

    n = len(num_cols); ncols = min(5, n); nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(ncols * 4, nrows * 3.5))
    for i, col in enumerate(num_cols):
        plt.subplot(nrows, ncols, i + 1)
        try:
            sns.histplot(df[col].dropna(), kde=True, bins=30); plt.title(f"{col}", fontsize=9); plt.xlabel(""); plt.ylabel(""); plt.xticks(fontsize=8); plt.yticks(fontsize=8)
        except Exception as e: logger.warning(f"Could not plot distribution for {col}: {e}")
    plt.suptitle(title, fontsize=14, y=1.02); plt.tight_layout(rect=[0, 0.03, 1, 0.98]); plt.savefig(save_path); plt.close(); logger.info("Feature distribution plots saved.")


def plot_correlation_matrix(df, features, save_path=None, title="Feature Correlation Matrix"):
    """Plot correlation matrix of specified numerical features."""
    if not config_loaded: logger.error("Config not loaded, cannot save plots."); return
    save_path = save_path or VIZ_PATH / "correlation_matrix.png"
    logger.info(f"Plotting correlation matrix to {save_path}")
    num_df = df[features].select_dtypes(include=np.number)
    if num_df.shape[1] < 2: logger.warning("Not enough numeric features for correlation matrix."); return

    plt.figure(figsize=(max(12, num_df.shape[1]*0.6), max(10, num_df.shape[1]*0.6)))
    corr = num_df.corr(); mask = np.triu(np.ones_like(corr, dtype=bool)); cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, linewidths=.5, cbar_kws={"shrink": .7})
    plt.title(title, fontsize=14); plt.xticks(rotation=60, ha='right', fontsize=8); plt.yticks(rotation=0, fontsize=8); plt.tight_layout(); plt.savefig(save_path); plt.close(); logger.info("Correlation matrix saved.")

def plot_oof_calibration(oof_df, save_path=None):
     """ Plots observed vs predicted probabilities from OOF predictions """
     if not config_loaded: logger.error("Config not loaded, cannot save plots."); return
     save_path = save_path or VIZ_PATH / "oof_calibration_curve.png"
     if 'oof_pred' not in oof_df.columns or TARGET not in oof_df.columns: logger.warning("OOF missing cols. Skipping calibration plot."); return
     logger.info(f"Plotting OOF calibration to {save_path}"); plt.figure(figsize=(7, 7))
     try:
        prob_true, prob_pred = calibration_curve(oof_df[TARGET], oof_df['oof_pred'], n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model Calibration'); plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='grey'); plt.xlabel("Mean Predicted Probability (bin)"); plt.ylabel("Fraction of Positives (actual rate)"); plt.title("OOF Prediction Calibration Curve"); plt.legend(); plt.grid(alpha=0.3); plt.xlim([0, 1]); plt.ylim([0, 1]); plt.savefig(save_path)
     except Exception as e: logger.error(f"Failed to plot calibration curve: {e}", exc_info=True)
     finally: plt.close()
     logger.info("OOF calibration plot saved.")

def plot_oof_distribution(oof_df, save_path=None):
    """ Plots the distribution of OOF predictions """
    if not config_loaded: logger.error("Config not loaded, cannot save plots."); return
    save_path = save_path or VIZ_PATH / "oof_prediction_distribution.png"
    if 'oof_pred' not in oof_df.columns: logger.warning("OOF missing 'oof_pred'. Skipping OOF distribution plot."); return
    logger.info(f"Plotting OOF prediction distribution to {save_path}"); plt.figure(figsize=(10, 6))
    sns.histplot(oof_df['oof_pred'], bins=50, kde=True); plt.title('Distribution of Out-of-Fold Predictions'); plt.xlabel('Predicted Probability'); plt.ylabel('Count'); plt.axvline(0.5, color='r', linestyle='--', alpha=0.7, label='0.5 Threshold'); plt.grid(True, alpha=0.3); plt.legend(); plt.savefig(save_path); plt.close(); logger.info("OOF prediction distribution plot saved.")

# --- Test Execution ---
if __name__ == "__main__":
    logger.info("Running visualization script directly (example usage)...")
    if not config_loaded: exit() # Stop if config failed
    try:
        if TRAIN_DATA_FILE.exists():
            train_data = pd.read_parquet(TRAIN_DATA_FILE)
            feature_names = sorted([col for col in train_data.columns if col.endswith('_Diff')])
            if feature_names: logger.info("Plotting from training data..."); plot_feature_distributions(train_data, feature_names); plot_correlation_matrix(train_data, feature_names)
            else: logger.warning("No diff features in train data.")
        else: logger.warning(f"Training data {TRAIN_DATA_FILE} not found.")
        if OOF_PREDS_FILE.exists():
            logger.info(f"Plotting from OOF data {OOF_PREDS_FILE}...")
            oof_data = pd.read_csv(OOF_PREDS_FILE)
            plot_oof_calibration(oof_data); plot_oof_distribution(oof_data)
        else: logger.info("OOF predictions file not found.")
    except Exception as e: logger.error(f"Error during visualization example: {e}", exc_info=True)