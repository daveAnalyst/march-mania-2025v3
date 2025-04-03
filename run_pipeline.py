# run_pipeline.py
import time
import pandas as pd
import os
import sys
from pathlib import Path

# --- Add project root to Python path ---
# This ensures modules can be imported correctly when run directly
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(PROJECT_ROOT)) # Add project root

# --- Import Project Modules & Config ---
try:
    from config import (
        PROCESSED_DATA_DIR, TRAIN_DATA_FILE, OOF_PREDS_FILE,
        TARGET, SUBMISSIONS_DIR, FINAL_SUBMISSION_FILE, TEST_DATA_FILE,
        CURRENT_SEASON, TEAM_STATS_FILE # Added missing imports
    )
    from src.utils import logger, seed_everything
    from src.data_loader import load_raw_data
    from src.features import create_all_features, create_training_matchups, create_prediction_matchups
    from src.models import train_evaluate_tf
    from src.submit import generate_tf_predictions
    from src.visualize import (
        plot_feature_distributions,
        plot_correlation_matrix,
        plot_oof_calibration,
        plot_oof_distribution
    )
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    print("Please ensure all required files (config.py, src/*) exist and contain the correct code.")
    exit(1) # Stop if imports fail


def run():
    """Runs the full end-to-end machine learning pipeline."""
    start_total_time = time.time()
    logger.info("Starting the March Madness Prediction Pipeline...")
    seed_everything() # Seed at the very beginning

    # === Stage 1: Load Raw Data ===
    logger.info("\n===== Stage 1: Load Raw Data =====")
    stage_start = time.time()
    raw_data = load_raw_data(reload=False, download_if_missing=True, use_cache=True)
    if not raw_data:
        logger.error("Pipeline halted: Failed to load raw data.")
        return
    logger.info(f"Stage 1 completed in {time.time() - stage_start:.2f} seconds.")

    # === Stage 2: Feature Engineering ===
    logger.info("\n===== Stage 2: Feature Engineering =====")
    stage_start = time.time()
    # Create and save aggregated team stats first
    team_stats = create_all_features(raw_data)
    if team_stats.empty:
        logger.error("Pipeline halted: Failed to create team stats.")
        return

    # Create and save training matchup data using the aggregated stats
    train_data, train_features = create_training_matchups(raw_data, team_stats)
    if train_data.empty:
        logger.error("Pipeline halted: Failed to create training data.")
        return

    # Create and save prediction matchup structure using aggregated stats
    # Handle case where CURRENT_SEASON stats might not be generated yet
    team_stats_for_pred = team_stats.copy() # Start with all stats
    if CURRENT_SEASON not in team_stats['Season'].unique():
        latest_season = team_stats['Season'].max()
        logger.warning(f"Stats for prediction season {CURRENT_SEASON} not found. Using latest ({latest_season}) as proxy.")
        team_stats_for_pred = team_stats[team_stats['Season']==latest_season].copy()
        if not team_stats_for_pred.empty:
            team_stats_for_pred['Season'] = CURRENT_SEASON
        else:
            logger.error("No proxy stats found for prediction!")
            team_stats_for_pred = pd.DataFrame() # Ensure it's empty

    if not team_stats_for_pred.empty:
        pred_data, pred_features = create_prediction_matchups(team_stats_for_pred, raw_data)
        if pred_data.empty:
            logger.error("Pipeline halted: Failed to create prediction data structure.")
            return
        # Critical Check: Feature Consistency
        if not train_features or not pred_features or set(train_features) != set(pred_features):
            logger.error("FATAL: Mismatch between training and prediction features! Check feature_engineering logic.")
            logger.error(f"Train Features ({len(train_features)}): {sorted(train_features)}")
            logger.error(f"Pred Features ({len(pred_features)}): {sorted(pred_features)}")
            return # Stop pipeline if features don't match
    else:
         logger.error("Pipeline halted: No stats available for prediction structure generation.")
         return

    logger.info(f"Stage 2 completed in {time.time() - stage_start:.2f} seconds.")

    # === Stage 3: Pre-Training Visualization ===
    logger.info("\n===== Stage 3: Pre-Training Visualization =====")
    stage_start = time.time()
    try:
        if not train_data.empty and train_features:
            logger.info("Generating pre-training visualizations...")
            plot_feature_distributions(train_data, train_features)
            plot_correlation_matrix(train_data, train_features)
        else:
            logger.warning("Skipping pre-training visualization as training data or features are missing.")
    except Exception as e:
        logger.warning(f"Pre-training visualization failed: {e}", exc_info=True)
    logger.info(f"Stage 3 completed in {time.time() - stage_start:.2f} seconds.")

    # === Stage 4: Model Training ===
    logger.info("\n===== Stage 4: Model Training (TensorFlow) =====")
    stage_start = time.time()
    oof_preds, scaler, features_from_train = train_evaluate_tf()
    if oof_preds is None:
        logger.error("Pipeline halted: Model training failed.")
        return
    logger.info(f"Stage 4 completed in {time.time() - stage_start:.2f} seconds.")

    # === Stage 5: Post-Training Visualization ===
    logger.info("\n===== Stage 5: Post-Training Visualization =====")
    stage_start = time.time()
    try:
        if OOF_PREDS_FILE.exists():
            logger.info("Generating post-training visualizations...")
            oof_df = pd.read_csv(OOF_PREDS_FILE)
            plot_oof_calibration(oof_df)
            plot_oof_distribution(oof_df)
        else:
            logger.warning(f"OOF predictions file not found at {OOF_PREDS_FILE}, skipping OOF visualization.")
    except Exception as e:
        logger.warning(f"Post-training visualization failed: {e}", exc_info=True)
    logger.info(f"Stage 5 completed in {time.time() - stage_start:.2f} seconds.")

    # === Stage 6: Generate Predictions ===
    logger.info("\n===== Stage 6: Generate Final Predictions =====")
    stage_start = time.time()
    submission_df = generate_tf_predictions()
    if submission_df is None:
        logger.error("Pipeline halted: Prediction generation failed.")
        return
    logger.info(f"Stage 6 completed in {time.time() - stage_start:.2f} seconds.")

    # === Pipeline End ===
    end_total_time = time.time()
    logger.info(f"\n🚀🚀🚀 Pipeline finished successfully in {end_total_time - start_total_time:.2f} seconds. 🚀🚀🚀")
    # Use the imported config variables for the final path
    final_submission_path = SUBMISSIONS_DIR / FINAL_SUBMISSION_FILE
    logger.info(f"Submission file generated at: {final_submission_path}")


if __name__ == "__main__":
    run()