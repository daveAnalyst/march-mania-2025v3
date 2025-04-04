# March Madness Mania 2025 Prediction Engine 🏀🧠

This repository contains code for predicting the outcomes of the 2025 NCAA Men's and Women's Division I Basketball Tournaments for the Kaggle March Machine Learning Mania 2025 competition.

**Competition Link:** [https://www.kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)

## Overview

This project implements an end-to-end Python pipeline designed to handle the complete machine learning workflow for this competition:

1.  **Data Ingestion:** Automatically downloads and caches competition data via the Kaggle API, performs integrity validation.
2.  **Feature Engineering:** Calculates a rich set of team-level statistics per season for both Men's and Women's teams, including advanced metrics. Generates matchup features based on the difference between opponent stats.
3.  **Modeling:** Trains a TensorFlow Neural Network using robust time-aware cross-validation.
4.  **Evaluation:** Assesses model performance locally using the Out-of-Fold (OOF) Brier score.
5.  **Prediction:** Generates predictions for all possible 2025 matchups in the required Kaggle submission format.

## Technology Stack

*   **Core:** Python 3.11+
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** TensorFlow/Keras, Scikit-learn
*   **Utilities:** Kaggle API, Joblib, Tqdm, Pathlib
*   **Visualization:** Matplotlib, Seaborn (via `src/visualize.py`)
*   **Environment:** Venv
*   **Version Control:** Git / GitHub

## Feature Engineering Highlights

The pipeline engineers the following features per team per season, then calculates the difference between opponents for modeling:

*   **Basic Stats:** Games Played, Win Percentage, Points Per Game (Scored, Allowed, Difference).
*   **Advanced Efficiency:** Offensive, Defensive, and Net Efficiency (Points Per 100 Possessions).
*   **Four Factors (Dean Oliver):** Effective Field Goal %, Turnover %, Offensive Rebound %, Free Throw Rate.
*   **Seed Information:** Integer representation of tournament seed (using 25 for non-tournament teams).
*   **Recent Momentum:** Win % and Average Score Difference over the last 10 games of the season.

## Model & Validation

*   **Model:** TensorFlow Sequential Neural Network.
    *   Architecture: Input -> Dense(256, ReLU)+BN+Dropout -> Dense(128, ReLU)+BN+Dropout -> Dense(64, ReLU)+BN+Dropout -> Dense(1, Sigmoid) *(Based on current `TF_PARAMS`)*
    *   Optimizer: Adam
    *   Loss Function: Custom Brier Score
*   **Validation:** 5-Fold `GroupKFold` cross-validation, grouped by `Season`. This is crucial to prevent data leakage and provide a reliable estimate of how the model generalizes to unseen seasons.

## Results

*   **Current Out-of-Fold (OOF) Brier Score:** **[0.2500]**


## Visualizations

The pipeline generates diagnostic plots saved to the `visualizations/` directory (ignored by Git). Key plots include:

*   **Feature Distributions & Correlations:** To understand the engineered features (generated pre-training).
*   **OOF Calibration Curve:** To assess if predicted probabilities match observed frequencies.
*   **OOF Prediction Distribution:** To see the spread and confidence of OOF predictions.

*
[OOF Calibration Curve](![Image](https://github.com/user-attachments/assets/71e86ff9-29b1-45a2-b1d9-5a0ef8e845f0))
![Image](https://github.com/user-attachments/assets/db3dab4e-a9d8-472f-b1e8-1e449b7ed66a)
![Image](https://github.com/user-attachments/assets/95dfdd7e-0e2e-45ec-8545-882008cf5b81)
![Image](https://github.com/user-attachments/assets/a378a158-93b5-4fc4-b5f3-ac74846db9d4)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/daveAnalyst/march-mania-2025-v3.git # Replace with your repo URL if different
    cd march-mania-2025-v3
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # Windows Powershell:
    .\.venv\Scripts\Activate.ps1
    # Windows Git Bash / Linux / macOS:
    source .venv/Scripts/activate
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Kaggle API Setup:** Ensure your `kaggle.json` API key is placed in the correct location (`~/.kaggle/kaggle.json` or `C:\Users\<User>\.kaggle\kaggle.json`). The `data_loader` script requires this for downloading data if it's not found locally.

## Usage

Execute the entire pipeline from the project root directory (`march-mania-2025-v2`):

```bash
python run_pipeline.py
