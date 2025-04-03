# March Madness 2025 Prediction Project
# March Madness Mania 2025 Prediction Engine 🏀🧠

This repository contains code for predicting the outcomes of the 2025 NCAA Men's and Women's Division I Basketball Tournaments for the Kaggle March Machine Learning Mania 2025 competition.

**Competition Link:** [https://www.kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)

## Overview

This project implements an end-to-end pipeline to:
1.  Download and preprocess historical NCAA game data.
2.  Engineer advanced team statistics and matchup features.
3.  Train a machine learning model (currently TensorFlow NN) using GroupKFold cross-validation.
4.  Generate predictions for all possible 2025 matchups in the required Kaggle format.

## Features Engineered

*   Team Aggregates per Season (Points, Win%, etc.)
*   Advanced Efficiency Metrics (Offensive/Defensive/Net Efficiency per Possession)
*   Dean Oliver's Four Factors (eFG%, TOV%, ORB%, FTRate)
*   Team Momentum (Last 10 games Win%, Score Difference)
*   Seed Information
*   Difference Features between opposing teams for all relevant stats.

## Model

*   **Current Model:** TensorFlow Sequential Neural Network.
*   **Validation:** 5-Fold GroupKFold (grouped by Season).
*   **Evaluation Metric:** Brier Score.
*   **Current OOF Brier Score:** [**0.2500**]

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd march-mania-2025
    ```
2.  **Create and activate a virtual environment:**
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
4.  **Kaggle API:** Ensure your `kaggle.json` API key is placed correctly (e.g., `~/.kaggle/kaggle.json` or `C:\Users\<User>\.kaggle\kaggle.json`).

## Usage

Run the entire pipeline (data download, feature engineering, training, prediction):
```bash
python run_pipeline.py 
