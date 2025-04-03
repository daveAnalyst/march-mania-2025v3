# src/features.py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging
from pathlib import Path
import gc # Garbage collector
import itertools # Added for combinations

# --- Import from Project Modules ---
try:
    from config import (PROCESSED_DATA_DIR, CURRENT_SEASON, DATA_START_YEAR_M,
                        DATA_START_YEAR_W, TARGET, TEAM_STATS_FILE, TRAIN_DATA_FILE,
                        TEST_DATA_FILE, TRAINING_END_SEASON)
    from src.utils import logger, reduce_mem_usage, extract_seed_number
    config_loaded = True
except ImportError as e:
     print(f"ERROR: features.py failed import: {e}")
     # Setup basic logger if utils failed
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
     logger = logging.getLogger('src.features')
     logger.error("Imports failed, feature engineering may not work correctly.")
     config_loaded = False # Flag that config wasn't loaded

tqdm.pandas() # Enable progress_apply for pandas

# --- Feature Calculation Helpers ---

def _calculate_possessions(df):
    """Internal helper to calculate possessions."""
    cols_needed = ['WFGA', 'WOR', 'WTO', 'WFTA', 'LFGA', 'LOR', 'LTO', 'LFTA']
    for col in cols_needed:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: logger.warning(f"Possession missing {col}. Filling 0."); df[col] = 0
    df = df.fillna(0)
    df['WPoss'] = df['WFGA'] - df['WOR'] + df['WTO'] + 0.475 * df['WFTA']
    df['LPoss'] = df['LFGA'] - df['LOR'] + df['LTO'] + 0.475 * df['LFTA']
    df.loc[df['WPoss'] <= 0, 'WPoss'] = 1; df.loc[df['LPoss'] <= 0, 'LPoss'] = 1
    return df

def _calculate_efficiency(df):
    """Internal helper for efficiency metrics."""
    df['WScore'] = pd.to_numeric(df['WScore'], errors='coerce').fillna(0)
    df['LScore'] = pd.to_numeric(df['LScore'], errors='coerce').fillna(0)
    df['WPoss'] = pd.to_numeric(df['WPoss'], errors='coerce').fillna(1)
    df['LPoss'] = pd.to_numeric(df['LPoss'], errors='coerce').fillna(1)
    df['WOffEff'] = 100 * (df['WScore'] / df['WPoss'])
    df['LOffEff'] = 100 * (df['LScore'] / df['LPoss'])
    df['WDefEff'] = df['LOffEff']; df['LDefEff'] = df['WOffEff']
    df['WNetEff'] = df['WOffEff'] - df['WDefEff']; df['LNetEff'] = df['LOffEff'] - df['LDefEff']
    return df

def _calculate_four_factors(df):
    """Internal helper for Four Factors."""
    ff_cols = ['WFGM', 'WFGM3', 'WFGA', 'WTO', 'WOR', 'LDR', 'LFGM', 'LFGM3', 'LFGA', 'LTO', 'LOR', 'WDR', 'WFTA', 'LFTA', 'WPoss', 'LPoss']
    for col in ff_cols:
         if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
         else: logger.warning(f"Four factor missing {col}. Filling 0."); df[col] = 0
    df = df.fillna(0)
    df['WeFGPct'] = np.where(df['WFGA'] > 0, (df['WFGM'] + 0.5 * df['WFGM3']) / df['WFGA'], 0)
    df['LeFGPct'] = np.where(df['LFGA'] > 0, (df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA'], 0)
    df['WTOVPct'] = np.where(df['WPoss'] > 0, df['WTO'] / df['WPoss'], 0)
    df['LTOVPct'] = np.where(df['LPoss'] > 0, df['LTO'] / df['LPoss'], 0)
    df['WORPct'] = np.where((df['WOR'] + df['LDR']) > 0, df['WOR'] / (df['WOR'] + df['LDR']), 0)
    df['LORPct'] = np.where((df['LOR'] + df['WDR']) > 0, df['LOR'] / (df['LOR'] + df['WDR']), 0)
    df['WFTRate'] = np.where(df['WFGA'] > 0, df['WFTA'] / df['WFGA'], 0)
    df['LFTRate'] = np.where(df['LFGA'] > 0, df['LFTA'] / df['LFGA'], 0)
    return df

def _aggregate_stats(games_df, prefix):
    """Internal helper to aggregate stats for W/L teams."""
    team_id_col = f'{prefix}TeamID'
    stats_to_agg_config = {
        f'{prefix}Score': ['mean', 'std', 'sum'], f'{prefix}Poss': ['mean'],
        f'{prefix}OffEff': ['mean', 'std'], f'{prefix}DefEff': ['mean', 'std'], f'{prefix}NetEff': ['mean', 'std'],
        f'{prefix}eFGPct': ['mean'], f'{prefix}TOVPct': ['mean'], f'{prefix}ORPct': ['mean'], f'{prefix}FTRate': ['mean'],
    }
    agg_dict = {k: v for k, v in stats_to_agg_config.items() if k in games_df.columns}
    if not agg_dict: logger.warning(f"No cols to agg for prefix {prefix}"); return pd.DataFrame(columns=['Season', 'TeamID', f'Games_{prefix}'])

    numeric_agg_cols = list(agg_dict.keys())
    for col in numeric_agg_cols: games_df[col] = pd.to_numeric(games_df[col], errors='coerce')

    try:
        agg_stats = games_df.groupby(['Season', team_id_col], observed=True, dropna=False).agg(agg_dict) # Don't dropna during group
        agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns.values]
        agg_stats = agg_stats.reset_index().rename(columns={team_id_col: 'TeamID'})
        games_played = games_df.groupby(['Season', team_id_col], observed=True, dropna=False).size().reset_index(name=f'Games_{prefix}')
        agg_stats = pd.merge(agg_stats, games_played, on=['Season', 'TeamID'], how='left')
        agg_stats = agg_stats.fillna(0) # Fill NaNs from std dev etc.
        return agg_stats
    except Exception as e:
        logger.error(f"Error during aggregation for prefix '{prefix}': {e}", exc_info=True)
        games_played = games_df.groupby(['Season', team_id_col], observed=True, dropna=False).size().reset_index(name=f'Games_{prefix}')
        games_played = games_played.rename(columns={team_id_col: 'TeamID'})
        return games_played # Return counts only if agg fails

def calculate_season_aggregates(detailed_results_df):
    """Calculates comprehensive aggregated stats per team per season."""
    logger.info(f"Calculating detailed season aggregates (Input shape: {detailed_results_df.shape})...")
    if detailed_results_df.empty: logger.warning("Detailed results empty."); return pd.DataFrame()

    # --- Pre-calculation Cleaning ---
    numeric_cols = ['WScore', 'LScore', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    logger.debug("Applying pre-aggregation numeric conversion...")
    cleaned_count = 0
    for col in numeric_cols:
        if col in detailed_results_df.columns:
            original_dtype = str(detailed_results_df[col].dtype)
            detailed_results_df[col] = pd.to_numeric(detailed_results_df[col], errors='coerce')
            new_dtype = str(detailed_results_df[col].dtype)
            if original_dtype != new_dtype: cleaned_count += 1; logger.debug(f"Col '{col}': {original_dtype} -> {new_dtype}")
    if cleaned_count > 0: logger.info(f"Coerced {cleaned_count} basic stat cols to numeric (NaNs introduced).")
    detailed_results_df = detailed_results_df.fillna(0); logger.debug("Filled NaNs from initial coercion.")

    # --- Calculate Derived Features ---
    games = detailed_results_df.copy(); del detailed_results_df; gc.collect()
    logger.debug("Calculating derived features..."); games = _calculate_possessions(games); games = _calculate_efficiency(games); games = _calculate_four_factors(games); logger.debug("Derived features calculated.")

    # --- Aggregation ---
    logger.debug("Aggregating stats..."); win_agg = _aggregate_stats(games, 'W'); loss_agg = _aggregate_stats(games, 'L'); del games; gc.collect()
    if win_agg.empty and loss_agg.empty: logger.error("Aggregates empty."); return pd.DataFrame()

    # --- Merge Aggregates ---
    logger.debug("Merging aggregates...");
    if not win_agg.empty and not loss_agg.empty:
        try: season_stats = pd.merge(win_agg, loss_agg, on=['Season', 'TeamID'], how='outer', validate="one_to_one")
        except pd.errors.MergeError as e:
            logger.error(f"Merge Error: {e}. Retrying after dropping duplicates."); win_agg = win_agg.drop_duplicates(subset=['Season', 'TeamID']); loss_agg = loss_agg.drop_duplicates(subset=['Season', 'TeamID']); season_stats = pd.merge(win_agg, loss_agg, on=['Season', 'TeamID'], how='outer')
    elif not win_agg.empty: season_stats = win_agg
    else: season_stats = loss_agg
    season_stats = season_stats.fillna(0); logger.debug("Aggregates merged.")

    # --- Calculate Overall Metrics ---
    logger.debug("Calculating overall metrics..."); season_stats['GamesPlayed'] = season_stats.get('Games_W', 0) + season_stats.get('Games_L', 0); valid_games_mask = season_stats['GamesPlayed'] > 0; season_stats['WinPct'] = 0.0; season_stats.loc[valid_games_mask, 'WinPct'] = season_stats.get('Games_W', 0) / season_stats['GamesPlayed']
    metrics_to_average = ['OffEff', 'DefEff', 'NetEff', 'eFGPct', 'TOVPct', 'ORPct', 'FTRate']
    for metric in metrics_to_average:
        w_col, l_col, avg_col = f'{metric}_mean_W', f'{metric}_mean_L', f'Avg{metric}'
        has_w, has_l = w_col in season_stats.columns, l_col in season_stats.columns; season_stats[avg_col] = 0.0
        if has_w and has_l: w_games, l_games = season_stats.get('Games_W', 0), season_stats.get('Games_L', 0); season_stats.loc[valid_games_mask, avg_col] = ((season_stats[w_col] * w_games + season_stats[l_col] * l_games) / season_stats['GamesPlayed'])
        elif has_w: season_stats[avg_col] = season_stats[w_col]
        elif has_l: season_stats[avg_col] = season_stats[l_col]
    w_score_sum, l_score_sum_l = season_stats.get('WScore_sum_W', 0), season_stats.get('LScore_sum_L', 0); l_score_sum_w, w_score_sum_l = season_stats.get('LScore_sum_W', 0), season_stats.get('WScore_sum_L', 0); season_stats['PPG_Score'] = 0.0; season_stats.loc[valid_games_mask, 'PPG_Score'] = (w_score_sum + l_score_sum_l) / season_stats['GamesPlayed']; season_stats['PPG_Allow'] = 0.0; season_stats.loc[valid_games_mask, 'PPG_Allow'] = (l_score_sum_w + w_score_sum_l) / season_stats['GamesPlayed']; season_stats['PPG_Diff'] = season_stats['PPG_Score'] - season_stats['PPG_Allow']; logger.debug("Overall metrics calculated.")

    # --- Feature Selection ---
    feature_cols = ['Season', 'TeamID', 'WinPct', 'PPG_Score', 'PPG_Allow', 'PPG_Diff', 'GamesPlayed'] + [f'Avg{metric}' for metric in metrics_to_average if f'Avg{metric}' in season_stats.columns]
    final_stats = pd.DataFrame(columns=feature_cols); final_stats = pd.concat([final_stats, season_stats[[col for col in feature_cols if col in season_stats.columns]]], ignore_index=True); final_stats = final_stats.fillna(0)

    logger.info(f"Final aggregated season stats created. Shape: {final_stats.shape}")
    return reduce_mem_usage(final_stats)

def add_seed_info(df, seeds_df):
    """Merges tournament seed number."""
    if seeds_df.empty: logger.warning("Seed df empty."); df['SeedNum'] = 25; return df
    logger.debug("Adding seed number..."); seeds = seeds_df[['Season', 'TeamID', 'Seed']].copy(); seeds['SeedNum'] = seeds['Seed'].apply(extract_seed_number); df_merged = pd.merge(df, seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left'); df_merged['SeedNum'] = df_merged['SeedNum'].fillna(25).astype(int); return df_merged

def calculate_momentum(compact_results, num_games=10):
    """ Calculates win pct and avg score diff over last N games """
    if compact_results.empty: logger.warning("Compact results empty."); return pd.DataFrame(columns=['Season', 'TeamID', f'Last{num_games}_WinPct', f'Last{num_games}_ScoreDiff'])
    logger.info(f"Calculating momentum (last {num_games} games)..."); df = compact_results.sort_values(['Season', 'DayNum']).copy()
    df['WScore'] = pd.to_numeric(df['WScore'], errors='coerce').fillna(0); df['LScore'] = pd.to_numeric(df['LScore'], errors='coerce').fillna(0)
    df_wins = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore']].rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score', 'LScore': 'OppScore'}); df_wins['Win'] = 1
    df_loss = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore']].rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score', 'WScore': 'OppScore'}); df_loss['Win'] = 0
    all_games = pd.concat([df_wins, df_loss], ignore_index=True).sort_values(['Season', 'TeamID', 'DayNum']); all_games['ScoreDiff'] = all_games['Score'] - all_games['OppScore']
    logger.debug(f"Calculating rolling window (size={num_games})..."); all_games['Win'] = pd.to_numeric(all_games['Win'], errors='coerce').fillna(0.5); all_games['ScoreDiff'] = pd.to_numeric(all_games['ScoreDiff'], errors='coerce').fillna(0)
    rolling_grp = all_games.groupby(['Season', 'TeamID'], group_keys=False, observed=True)[['Win', 'ScoreDiff']]; rolling_stats = rolling_grp.rolling(window=num_games, min_periods=max(1, num_games // 2)).mean(); rolling_stats = rolling_stats.rename(columns={'Win': f'Last{num_games}_WinPct', 'ScoreDiff': f'Last{num_games}_ScoreDiff'})
    # Get the last value for each group efficiently
    rolling_stats = rolling_stats.reset_index(); last_indices = rolling_stats.loc[rolling_stats.groupby(['Season', 'TeamID'], observed=True)['level_2'].idxmax()]; end_season_momentum = last_indices[['Season', 'TeamID', f'Last{num_games}_WinPct', f'Last{num_games}_ScoreDiff']]
    logger.info(f"Momentum features done. Shape: {end_season_momentum.shape}"); return end_season_momentum

# --- Main Feature Engineering Pipeline ---
def create_all_features(raw_data_dict):
    """Orchestrates creation of features and saves aggregated team stats."""
    if not config_loaded: logger.error("Config failed to load, cannot run feature engineering."); return pd.DataFrame()
    logger.info("Starting feature engineering pipeline..."); all_team_stats_list = []; pipeline_success = True
    for gender in ['M', 'W']:
        logger.info(f"--- Processing {gender} data ---"); start_year = DATA_START_YEAR_M if gender == 'M' else DATA_START_YEAR_W
        detailed_key, compact_key, seeds_key = f'{gender}RegularSeasonDetailedResults', f'{gender}RegularSeasonCompactResults', f'{gender}NCAATourneySeeds'
        tourney_compact_key, tourney_detailed_key = f'{gender}NCAATourneyCompactResults', f'{gender}NCAATourneyDetailedResults'
        if detailed_key not in raw_data_dict or raw_data_dict[detailed_key].empty: logger.error(f"Essential {detailed_key} missing for {gender}."); pipeline_success = False; continue
        logger.debug(f"Combining detailed results for {gender}..."); detailed_results = pd.concat([raw_data_dict.get(detailed_key, pd.DataFrame()), raw_data_dict.get(tourney_detailed_key, pd.DataFrame())], ignore_index=True).pipe(reduce_mem_usage, verbose=False); detailed_results = detailed_results[detailed_results['Season'] >= start_year]
        if detailed_results.empty: logger.warning(f"No detailed results {gender}>={start_year}."); continue
        logger.debug(f"Detailed results shape: {detailed_results.shape}")
        team_stats = calculate_season_aggregates(detailed_results); del detailed_results; gc.collect()
        if team_stats.empty: logger.warning(f"No aggregates for {gender}."); continue
        seeds_df = raw_data_dict.get(seeds_key, pd.DataFrame()); team_stats = add_seed_info(team_stats, seeds_df)
        logger.debug(f"Combining compact results for {gender} momentum..."); compact_results = pd.concat([raw_data_dict.get(compact_key, pd.DataFrame()), raw_data_dict.get(tourney_compact_key, pd.DataFrame())], ignore_index=True).pipe(reduce_mem_usage, verbose=False); compact_results = compact_results[compact_results['Season'] >= start_year]
        if not compact_results.empty: momentum_df = calculate_momentum(compact_results, num_games=10); team_stats = pd.merge(team_stats, momentum_df, on=['Season', 'TeamID'], how='left'); team_stats[[col for col in team_stats.columns if 'Last10' in col]] = team_stats[[col for col in team_stats.columns if 'Last10' in col]].fillna(0); del momentum_df, compact_results; gc.collect()
        else: logger.warning(f"No compact results for {gender}, skipping momentum."); team_stats['Last10_WinPct'] = 0.0; team_stats['Last10_ScoreDiff'] = 0.0
        team_stats['Gender'] = gender; all_team_stats_list.append(team_stats); logger.info(f"Finished features for {gender} (Shape: {team_stats.shape}).")
    if not pipeline_success or not all_team_stats_list: logger.error("Feature engineering failed."); return pd.DataFrame()
    final_team_stats = pd.concat(all_team_stats_list, ignore_index=True).pipe(reduce_mem_usage)
    logger.info(f"Saving aggregated team stats to {TEAM_STATS_FILE}..."); PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True); final_team_stats.to_parquet(TEAM_STATS_FILE, index=False); logger.info(f"✅ Aggregated team stats saved (Shape: {final_team_stats.shape})"); return final_team_stats

def _create_matchups_core(df_source, team_stats_df, id1_col='Team1ID', id2_col='Team2ID'):
    """Core logic to create matchups and difference features."""
    logger.debug(f"Matchups core: Source={df_source.shape}, Stats={team_stats_df.shape}, ID1={id1_col}, ID2={id2_col}")
    if not all(c in df_source.columns for c in ['Season', id1_col, id2_col]): logger.error(f"Source missing cols"); return pd.DataFrame(), []
    if not all(c in team_stats_df.columns for c in ['Season', 'TeamID']): logger.error("Stats missing cols"); return pd.DataFrame(), []
    df_source_slim = df_source[['Season', id1_col, id2_col]].drop_duplicates() # Avoid duplicate source pairs
    logger.debug(f"Merging T1 ({id1_col})..."); df = pd.merge(df_source_slim, team_stats_df, left_on=['Season', id1_col], right_on=['Season', 'TeamID'], how='inner'); logger.debug(f"Post T1 merge: {df.shape}")
    if df.empty: logger.warning(f"Empty after T1 merge."); return pd.DataFrame(), []
    logger.debug(f"Merging T2 ({id2_col})..."); df = pd.merge(df, team_stats_df, left_on=['Season', id2_col], right_on=['Season', 'TeamID'], how='inner', suffixes=('_T1', '_T2')); logger.debug(f"Post T2 merge: {df.shape}")
    if df.empty: logger.warning(f"Empty after T2 merge."); return pd.DataFrame(), []
    base_features = sorted([col for col in team_stats_df.columns if col not in ['Season', 'TeamID', 'Gender']])
    diff_feature_cols = []; feature_cols_to_keep = ['Season'] # Start only with Season
    logger.debug(f"Calculating diffs for {len(base_features)} base features...")
    for base in base_features:
        t1_col, t2_col, diff_col = f'{base}_T1', f'{base}_T2', f'{base}_Diff'
        if t1_col in df.columns and t2_col in df.columns: df[diff_col] = df[t1_col] - df[t2_col]; diff_feature_cols.append(diff_col); feature_cols_to_keep.append(diff_col)
    final_df = df[feature_cols_to_keep].copy(); final_df['Team1ID'] = df[id1_col]; final_df['Team2ID'] = df[id2_col] # Add original IDs back
    rows_before = len(final_df); final_df = final_df.dropna(subset=diff_feature_cols); rows_after = len(final_df)
    if rows_before > rows_after: logger.warning(f"Dropped {rows_before - rows_after} rows with NaNs in diff features.")
    logger.debug(f"Matchups core finished. Shape={final_df.shape}, Features={len(diff_feature_cols)}"); return final_df, diff_feature_cols

def create_training_matchups(raw_data_dict, team_stats_df):
    """Creates the training dataset with difference features."""
    if not config_loaded: logger.error("Config not loaded."); return pd.DataFrame(), []
    logger.info("Creating training matchup data..."); all_training_matchups = []; final_feature_names = []
    for gender in ['M', 'W']:
        logger.info(f"--- Creating {gender} training matchups ---"); compact_key, tourney_key = f'{gender}RegularSeasonCompactResults', f'{gender}NCAATourneyCompactResults'; compact_results = pd.concat([raw_data_dict.get(compact_key, pd.DataFrame()), raw_data_dict.get(tourney_key, pd.DataFrame())], ignore_index=True)
        start_year = DATA_START_YEAR_M if gender == 'M' else DATA_START_YEAR_W; compact_results = compact_results[compact_results['Season'] >= start_year]
        if compact_results.empty: logger.warning(f"No compact results for {gender}>={start_year}."); continue
        stats_gender = team_stats_df[team_stats_df['Gender'] == gender].copy();
        if stats_gender.empty: logger.warning(f"No team stats for {gender}."); continue
        logger.debug(f"Creating winner-first matchups for {gender}..."); matchups_w, feature_names_w = _create_matchups_core(compact_results, stats_gender, id1_col='WTeamID', id2_col='LTeamID')
        if not matchups_w.empty:
            matchups_w[TARGET] = 1;
            if not final_feature_names: final_feature_names = feature_names_w
            logger.debug(f"Creating loser-first matchups for {gender}..."); compact_swapped = compact_results.rename(columns={'WTeamID': 'LoserTmp', 'LTeamID': 'WinnerTmp'}).rename(columns={'LoserTmp': 'LTeamID', 'WinnerTmp': 'WTeamID'}); matchups_l, feature_names_l = _create_matchups_core(compact_swapped, stats_gender, id1_col='LTeamID', id2_col='WTeamID')
            if not matchups_l.empty: matchups_l[TARGET] = 0; combined = pd.concat([matchups_w, matchups_l], ignore_index=True); combined['Gender'] = gender; all_training_matchups.append(combined); logger.info(f"Created {len(combined)} training matchups for {gender}.")
            else: logger.warning(f"Could not create loser-first matchups for {gender}."); matchups_w['Gender'] = gender; all_training_matchups.append(matchups_w)
        else: logger.warning(f"Could not create winner-first matchups for {gender}.")
    if not all_training_matchups: logger.error("No training matchups generated."); return pd.DataFrame(), []
    training_data = pd.concat(all_training_matchups, ignore_index=True).pipe(reduce_mem_usage)
    logger.info(f"Saving training data to {TRAIN_DATA_FILE}..."); PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True); training_data.to_parquet(TRAIN_DATA_FILE, index=False); logger.info(f"✅ Training data saved (Shape: {training_data.shape})"); logger.info(f"Using {len(final_feature_names)} difference features."); return training_data, final_feature_names

def create_prediction_matchups(team_stats_df, raw_data_dict, season_to_predict=CURRENT_SEASON):
    """Creates the prediction structure with difference features."""
    if not config_loaded: logger.error("Config not loaded."); return pd.DataFrame(), []
    logger.info(f"Creating prediction matchups for {season_to_predict}..."); all_prediction_matchups = []; final_feature_names = []
    for gender in ['M', 'W']:
        logger.info(f"--- Creating {gender} prediction matchups ---"); stats_pred_season = team_stats_df[(team_stats_df['Season'] == season_to_predict) & (team_stats_df['Gender'] == gender)].copy()
        if stats_pred_season.empty: logger.warning(f"No {gender} stats for {season_to_predict}."); continue
        teams_df = raw_data_dict.get(f'{gender}Teams', pd.DataFrame())
        if teams_df.empty: logger.warning(f"No {gender}Teams file, using teams from stats."); team_ids = sorted(stats_pred_season['TeamID'].unique())
        else: team_ids = sorted(teams_df['TeamID'].unique()) # Assume all teams relevant per rules
        if len(team_ids) < 2: logger.warning(f"<2 teams for {gender}."); continue
        logger.debug(f"Creating combinations for {len(team_ids)} teams ({gender})..."); matchups = list(itertools.combinations(team_ids, 2)); pred_structure = pd.DataFrame(matchups, columns=['Team1ID', 'Team2ID']); pred_structure['Season'] = season_to_predict; logger.debug(f"{len(pred_structure)} raw matchups for {gender}.")
        pred_matchups, feature_names = _create_matchups_core(pred_structure, stats_pred_season, id1_col='Team1ID', id2_col='Team2ID')
        if pred_matchups.empty: logger.warning(f"No prediction matchups after feature calc for {gender}."); continue
        pred_matchups['Team1ID'] = pred_matchups['Team1ID'].astype(int); pred_matchups['Team2ID'] = pred_matchups['Team2ID'].astype(int); pred_matchups['ID'] = pred_matchups.apply(lambda row: f"{int(row['Season'])}_{min(row['Team1ID'], row['Team2ID'])}_{max(row['Team1ID'], row['Team2ID'])}", axis=1)
        swap_mask = pred_matchups['Team1ID'] > pred_matchups['Team2ID']; logger.info(f"Swapping features for {swap_mask.sum()} matchups for {gender} ID format.")
        diff_cols = [col for col in pred_matchups.columns if col.endswith('_Diff')];
        for col in diff_cols: pred_matchups.loc[swap_mask, col] = -pred_matchups.loc[swap_mask, col]
        pred_matchups['Gender'] = gender; all_prediction_matchups.append(pred_matchups);
        if not final_feature_names: final_feature_names = feature_names
    if not all_prediction_matchups: logger.error("No prediction matchups generated."); return pd.DataFrame(), []
    prediction_data = pd.concat(all_prediction_matchups, ignore_index=True).pipe(reduce_mem_usage)
    if not final_feature_names: logger.error("Could not determine feature names."); return pd.DataFrame(), []
    missing_pred_cols = set(final_feature_names) - set(prediction_data.columns);
    if missing_pred_cols: logger.warning(f"Prediction data missing {missing_pred_cols}. Filling 0.");
    for col in missing_pred_cols: prediction_data[col] = 0
    prediction_data[final_feature_names] = prediction_data[final_feature_names].fillna(0)
    logger.info(f"Saving prediction structure data to {TEST_DATA_FILE}..."); PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True); cols_to_keep = ['ID'] + final_feature_names; prediction_data = prediction_data.drop_duplicates(subset=['ID']) # Ensure unique IDs
    prediction_data[cols_to_keep].to_parquet(TEST_DATA_FILE, index=False); logger.info(f"✅ Prediction structure saved (Shape: {prediction_data.shape})"); return prediction_data, final_feature_names

if __name__ == "__main__":
    logger.info("Running feature engineering script directly...");
    try: from src.data_loader import load_raw_data
    except ImportError: logger.error("Cannot run standalone: data_loader import failed."); exit()
    raw_data = load_raw_data(reload=False, download_if_missing=True)
    if raw_data:
        team_stats = create_all_features(raw_data)
        if not team_stats.empty:
            train_matchups, train_features = create_training_matchups(raw_data, team_stats)
            team_stats_for_pred = team_stats.copy()
            if config_loaded and CURRENT_SEASON not in team_stats['Season'].unique():
                 latest_season = team_stats['Season'].max(); logger.warning(f"Using season {latest_season} stats as proxy for {CURRENT_SEASON}.")
                 team_stats_for_pred = team_stats[team_stats['Season']==latest_season].copy()
                 if not team_stats_for_pred.empty: team_stats_for_pred['Season'] = CURRENT_SEASON
                 else: logger.error("No proxy stats found!"); team_stats_for_pred = pd.DataFrame()
            elif not config_loaded: logger.error("Config not loaded, cannot determine prediction season logic.")
            else: logger.info(f"Stats for {CURRENT_SEASON} found.")
            if not team_stats_for_pred.empty: pred_matchups, pred_features = create_prediction_matchups(team_stats_for_pred, raw_data)
            else: logger.error("No stats for prediction.")
        else: logger.error("Team stats failed.")
    else: logger.error("Data loading failed.")