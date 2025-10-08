# scripts/plot_mlwp.py

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Import our custom modules
from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger, get_latest_study_dir

logger = setup_logger("mlwp_visualization")

def calculate_persistence_baseline(config: dict, timedelta_days: int) -> float:
    """
    Calculates persistence baseline RMSE for a specific forecast lead time.
    
    Args:
        config: Configuration dictionary
        timedelta_days: Forecast lead time in days
    
    Returns:
        float: Persistence RMSE for this lead time
    """
    # Load the full raw target dataset
    gt_file = PROJECT_ROOT / config['target_data_raw']
    full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
    full_df.index = full_df.index + pd.Timedelta(hours=12) # Apply noon-shift
    
    # Drop columns to match the model's target
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        full_df = full_df.drop(columns=drop_cols, errors='ignore')
        
    # Filter for the test set years
    test_years = config['split_years']['test']
    test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
    
    # Persistence baseline: use data from `timedelta_days` days ago
    persistence_preds = test_gt.shift(timedelta_days)
    
    # Find valid overlapping data
    valid_mask = ~(test_gt.isna().any(axis=1) | persistence_preds.isna().any(axis=1))
    valid_gt = test_gt[valid_mask]
    valid_preds = persistence_preds[valid_mask]
    
    if len(valid_gt) > 0:
        rmse = np.sqrt(mean_squared_error(valid_gt, valid_preds))
        
        # Debug logging to understand the strange pattern
        if timedelta_days <= 20:
            logger.info(f"Persistence {timedelta_days}d: RMSE={rmse:.4f}, samples={len(valid_gt)}, "
                       f"date_range={valid_gt.index.min().strftime('%Y-%m-%d')} to {valid_gt.index.max().strftime('%Y-%m-%d')}")
            
            # Check for weekly patterns by analyzing day-of-week correlation
            if timedelta_days in [7, 14]:
                gt_dow = valid_gt.index.dayofweek
                pred_dow = valid_preds.index.dayofweek
                same_dow_ratio = (gt_dow == pred_dow).mean()
                logger.info(f"  Same day-of-week ratio: {same_dow_ratio:.3f} (1.0 = always same weekday)")
        
        return rmse
    else:
        logger.warning(f"No valid data for persistence baseline at {timedelta_days} days")
        return np.nan

def calculate_climatology_baseline(config: dict) -> float:
    """
    Calculates climatology baseline RMSE (same for all lead times).
    """
    # Load the full raw target dataset
    gt_file = PROJECT_ROOT / config['target_data_raw']
    full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
    full_df.index = full_df.index + pd.Timedelta(hours=12) # Apply noon-shift
    
    # Drop columns to match the model's target
    drop_cols = config.get('data_processing', {}).get('drop_columns', [])
    if drop_cols:
        full_df = full_df.drop(columns=drop_cols, errors='ignore')
        
    # Filter for the test set years
    test_years = config['split_years']['test']
    test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
    
    # Climatology baseline: use historical averages (1990-1999)
    historical_data = full_df.loc['1990':'1999']
    daily_climatology = historical_data.groupby([historical_data.index.month, historical_data.index.day]).mean()
    daily_climatology.index.names = ['month', 'day']
    
    preds_df = pd.DataFrame(index=test_gt.index)
    preds_df['month'] = preds_df.index.month
    preds_df['day'] = preds_df.index.day
    
    merged = pd.merge(preds_df, daily_climatology, on=['month', 'day'], how='left').ffill().bfill()
    merged.index = test_gt.index
    climatology_preds = merged[test_gt.columns]
    
    return np.sqrt(mean_squared_error(test_gt, climatology_preds))

def load_weather_rmse_data(weather_rmse_file: Path) -> pd.DataFrame:
    """
    Loads weather RMSE data from the CSV file created by calculate_rmse_mlwp.py
    
    Returns:
        DataFrame with weather RMSE metrics by timedelta
    """
    if not weather_rmse_file.exists():
        logger.warning(f"Weather RMSE file not found: {weather_rmse_file}")
        return pd.DataFrame()
    
    logger.info(f"Loading weather RMSE data from: {weather_rmse_file}")
    weather_df = pd.read_csv(weather_rmse_file)
    
    # Select total_rmse and individual variable RMSE columns for plotting
    if 'total_rmse' in weather_df.columns:
        # Find all variable average RMSE columns
        var_cols = [col for col in weather_df.columns if col.endswith('_avg_rmse')]
        # Convert hours to days if needed
        if 'timedelta_hours' in weather_df.columns:
            weather_df['timedelta_days'] = weather_df['timedelta_hours'] / 24
        base_cols = ['mlwp_model', 'timedelta_days', 'total_rmse']
        return weather_df[base_cols + var_cols].copy()
    else:
        logger.warning("total_rmse column not found in weather RMSE data")
        return pd.DataFrame()

def visualize_results(config_name: str, model_type: str, study_name: str, weather_rmse_file: str = None):
    """
    Aggregates MLWP evaluation metrics from a study and generates a summary plot 
    with optional weather prediction RMSE comparison.
    
    Args:
        config_name: Configuration file name
        model_type: Model architecture type
        study_name: Study name to load results from
        weather_rmse_file: Optional path to weather RMSE CSV file
    """
    logger.info(f"--- Visualizing MLWP Evaluation Results for study '{study_name}' ---")
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find the study directory
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir, model_type) if study_name == 'latest' else results_parent_dir / study_name
        logger.info(f"Loading results from: {study_dir}")
    except FileNotFoundError as e:
        logger.error(e); return

    # Aggregate all metric files
    eval_dir = study_dir / "mlwp_evaluation"
    metric_files = list(eval_dir.glob('**/metrics_*.json'))
    if not metric_files:
        logger.error(f"No metric files found in {eval_dir}. Please run mlwp_evaluation.py first."); return
        
    results = []
    for f in metric_files:
        mlwp_name = f.parent.name
        timedelta_str = f.stem.replace('metrics_td', '')
        with open(f, 'r') as fp:
            data = json.load(fp)
            if data['rmse'] is not None:
                results.append({
                    'mlwp_model': mlwp_name,
                    'timedelta_days': int(timedelta_str), # Direct days, not hours
                    'rmse': data['rmse']
                })
    
    results_df = pd.DataFrame(results)
    
    # Load weather RMSE data if provided
    weather_df = pd.DataFrame()
    if weather_rmse_file:
        weather_file_path = PROJECT_ROOT / weather_rmse_file
        weather_df = load_weather_rmse_data(weather_file_path)
    
    # Generate the plot
    sns.set_theme(style="whitegrid")
    
    # Create figure with dual y-axes if we have weather data
    if not weather_df.empty:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx()
        
        # Calculate persistence baselines for every day from 1 to 20
        max_timedelta = max(results_df['timedelta_days'].max(), 20)
        persistence_timedeltas = list(range(1, int(max_timedelta) + 1))
        persistence_rmse_values = []
        logger.info("\\n=== Calculating Persistence Baselines ===")
        for td in persistence_timedeltas:
            persistence_rmse = calculate_persistence_baseline(config, td)
            persistence_rmse_values.append(persistence_rmse)
        
        # Plot energy prediction RMSE on left axis
        for mlwp in results_df['mlwp_model'].unique():
            data_subset = results_df[results_df['mlwp_model'] == mlwp]
            ax1.plot(data_subset['timedelta_days'], data_subset['rmse'], 
                    marker='o', linewidth=2.5, label=f'{mlwp} (Energy Prediction)')
        
        # Plot weather prediction RMSE on right axis
        colors = ['red', 'orange', 'purple', 'brown', 'pink']  # Colors for different variables
        linestyles = ['--', '-.', ':']
        
        for mlwp in weather_df['mlwp_model'].unique():
            data_subset = weather_df[weather_df['mlwp_model'] == mlwp]
            
            # Plot total RMSE with prominent styling
            ax2.plot(data_subset['timedelta_days'], data_subset['total_rmse'], 
                    marker='s', linewidth=3, linestyle='--', alpha=0.8,
                    color='red', label=f'{mlwp} (Total Weather RMSE)')
            
            # Plot individual variable RMSE with thinner lines
            var_cols = [col for col in weather_df.columns if col.endswith('_avg_rmse')]
            for i, var_col in enumerate(var_cols):
                var_name = var_col.replace('_avg_rmse', '')
                # Use different colors and styles for different variables
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                
                ax2.plot(data_subset['timedelta_days'], data_subset[var_col], 
                        marker='', linewidth=1.5, linestyle=linestyle, alpha=0.6,
                        color=color, label=f'{mlwp} ({var_name})')
        
        # Plot baselines on left axis
        # Variable persistence baseline (smooth curve for all days)
        ax1.plot(persistence_timedeltas, persistence_rmse_values, color='red', linestyle=':', alpha=0.7,
                linewidth=2, label="Persistence Baseline")
        
        # Fixed climatology baseline
        climatology_rmse = calculate_climatology_baseline(config)
        ax1.axhline(climatology_rmse, color='green', linestyle=':', alpha=0.7,
                   label=f"Climatology Baseline ({climatology_rmse:.4f})")
        
        # Customize axes
        ax1.set_xlabel('Forecast Lead Time (Days)', fontsize=12)
        ax1.set_ylabel('Energy Prediction RMSE (Capacity Factor)', fontsize=12, color='blue')
        ax2.set_ylabel('Weather Prediction RMSE (Normalized)', fontsize=12, color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends and position outside plot area
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10)
        
        plt.title(f'Energy vs Weather Prediction Performance\n(Study: {study_dir.name})', fontsize=14)
        
    else:
        # Original single-axis plot if no weather data
        plt.figure(figsize=(12, 7))
        
        # Calculate baselines for every day from 1 to 20
        max_timedelta = max(results_df['timedelta_days'].max(), 20)
        persistence_timedeltas = list(range(1, int(max_timedelta) + 1))
        persistence_rmse_values = []
        logger.info("\\n=== Calculating Persistence Baselines ===")
        for td in persistence_timedeltas:
            persistence_rmse = calculate_persistence_baseline(config, td)
            persistence_rmse_values.append(persistence_rmse)
        climatology_rmse = calculate_climatology_baseline(config)
        
        # Plot a line for each MLWP model's performance
        sns.lineplot(data=results_df, x='timedelta_days', y='rmse', hue='mlwp_model', marker='o', linewidth=2.5)
        
        # Plot baselines
        plt.plot(persistence_timedeltas, persistence_rmse_values, color='red', linestyle=':', 
                linewidth=2, label="Persistence Baseline")
        plt.axhline(climatology_rmse, color='green', linestyle='-.', 
                   label=f"Climatology Baseline (RMSE: {climatology_rmse:.4f})")
        
        plt.title(f'CNN Performance on MLWP Forecasts vs. Lead Time\n(Study: {study_dir.name})')
        plt.xlabel('Forecast Lead Time (Days)')
        plt.ylabel('Energy Prediction RMSE')
        plt.legend()
    
    plt.xticks(results_df['timedelta_days'].unique()) # Ensure ticks are on the data points
    plt.tight_layout()
    
    # Save the plot
    save_path = study_dir / "mlwp_evaluation_plot.png"
    plt.savefig(save_path)
    logger.info(f"Summary plot saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MLWP evaluation results from a study.')
    parser.add_argument('--config', '-c', default='datapp_de', help='Configuration file name.')
    # NOTE: model_type is still needed to find the correct study folder
    parser.add_argument('--model-type', '-m', default='cnn', choices=['cnn'], help='The model architecture evaluated.')
    parser.add_argument('--study-name', '-s', default='latest', help='Study name to load results from.')
    parser.add_argument('--weather-rmse', '-w', default='results/mlwp_weather_rmse.csv', 
                       help='Path to weather RMSE CSV file (relative to project root, default: results/mlwp_weather_rmse.csv)')
    args = parser.parse_args()
    
    visualize_results(
        config_name=args.config,
        model_type=args.model_type,
        study_name=args.study_name,
        weather_rmse_file=args.weather_rmse
    )