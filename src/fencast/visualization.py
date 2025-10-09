# src/fencast/visualization.py

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from fencast.utils.paths import load_config, PROJECT_ROOT
from fencast.utils.tools import setup_logger, get_latest_study_dir, calculate_persistence_baseline, calculate_climatology_baseline

logger = setup_logger("mlwp_visualization")


def load_weather_rmse_data(weather_rmse_file: Path) -> pd.DataFrame:
    """
    Loads weather RMSE data from the CSV file created by MLWP analysis.
    
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


def load_energy_prediction_data(study_dir: Path) -> pd.DataFrame:
    """
    Loads energy prediction RMSE data from MLWP evaluation results.
    
    Returns:
        DataFrame with energy prediction RMSE by timedelta
    """
    eval_dir = study_dir / "mlwp_evaluation"
    metric_files = list(eval_dir.glob('**/metrics_*.json'))
    
    if not metric_files:
        logger.error(f"No metric files found in {eval_dir}. Please run mlwp_evaluation.py first.")
        return pd.DataFrame()
        
    results = []
    for f in metric_files:
        mlwp_name = f.parent.name
        timedelta_str = f.stem.replace('metrics_', '')  # Get 'td03', 'td07', etc.
        with open(f, 'r') as fp:
            data = json.load(fp)
            if data['rmse'] is not None:
                # Try to get forecast lead time from saved metrics first
                if 'forecast_lead_time_days' in data:
                    actual_lead_time_days = data['forecast_lead_time_days']
                else:
                    # Try to get actual forecast lead time from MLWP files
                    try:
                        from fencast.utils.tools import get_mlwp_forecast_lead_time
                        # Use any variable to get the forecast lead time (all should be the same)
                        actual_lead_time_days = get_mlwp_forecast_lead_time(mlwp_name, timedelta_str, 'u_component_of_wind')
                    except Exception:
                        # Fallback: extract number from filename and convert
                        td_num = int(timedelta_str.replace('td', ''))
                        actual_lead_time_days = (td_num + 1) * 6 / 24  # 6h steps shifted by 1, convert to days
                
                results.append({
                    'mlwp_model': mlwp_name,
                    'timedelta_days': actual_lead_time_days,
                    'rmse': data['rmse']
                })
    
    return pd.DataFrame(results)


class MLWPPlotter:
    """
    Flexible plotting class for MLWP evaluation results.
    """
    
    def __init__(self, config: dict, study_dir: Path):
        self.config = config
        self.study_dir = study_dir
        self.energy_data = load_energy_prediction_data(study_dir)
        self.weather_data = pd.DataFrame()
        
    def load_weather_data(self, weather_rmse_file: Path):
        """Load weather RMSE data for comparison."""
        self.weather_data = load_weather_rmse_data(weather_rmse_file)
        
    def plot_mlwp_results(self, 
                         show_persistence: bool = True,
                         show_climatology: bool = True, 
                         show_weather_total: bool = True,
                         show_weather_variables: bool = False,
                         persistence_lead_times: Optional[List[int]] = None,
                         figsize: tuple = (16, 8),
                         save_path: Optional[Path] = None) -> None:
        """
        Create a comprehensive MLWP evaluation plot with flexible options.
        
        Args:
            show_persistence: Whether to show persistence baseline
            show_climatology: Whether to show climatology baseline
            show_weather_total: Whether to show total weather RMSE
            show_weather_variables: Whether to show individual weather variable RMSE
            persistence_lead_times: List of lead times for persistence (default: 1-20)
            figsize: Figure size tuple
            save_path: Path to save the plot (default: study_dir/mlwp_evaluation_plot.png)
        """
        if self.energy_data.empty:
            logger.error("No energy prediction data available for plotting")
            return
            
        # Set default persistence lead times
        if persistence_lead_times is None:
            max_timedelta = max(self.energy_data['timedelta_days'].max(), 20)
            persistence_lead_times = list(range(1, int(max_timedelta) + 1))
        
        # Generate the plot
        sns.set_theme(style="whitegrid")
        
        # Create figure with dual y-axes if we have weather data
        if not self.weather_data.empty and (show_weather_total or show_weather_variables):
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()
            
            # Plot energy prediction RMSE on left axis
            self._plot_energy_predictions(ax1)
            
            # Plot weather prediction RMSE on right axis
            self._plot_weather_predictions(ax2, show_weather_total, show_weather_variables)
            
            # Plot baselines on left axis
            self._plot_baselines(ax1, show_persistence, show_climatology, persistence_lead_times)
            
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
            
            plt.title(f'Energy vs Weather Prediction Performance\\n(Study: {self.study_dir.name})', fontsize=14)
            
        else:
            # Single-axis plot for energy predictions only
            plt.figure(figsize=(12, 7))
            
            # Plot energy predictions
            self._plot_energy_predictions(plt.gca())
            
            # Plot baselines
            self._plot_baselines(plt.gca(), show_persistence, show_climatology, persistence_lead_times)
            
            plt.title(f'CNN Performance on MLWP Forecasts vs. Lead Time\\n(Study: {self.study_dir.name})')
            plt.xlabel('Forecast Lead Time (Days)')
            plt.ylabel('Energy Prediction RMSE')
            plt.legend()
        
        plt.xticks(self.energy_data['timedelta_days'].unique())
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.study_dir / "mlwp_evaluation_plot.png"
        plt.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
        
    def _plot_energy_predictions(self, ax):
        """Plot energy prediction RMSE lines."""
        for mlwp in self.energy_data['mlwp_model'].unique():
            data_subset = self.energy_data[self.energy_data['mlwp_model'] == mlwp]
            ax.plot(data_subset['timedelta_days'], data_subset['rmse'], 
                   marker='o', linewidth=2.5, label=f'{mlwp} (Energy Prediction)')
    
    def _plot_weather_predictions(self, ax, show_total: bool, show_variables: bool):
        """Plot weather prediction RMSE lines."""
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        linestyles = ['--', '-.', ':']
        
        for mlwp in self.weather_data['mlwp_model'].unique():
            data_subset = self.weather_data[self.weather_data['mlwp_model'] == mlwp]
            
            if show_total:
                # Plot total RMSE with prominent styling
                ax.plot(data_subset['timedelta_days'], data_subset['total_rmse'], 
                       marker='s', linewidth=3, linestyle='--', alpha=0.8,
                       color='red', label=f'{mlwp} (Total Weather RMSE)')
            
            if show_variables:
                # Plot individual variable RMSE with thinner lines
                var_cols = [col for col in self.weather_data.columns if col.endswith('_avg_rmse')]
                for i, var_col in enumerate(var_cols):
                    var_name = var_col.replace('_avg_rmse', '')
                    color = colors[i % len(colors)]
                    linestyle = linestyles[i % len(linestyles)]
                    
                    ax.plot(data_subset['timedelta_days'], data_subset[var_col], 
                           marker='', linewidth=1.5, linestyle=linestyle, alpha=0.6,
                           color=color, label=f'{mlwp} ({var_name})')
    
    def _plot_baselines(self, ax, show_persistence: bool, show_climatology: bool, persistence_lead_times: List[int]):
        """Plot baseline comparisons."""
        if show_persistence:
            logger.info("\\n=== Calculating Persistence Baselines ===")
            persistence_rmse_values = []
            for td in persistence_lead_times:
                persistence_rmse = calculate_persistence_baseline(self.config, td, logger)
                persistence_rmse_values.append(persistence_rmse)
            
            ax.plot(persistence_lead_times, persistence_rmse_values, 
                   color='red', linestyle=':', alpha=0.7, linewidth=2, 
                   label="Persistence Baseline")
        
        if show_climatology:
            climatology_rmse = calculate_climatology_baseline(self.config)
            ax.axhline(climatology_rmse, color='green', linestyle=':', alpha=0.7,
                      label=f"Climatology Baseline ({climatology_rmse:.4f})")


def create_mlwp_plot(config_name: str, 
                     model_type: str, 
                     study_name: str, 
                     weather_rmse_file: Optional[str] = None,
                     show_persistence: bool = True,
                     show_climatology: bool = True, 
                     show_weather_total: bool = True,
                     show_weather_variables: bool = False,
                     persistence_lead_times: Optional[List[int]] = None,
                     figsize: tuple = (16, 8)) -> None:
    """
    High-level function to create MLWP evaluation plots.
    
    Args:
        config_name: Configuration file name
        model_type: Model architecture type
        study_name: Study name to load results from
        weather_rmse_file: Optional path to weather RMSE CSV file
        show_persistence: Whether to show persistence baseline
        show_climatology: Whether to show climatology baseline
        show_weather_total: Whether to show total weather RMSE
        show_weather_variables: Whether to show individual weather variable RMSE
        persistence_lead_times: List of lead times for persistence
        figsize: Figure size tuple
    """
    logger.info(f"--- Creating MLWP Evaluation Plot for study '{study_name}' ---")
    
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find the study directory
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir, model_type) if study_name == 'latest' else results_parent_dir / study_name
        logger.info(f"Loading results from: {study_dir}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Create plotter instance
    plotter = MLWPPlotter(config, study_dir)
    
    # Load weather data if provided
    if weather_rmse_file:
        weather_file_path = PROJECT_ROOT / weather_rmse_file
        plotter.load_weather_data(weather_file_path)
    
    # Create the plot
    plotter.plot_mlwp_results(
        show_persistence=show_persistence,
        show_climatology=show_climatology,
        show_weather_total=show_weather_total,
        show_weather_variables=show_weather_variables,
        persistence_lead_times=persistence_lead_times,
        figsize=figsize
    )