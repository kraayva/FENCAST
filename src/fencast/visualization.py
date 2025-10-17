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


def load_energy_prediction_data(study_dir: Path, final_model: bool = False) -> pd.DataFrame:
    """
    Loads energy prediction RMSE data from MLWP evaluation results.
    
    Returns:
        DataFrame with energy prediction RMSE by timedelta
    """
    eval_dir = study_dir / "final_model" / "mlwp_evaluation" if final_model else study_dir / "mlwp_evaluation"
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


def load_energy_prediction_data_per_region(study_dir: Path, final_model: bool = False) -> pd.DataFrame:
    """
    Loads per-region energy prediction RMSE data from MLWP evaluation results.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'region', 'rmse']
    """
    eval_dir = study_dir / "final_model" / "mlwp_evaluation" if final_model else study_dir / "mlwp_evaluation"
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
            if data['rmse'] is not None and 'region_metrics' in data:
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
                
                # Add per-region results
                for region, region_data in data['region_metrics'].items():
                    results.append({
                        'mlwp_model': mlwp_name,
                        'timedelta_days': actual_lead_time_days,
                        'region': region,
                        'rmse': region_data['rmse']
                    })
    
    return pd.DataFrame(results)


class MLWPPlotter:
    """
    Flexible plotting class for MLWP evaluation results.
    """
    
    def __init__(self, config: dict, study_dir: Path, per_region: bool = False, mlwp_name: str = 'pangu', final_model: bool = False):
        self.config = config
        self.study_dir = study_dir
        self.per_region = per_region
        self.mlwp_name = mlwp_name
        self.final_model = final_model
        if per_region:
            self.energy_data = load_energy_prediction_data_per_region(study_dir, final_model)
        else:
            self.energy_data = load_energy_prediction_data(study_dir, final_model)
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
            save_path: Path to save the plot (default: 
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
            ax1.set_ylabel('Solar Capacity Factor RMSE', fontsize=12, color='blue')
            ax2.set_ylabel('Weather Prediction RMSE (Normalized)', fontsize=12, color='red')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Combine legends and position outside plot area
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=8)
            
            plt.title(f'Energy vs Weather Prediction Performance\\n(Study: {self.study_dir.name})', fontsize=14)
            
        else:
            # Single-axis plot for energy predictions only
            if self.per_region:
                # Create two subplots for different region groups with shared y-axis
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
                
                # Define region groups
                group1_regions = {'DE2', 'DE1', 'DEC', 'DEB', 'DE7', 'DEG', 'DED'}
                group2_regions = {'DE3', 'DE4', 'DE6', 'DE8', 'DE9', 'DEA', 'DEE', 'DEF'}
                
                # Plot Group 1 (Southern/Central regions)
                self._plot_energy_predictions(ax1, region_filter=group1_regions)
                self._plot_baselines(ax1, show_persistence, show_climatology, persistence_lead_times, region_filter=group1_regions)
                ax1.set_title('Southern & Central German States')
                ax1.set_xlabel('Forecast Lead Time (Days)')
                ax1.set_ylabel('Solar Capacity Factor RMSE')
                
                # Plot Group 2 (Northern/Eastern regions)
                self._plot_energy_predictions(ax2, region_filter=group2_regions)
                self._plot_baselines(ax2, show_persistence, show_climatology, persistence_lead_times, region_filter=group2_regions)
                ax2.set_title('Northern & Eastern German States')
                ax2.set_xlabel('Forecast Lead Time (Days)')
                ax2.set_ylabel('Solar Capacity Factor RMSE')
                
                # Set same x-ticks for both plots
                x_ticks = sorted(self.energy_data['timedelta_days'].unique())
                ax1.set_xticks(x_ticks)
                ax2.set_xticks(x_ticks)
                
                # Create a single shared legend for all main regions
                self._create_shared_legend(fig, ax1, ax2)
                
                plt.suptitle('Predicted CF RMSE from MLWP Forecasts vs. Lead Time (Per Region)', fontsize=16)
                plt.tight_layout()
                
            else:
                # Original single plot
                plt.figure(figsize=(12, 7))
                
                # Plot energy predictions
                self._plot_energy_predictions(plt.gca())
                
                # Plot baselines
                self._plot_baselines(plt.gca(), show_persistence, show_climatology, persistence_lead_times)
                
                plt.title(f'Predicted CF RMSE from MLWP Forecasts vs. Lead Time')
                plt.xlabel('Forecast Lead Time (Days)')
                plt.ylabel('Solar Capacity Factor RMSE')

                # Handle legend positioning based on number of entries
                handles, labels = plt.gca().get_legend_handles_labels()
                if len(handles) > 10:  # Many entries, place outside
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    plt.legend()
        
        if not self.per_region:
            plt.xticks(self.energy_data['timedelta_days'].unique())
            
            # Use subplots_adjust for better control, especially with many legend entries
            if len(self.energy_data['region'].unique() if 'region' in self.energy_data.columns else []) > 3:
                plt.subplots_adjust(right=0.75)  # Make room for legend
            else:
                plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            # Choose suffix based on plotting mode
            suffix = "regions" if self.per_region else "mean"
            filename = f"{self.mlwp_name}_cf_rmse_{suffix}.png"
            model_subdir = "final_model" if hasattr(self, 'final_model') and self.final_model else "best_model"
            plot_dir = self.study_dir / model_subdir
            plot_dir.mkdir(exist_ok=True)
            save_path = plot_dir / filename
        plt.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
        
    def _get_region_color_and_name(self, region_code):
        """Get color and main region name based on region code."""
        # Extract main region code (first 3 characters)
        main_region = region_code[:3]
        
        # Define color mapping for main regions
        region_colors = {
            'DE1': '#FFD700',      # Yellow - Baden-Württemberg
            'DE2': '#FF69B4',      # Pink - Bayern
            'DE3': '#B22222',      # Brick-red - Berlin
            'DE4': '#87CEEB',      # Light blue - Brandenburg
            'DE6': '#006400',      # Dark-green - Hamburg
            'DE7': '#0000FF',      # Blue - Hessen
            'DE8': '#FFFF00',      # Yellow - Mecklenburg-Vorpommern (lighter than DE1)
            'DE9': '#FFB6C1',      # Light red - Niedersachsen
            'DEA': '#008000',      # Green - Nordrhein-Westfalen
            'DEB': '#FF4500',      # Dark orange - Rheinland-Pfalz
            'DEC': '#FFA500',      # Orange - Saarland
            'DED': '#8B4513',      # Brown - Sachsen
            'DEE': '#90EE90',      # Light green - Sachsen-Anhalt
            'DEF': '#808080',      # Grey - Schleswig-Holstein
            'DEG': '#9ACD32'       # Yellow-green - Thüringen
        }
        
        region_names = {
            'DE1': 'Baden-Württemberg',
            'DE2': 'Bayern',
            'DE3': 'Berlin',
            'DE4': 'Brandenburg',
            'DE6': 'Hamburg',
            'DE7': 'Hessen',
            'DE8': 'Mecklenburg-Vorpommern',
            'DE9': 'Niedersachsen',
            'DEA': 'Nordrhein-Westfalen',
            'DEB': 'Rheinland-Pfalz',
            'DEC': 'Saarland',
            'DED': 'Sachsen',
            'DEE': 'Sachsen-Anhalt',
            'DEF': 'Schleswig-Holstein',
            'DEG': 'Thüringen'
        }
        
        return region_colors.get(main_region, '#000000'), region_names.get(main_region, main_region)

    def _create_shared_legend(self, fig, ax1, ax2):
        """Create a single shared legend containing all main region names."""
        import matplotlib.patches as mpatches
        
        # Get all main regions that appear in the data
        all_regions = set()
        if 'region' in self.energy_data.columns:
            for region in self.energy_data['region'].unique():
                all_regions.add(region[:3])
        
        # Create legend entries for all main regions
        legend_elements = []
        for main_region in sorted(all_regions):
            color, region_name = self._get_region_color_and_name(main_region + '00')  # Dummy full code
            legend_elements.append(mpatches.Patch(color=color, label=region_name))
        
        # Add climatology baseline if it's shown
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        # Check for climatology baseline in either plot
        for handle, label in zip(handles1 + handles2, labels1 + labels2):
            if 'Climatology' in label:
                legend_elements.append(handle)
                break
        
        # Create the shared legend positioned in the center
        fig.legend(handles=legend_elements, 
                  loc='center',
                  bbox_to_anchor=(0.4, 0.2),
                  ncol=2,
                  fontsize=10)

    def _plot_energy_predictions(self, ax, region_filter=None):
        """Plot energy prediction RMSE lines."""
        if self.per_region:
            # Plot one line per region for each MLWP model
            main_regions_labeled = set()  # Track which main regions have been labeled
            
            for mlwp in self.energy_data['mlwp_model'].unique():
                mlwp_data = self.energy_data[self.energy_data['mlwp_model'] == mlwp]
                
                for region in mlwp_data['region'].unique():
                    main_region_code = region[:3]
                    
                    # Apply region filter if specified
                    if region_filter is not None and main_region_code not in region_filter:
                        continue
                    
                    region_data = mlwp_data[mlwp_data['region'] == region]
                    color, main_region_name = self._get_region_color_and_name(region)
                    
                    # Don't create individual labels since we'll use a shared legend
                    label = None
                    
                    ax.plot(region_data['timedelta_days'], region_data['rmse'], 
                           marker='o', linewidth=2, color=color, label=label)
        else:
            # Original behavior: one line per MLWP model (averaged across regions)
            for mlwp in self.energy_data['mlwp_model'].unique():
                data_subset = self.energy_data[self.energy_data['mlwp_model'] == mlwp]
                ax.plot(data_subset['timedelta_days'], data_subset['rmse'], 
                       marker='o', linewidth=2.5, label=f'Solar CF predicted from {mlwp}')
    
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
    
    def _plot_baselines(self, ax, show_persistence: bool, show_climatology: bool, persistence_lead_times: List[int], region_filter=None):
        """Plot baseline comparisons."""
        if show_persistence:
            logger.info("\n=== Calculating Persistence Baselines ===")
            
            # Load data for persistence calculation
            import pandas as pd
            from fencast.utils.paths import PROJECT_ROOT, PROCESSED_DATA_DIR
            
            setup_name = self.config.get('setup_name', 'default_setup')
            
            # Load the same processed data as the persistence script for consistency
            labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
            if not labels_file.exists():
                # Try FFNN labels if CNN labels don't exist
                labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
                
            if not labels_file.exists():
                # Fallback to raw data if processed data doesn't exist
                logger.warning("Processed labels not found, falling back to raw target data")
                gt_file = PROJECT_ROOT / self.config['target_data_raw']
                full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
                full_df.index = full_df.index + pd.Timedelta(hours=12)  # Apply noon-shift
                
                # Drop columns to match the model's target
                drop_cols = self.config.get('data_processing', {}).get('drop_columns', [])
                if drop_cols:
                    full_df = full_df.drop(columns=drop_cols, errors='ignore')
            else:
                # Use processed data (same as persistence script)
                logger.info(f"Using processed data from: {labels_file}")
                full_df = pd.read_parquet(labels_file)
                
            # Filter for the test set years
            test_years = self.config['split_years']['test']
            test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
            
            if self.per_region:
                # Calculate persistence baseline per-region efficiently
                regions_to_plot = []
                for region in test_gt.columns:
                    main_region_code = region[:3]
                    if region_filter is None or main_region_code in region_filter:
                        regions_to_plot.append(region)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(regions_to_plot)))  # Different colormap for baselines
                for i, region in enumerate(regions_to_plot):
                    # Calculate persistence for this region for all lead times at once
                    region_data = test_gt[[region]]
                    # Use None logger to reduce verbosity for per-region calculations
                    region_persistence_results = calculate_persistence_baseline(region_data, persistence_lead_times, None)
                    region_persistence_values = [region_persistence_results[td]['rmse'] for td in persistence_lead_times]
                    
                    ax.plot(persistence_lead_times, region_persistence_values, 
                           color=colors[i], linestyle=':', alpha=0.7, linewidth=1.5, 
                           marker='s', markersize=3, markerfacecolor=colors[i], markeredgecolor=colors[i])
            else:
                # Calculate persistence baselines for all lead times at once
                persistence_results = calculate_persistence_baseline(test_gt, persistence_lead_times, logger)
                persistence_rmse_values = [persistence_results[td]['rmse'] for td in persistence_lead_times]
                
                ax.plot(persistence_lead_times, persistence_rmse_values, 
                       color='red', linestyle=':', alpha=0.7, linewidth=2, 
                       marker='s', markersize=4, markerfacecolor='red', markeredgecolor='red',
                       label="Persistence Baseline")
        
        if show_climatology:
            climatology_rmse = calculate_climatology_baseline(self.config)
            ax.axhline(climatology_rmse, color='green', linestyle=':', alpha=0.7,
                      label=f"Climatology Baseline ({climatology_rmse:.4f})")


def load_energy_prediction_data_seasonal(study_dir: Path, config: dict, final_model: bool = False) -> pd.DataFrame:
    """
    Loads energy prediction RMSE data with seasonal breakdown computed from predictions and ground truth.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'season', 'rmse']
    """
    eval_dir = study_dir / "final_model" / "mlwp_evaluation" if final_model else study_dir / "mlwp_evaluation"
    metric_files = list(eval_dir.glob('**/metrics_*.json'))
    
    if not metric_files:
        logger.error(f"No metric files found in {eval_dir}. Please run mlwp_evaluation.py first.")
        return pd.DataFrame()
    
    # Load ground truth data
    from fencast.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
    setup_name = config.get('setup_name', 'default_setup')
    
    # Try to load processed labels
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    if not labels_file.exists():
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
        
    if not labels_file.exists():
        # Fallback to raw data
        gt_file = PROJECT_ROOT / config['target_data_raw']
        gt_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
        gt_df.index = gt_df.index + pd.Timedelta(hours=12)
        drop_cols = config.get('data_processing', {}).get('drop_columns', [])
        if drop_cols:
            gt_df = gt_df.drop(columns=drop_cols, errors='ignore')
    else:
        gt_df = pd.read_parquet(labels_file)
    
    # Filter for test years
    test_years = config['split_years']['test']
    gt_df = gt_df[gt_df.index.year.isin(test_years)].dropna()
    
    # Add season mapping
    gt_df = gt_df.copy()
    gt_df['season'] = gt_df.index.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring', 
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    results = []
    for f in metric_files:
        mlwp_name = f.parent.name
        timedelta_str = f.stem.replace('metrics_', '')
        
        # Load predictions
        pred_file = f.parent / f"predictions_{timedelta_str}.csv"
        if not pred_file.exists():
            continue
            
        pred_df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        
        # Get forecast lead time
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'forecast_lead_time_days' in data:
                actual_lead_time_days = data['forecast_lead_time_days']
            else:
                try:
                    from fencast.utils.tools import get_mlwp_forecast_lead_time
                    actual_lead_time_days = get_mlwp_forecast_lead_time(mlwp_name, timedelta_str, 'u_component_of_wind')
                except Exception:
                    td_num = int(timedelta_str.replace('td', ''))
                    actual_lead_time_days = (td_num + 1) * 6 / 24
        
        # Align predictions with ground truth
        common_indices = pred_df.index.intersection(gt_df.index)
        pred_aligned = pred_df.loc[common_indices]
        gt_aligned = gt_df.loc[common_indices]
        
        # Calculate seasonal RMSE (mean across all regions)
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_mask = gt_aligned['season'] == season
            if season_mask.sum() == 0:
                continue
                
            # Get seasonal data
            gt_season = gt_aligned[season_mask].drop(columns=['season'])
            pred_season = pred_aligned[season_mask]
            
            # Align columns (both should have same region columns)
            common_cols = gt_season.columns.intersection(pred_season.columns)
            if len(common_cols) == 0:
                continue
                
            gt_values = gt_season[common_cols].values.flatten()
            pred_values = pred_season[common_cols].values.flatten()
            
            # Calculate RMSE
            valid_mask = ~(np.isnan(gt_values) | np.isnan(pred_values))
            if np.sum(valid_mask) > 0:
                seasonal_rmse = np.sqrt(np.mean((gt_values[valid_mask] - pred_values[valid_mask]) ** 2))
                
                results.append({
                    'mlwp_model': mlwp_name,
                    'timedelta_days': actual_lead_time_days,
                    'season': season,
                    'rmse': seasonal_rmse
                })
    
    return pd.DataFrame(results)


def calculate_seasonal_persistence_baseline(config: dict, persistence_lead_times: List[int], logger=None) -> pd.DataFrame:
    """Calculate persistence baseline RMSE for each season."""
    import pandas as pd
    from fencast.utils.paths import PROJECT_ROOT, PROCESSED_DATA_DIR
    
    setup_name = config.get('setup_name', 'default_setup')
    
    # Load processed data
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    if not labels_file.exists():
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
        
    if not labels_file.exists():
        # Fallback to raw data
        gt_file = PROJECT_ROOT / config['target_data_raw']
        full_df = pd.read_csv(gt_file, index_col='Date', parse_dates=True)
        full_df.index = full_df.index + pd.Timedelta(hours=12)
        drop_cols = config.get('data_processing', {}).get('drop_columns', [])
        if drop_cols:
            full_df = full_df.drop(columns=drop_cols, errors='ignore')
    else:
        full_df = pd.read_parquet(labels_file)
    
    # Filter for test years
    test_years = config['split_years']['test']
    test_gt = full_df[full_df.index.year.isin(test_years)].dropna()
    
    # Add season column
    test_gt = test_gt.copy()
    test_gt['season'] = test_gt.index.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring', 
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    results = []
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        season_data = test_gt[test_gt['season'] == season].drop(columns=['season'])
        if len(season_data) > 0:
            persistence_results = calculate_persistence_baseline(season_data, persistence_lead_times, None)
            for td in persistence_lead_times:
                results.append({
                    'timedelta_days': td,
                    'season': season,
                    'rmse': persistence_results[td]['rmse']
                })
    
    return pd.DataFrame(results)


class MLWPSeasonalPlotter:
    """Plotting class for seasonal MLWP evaluation results."""
    
    def __init__(self, config: dict, study_dir: Path, mlwp_name: str = 'pangu', final_model: bool = False):
        self.config = config
        self.study_dir = study_dir
        self.mlwp_name = mlwp_name
        self.final_model = final_model
        self.energy_data = load_energy_prediction_data_seasonal(study_dir, config, final_model)
        
    def plot_seasonal_results(self, 
                             persistence_lead_times: Optional[List[int]] = None,
                             figsize: tuple = (16, 8),
                             save_path: Optional[Path] = None) -> None:
        """Create seasonal MLWP evaluation plot."""
        
        if self.energy_data.empty:
            logger.error("No seasonal energy prediction data available for plotting")
            return
        
        # Set default persistence lead times
        if persistence_lead_times is None:
            max_timedelta = max(self.energy_data['timedelta_days'].max(), 20)
            persistence_lead_times = list(range(1, int(max_timedelta) + 1))
        
        # Calculate seasonal persistence baselines
        logger.info("=== Calculating Seasonal Persistence Baselines ===")
        persistence_data = calculate_seasonal_persistence_baseline(self.config, persistence_lead_times, logger)
        
        # Generate the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Define season colors
        season_colors = {
            'Winter': '#1f77b4',    # Blue
            'Spring': '#2ca02c',    # Green  
            'Summer': '#ff7f0e',    # Orange
            'Autumn': '#d62728'     # Red
        }
        
        # Plot model predictions by season
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_data = self.energy_data[self.energy_data['season'] == season]
            if not season_data.empty:
                plt.plot(season_data['timedelta_days'], season_data['rmse'], 
                        marker='o', linewidth=2.5, color=season_colors[season],
                        label=f'{season} (Model)')
        
        # Plot persistence baselines by season
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_pers = persistence_data[persistence_data['season'] == season]
            if not season_pers.empty:
                plt.plot(season_pers['timedelta_days'], season_pers['rmse'], 
                        marker='s', linewidth=2, linestyle=':', alpha=0.7,
                        color=season_colors[season], label=f'{season} (Persistence)')
        
        plt.title(f'Seasonal Performance: {self.mlwp_name.upper()} Model vs Persistence Baseline')
        plt.xlabel('Forecast Lead Time (Days)')
        plt.ylabel('Solar Capacity Factor RMSE')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-ticks
        plt.xticks(sorted(self.energy_data['timedelta_days'].unique()))
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            filename = f"{self.mlwp_name}_cf_rmse_seasonal.png"
            model_subdir = "final_model" if hasattr(self, 'final_model') and self.final_model else "best_model"
            plot_dir = self.study_dir / model_subdir
            plot_dir.mkdir(exist_ok=True)
            save_path = plot_dir / filename
        plt.savefig(save_path)
        logger.info(f"Seasonal plot saved to: {save_path}")


def create_mlwp_seasonal_plot(config_name: str, 
                             model_type: str, 
                             study_name: str, 
                             persistence_lead_times: Optional[List[int]] = None,
                             figsize: tuple = (16, 8),
                             mlwp_name: str = 'pangu',
                             final_model: bool = False) -> None:
    """Create MLWP seasonal evaluation plot."""
    logger.info(f"--- Creating MLWP Seasonal Plot for study '{study_name}' ---")
    
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find study directory
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir, model_type) if study_name == 'latest' else results_parent_dir / study_name
        logger.info(f"Loading results from: {study_dir}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Create seasonal plotter
    plotter = MLWPSeasonalPlotter(config, study_dir, mlwp_name=mlwp_name, final_model=final_model)
    
    # Create the plot
    plotter.plot_seasonal_results(
        persistence_lead_times=persistence_lead_times,
        figsize=figsize
    )


def load_energy_prediction_rmse_mae_data(study_dir: Path, final_model: bool = False) -> pd.DataFrame:
    """
    Loads RMSE and MAE data from MLWP evaluation results.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'rmse', 'mae']
    """
    eval_dir = study_dir / "final_model" / "mlwp_evaluation" if final_model else study_dir / "mlwp_evaluation"
    metric_files = list(eval_dir.glob('**/metrics_*.json'))
    
    if not metric_files:
        logger.error(f"No metric files found in {eval_dir}. Please run mlwp_evaluation.py first.")
        return pd.DataFrame()
    
    results = []
    for f in metric_files:
        mlwp_name = f.parent.name
        timedelta_str = f.stem.replace('metrics_', '')
        
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        # Get forecast lead time
        if 'forecast_lead_time_days' in data:
            actual_lead_time_days = data['forecast_lead_time_days']
        else:
            try:
                from fencast.utils.tools import get_mlwp_forecast_lead_time
                actual_lead_time_days = get_mlwp_forecast_lead_time(mlwp_name, timedelta_str, 'u_component_of_wind')
            except Exception:
                td_num = int(timedelta_str.replace('td', ''))
                actual_lead_time_days = (td_num + 1) * 6 / 24
        
        # Extract RMSE and MAE
        rmse = data.get('rmse', 0.0)
        mae = data.get('mae', 0.0)
        
        results.extend([
            {
                'mlwp_model': mlwp_name,
                'timedelta_days': actual_lead_time_days,
                'metric_type': 'RMSE',
                'value': rmse
            },
            {
                'mlwp_model': mlwp_name,
                'timedelta_days': actual_lead_time_days,
                'metric_type': 'MAE',
                'value': mae
            }
        ])
    
    return pd.DataFrame(results)


class MLWPRmseMaePlotter:
    """Plotting class for RMSE vs MAE comparison."""
    
    def __init__(self, config: dict, study_dir: Path, mlwp_name: str = 'pangu', final_model: bool = False):
        self.config = config
        self.study_dir = study_dir
        self.mlwp_name = mlwp_name
        self.final_model = final_model
        self.data = load_energy_prediction_rmse_mae_data(study_dir, final_model)
        
    def plot_rmse_mae_comparison(self, 
                                figsize: tuple = (16, 8),
                                save_path: Optional[Path] = None) -> None:
        """Create RMSE vs MAE comparison plot."""
        
        if self.data.empty:
            logger.error("No RMSE/MAE data available for plotting")
            return
        
        # Generate the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Get data for each metric
        rmse_data = self.data[self.data['metric_type'] == 'RMSE']
        mae_data = self.data[self.data['metric_type'] == 'MAE']
        
        # Plot RMSE and MAE
        plt.plot(rmse_data['timedelta_days'], rmse_data['value'], 
                marker='o', linewidth=2.5, color='#1f77b4', label='RMSE')
        plt.plot(mae_data['timedelta_days'], mae_data['value'], 
                marker='s', linewidth=2.5, color='#ff7f0e', label='MAE')
        
        plt.title(f'Model Performance Comparison: {self.mlwp_name.upper()} RMSE vs MAE')
        plt.xlabel('Forecast Lead Time (Days)')
        plt.ylabel('Solar Capacity Factor Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set x-ticks
        plt.xticks(sorted(rmse_data['timedelta_days'].unique()))
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            filename = f"{self.mlwp_name}_cf_rmse_mae.png"
            model_subdir = "final_model" if hasattr(self, 'final_model') and self.final_model else "best_model"
            plot_dir = self.study_dir / model_subdir
            plot_dir.mkdir(exist_ok=True)
            save_path = plot_dir / filename
        plt.savefig(save_path)
        logger.info(f"RMSE vs MAE plot saved to: {save_path}")


def create_mlwp_rmse_mae_plot(config_name: str, 
                             model_type: str, 
                             study_name: str, 
                             figsize: tuple = (16, 8),
                             mlwp_name: str = 'pangu',
                             final_model: bool = False) -> None:
    """Create MLWP RMSE vs MAE comparison plot."""
    logger.info(f"--- Creating MLWP RMSE vs MAE Plot for study '{study_name}' ---")
    
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find study directory
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir, model_type) if study_name == 'latest' else results_parent_dir / study_name
        logger.info(f"Loading results from: {study_dir}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Create RMSE vs MAE plotter
    plotter = MLWPRmseMaePlotter(config, study_dir, mlwp_name=mlwp_name, final_model=final_model)
    
    # Create the plot
    plotter.plot_rmse_mae_comparison(figsize=figsize)


def create_mlwp_plot(config_name: str, 
                     model_type: str, 
                     study_name: str, 
                     weather_rmse_file: Optional[str] = None,
                     show_persistence: bool = True,
                     show_climatology: bool = True, 
                     show_weather_total: bool = True,
                     show_weather_variables: bool = False,
                     persistence_lead_times: Optional[List[int]] = None,
                     figsize: tuple = (16, 8),
                     per_region: bool = False,
                     mlwp_name: str = 'pangu',
                     final_model: bool = False) -> None:
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
    plotter = MLWPPlotter(config, study_dir, per_region=per_region, mlwp_name=mlwp_name, final_model=final_model)
    
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