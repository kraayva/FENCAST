# src/fencast/visualization.py

import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.patches as mpatches

from fencast.utils.paths import load_config, PROJECT_ROOT, PROCESSED_DATA_DIR
from fencast.utils.tools import (
    setup_logger,
    get_latest_study_dir,
    calculate_persistence_baseline,
    calculate_climatology_baseline,
    load_ground_truth_data,
)

logger = setup_logger("mlwp_visualization")


# Helper functions to reduce code duplication

def _load_metrics_files(study_dir: Path, model_name: str) -> List[Path]:
    """Load and validate metrics files from evaluation directory."""
    eval_dir = study_dir / model_name / "mlwp_evaluation"
    metric_files = list(eval_dir.glob('**/metrics_*.json'))
    
    if not metric_files:
        logger.error(f"No metric files found in {eval_dir}. Please run mlwp_evaluation.py first.")
        return []
    
    return metric_files


def _setup_study_directory(config_name: str, study_name: str) -> tuple:
    """Common setup logic for study directory and config loading."""
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    # Find study directory
    results_parent_dir = PROJECT_ROOT / "results" / setup_name
    try:
        study_dir = get_latest_study_dir(results_parent_dir) if study_name == 'latest' else results_parent_dir / study_name
        logger.info(f"Loading results from: {study_dir}")
        return config, study_dir
    except FileNotFoundError as e:
        logger.error(e)
        raise


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


def _process_metrics_file(file_path: Path) -> dict:
    """Process a single metrics file and extract common data."""
    mlwp_name = file_path.parent.name
    timedelta_str = file_path.stem.replace('metrics_', '')
    
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    
    if data['rmse'] is None:
        return None
    
    # Convert timedelta string to days: td00->0.25, td01->0.5, etc.
    if 'forecast_lead_time_days' in data:
        actual_lead_time_days = data['forecast_lead_time_days']
    else:
        td_num = int(timedelta_str.replace('td', ''))
        actual_lead_time_days = (td_num + 1) * 6 / 24  # 6h steps shifted by 1, convert to days
    
    return {
        'mlwp_name': mlwp_name,
        'timedelta_str': timedelta_str,
        'timedelta_days': actual_lead_time_days,
        'data': data
    }


def load_energy_prediction_data(study_dir: Path, model_name: str = "best_model") -> pd.DataFrame:
    """
    Loads energy prediction RMSE data from MLWP evaluation results.
    
    Returns:
        DataFrame with energy prediction RMSE by timedelta
    """
    metric_files = _load_metrics_files(study_dir, model_name)
    if not metric_files:
        return pd.DataFrame()
    
    results = []
    for file_path in metric_files:
        processed = _process_metrics_file(file_path)
        if processed is not None:
            results.append({
                'mlwp_model': processed['mlwp_name'],
                'timedelta_days': processed['timedelta_days'],
                'rmse': processed['data']['rmse']
            })
    
    return pd.DataFrame(results)


def load_energy_prediction_data_per_region(study_dir: Path, model_name: str = "best_model") -> pd.DataFrame:
    """
    Loads per-region energy prediction RMSE data from MLWP evaluation results.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'region', 'rmse']
    """
    metric_files = _load_metrics_files(study_dir, model_name)
    if not metric_files:
        return pd.DataFrame()
    
    results = []
    for file_path in metric_files:
        processed = _process_metrics_file(file_path)
        if processed is not None and 'region_metrics' in processed['data']:
            # Add per-region results
            for region, region_data in processed['data']['region_metrics'].items():
                results.append({
                    'mlwp_model': processed['mlwp_name'],
                    'timedelta_days': processed['timedelta_days'],
                    'region': region,
                    'rmse': region_data['rmse']
                })
    
    return pd.DataFrame(results)


class MLWPPlotter:
    """
    Flexible plotting class for MLWP evaluation results.
    """
    
    def __init__(self, config: dict, 
                 study_dir: Path, 
                 per_region: bool = False, 
                 mlwp_names: list = None, 
                 model_name: str = "best_model"):
        self.config = config
        self.study_dir = study_dir
        self.per_region = per_region
        self.mlwp_names = mlwp_names or ['pangu']
        self.model_name = model_name
        if per_region:
            self.energy_data = load_energy_prediction_data_per_region(study_dir, model_name)
        else:
            self.energy_data = load_energy_prediction_data(study_dir, model_name)
        
        # Filter to only the requested MLWP models
        if not self.energy_data.empty and 'mlwp_model' in self.energy_data.columns:
            self.energy_data = self.energy_data[self.energy_data['mlwp_model'].isin(self.mlwp_names)]
        
        self.weather_data = pd.DataFrame()
        
    def load_weather_data(self, weather_rmse_file: Path):
        """Load weather RMSE data for comparison."""
        self.weather_data = load_weather_rmse_data(weather_rmse_file)

    def _compute_target_stats(self):
        """Compute mean/std of the target (capacity factor) from processed labels used to normalize RMSE.

        Returns (mean, std) tuple for the test set used in plotting.
        If processed labels are not available, returns (None, None).
        """

        setup_name = self.config.get('setup_name', 'default_setup')
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"

        if not labels_file.exists():
            logger.warning("Processed labels not found for normalization; will skip normalization")
            return None, None

        df = pd.read_parquet(labels_file)
        # Restrict to test years
        test_years = self.config['split_years']['test']
        df_test = df[df.index.year.isin(test_years)].dropna()
        if df_test.empty:
            logger.warning("No test-set labels found for normalization; will skip normalization")
            return None, None

        # Compute std across all regions/columns (flatten)
        vals = df_test.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None, None

        return np.mean(vals), np.std(vals)
        
    def plot_mlwp_results(self, 
                         show_persistence: bool = True,
                         show_climatology: bool = True, 
                         show_weather_total: bool = True,
                         show_weather_variables: bool = False,
                         persistence_lead_times: List[int] = list(range(1, 11)),
                         figsize: tuple = (16, 8),
                         save_path: Optional[Path] = None) -> None:
        """
        Create a comprehensive MLWP evaluation plot with flexible options.
        
        Args:
            show_persistence: Whether to show persistence baseline
            show_climatology: Whether to show climatology baseline
            show_weather_total: Whether to show total weather RMSE
            show_weather_variables: Whether to show individual weather variable RMSE
            persistence_lead_times: List of lead times for persistence (default: 1-10)
            figsize: Figure size tuple
            save_path: Path to save the plot (default: results/{setup_name})
        """
        if self.energy_data.empty:
            logger.error("No energy prediction data available for plotting")
            return
        
        # Generate the plot
        sns.set_theme(style="whitegrid")

        # If weather data present and user wants weather variables, offer an option to normalize energy RMSE
        if not self.weather_data.empty and (show_weather_total or show_weather_variables):
            # Compute target mean/std for normalization
            target_mean, target_std = self._compute_target_stats()

            if target_std and target_std > 0:
                logger.info(f"Normalizing energy RMSE and persistence by target std = {target_std:.4f}")
                # When normalized, plot everything on a single axis (same units as weather normalized RMSE)
                fig, ax1 = plt.subplots(figsize=figsize)

                # Plot energy prediction RMSE normalized
                self._plot_energy_predictions(ax1, normalize_by_std=target_std)

                # Plot weather prediction RMSE on same axis (already normalized to ERA5 std)
                # Weather RMSE is dimensionless; energy RMSE divided by energy std becomes comparable
                self._plot_weather_predictions(ax1, show_weather_total, show_weather_variables, use_single_axis=True)

                # Plot baselines normalized as well
                self._plot_baselines(ax1, show_persistence, show_climatology, persistence_lead_times, normalize_by_std=target_std)

                ax1.set_xlabel('Forecast Lead Time (Days)', fontsize=12)
                ax1.set_ylabel('Normalized RMSE (σ units)', fontsize=12)
                ax1.tick_params(axis='y')

                # Combine legends and position outside plot area
                lines, labels = ax1.get_legend_handles_labels()
                ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=8)
                plt.title(f'Normalized Energy vs Weather Prediction Performance\\n(Study: {self.study_dir.name})', fontsize=14)
            else:
                # Fallback to dual-axis plot if we can't compute normalization stats
                fig, ax1 = plt.subplots(figsize=figsize)
                ax2 = ax1.twinx()
                self._plot_energy_predictions(ax1)
                self._plot_weather_predictions(ax2, show_weather_total, show_weather_variables)
                self._plot_baselines(ax1, show_persistence, show_climatology, persistence_lead_times)
                ax1.set_xlabel('Forecast Lead Time (Days)', fontsize=12)
                ax1.set_ylabel('Solar Capacity Factor RMSE', fontsize=12, color='blue')
                ax2.set_ylabel('Weather Prediction RMSE (Normalized)', fontsize=12, color='red')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax2.tick_params(axis='y', labelcolor='red')
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=8)
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
            mlwp_names_str = "_".join(self.mlwp_names)
            filename = f"{mlwp_names_str}_cf_rmse_{suffix}"
            filename += "_weather.png" if not self.weather_data.empty else ".png"
            model_subdir = self.model_name
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

    def _plot_energy_predictions(self, ax, region_filter=None, normalize_by_std: float = None):
        """Plot energy prediction RMSE lines.

        If normalize_by_std is provided, divide RMSE values by this standard deviation
        to create dimensionless sigma-units comparable to weather RMSE.
        """
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
                y = data_subset['rmse'].values
                label = f'Solar CF predicted from {mlwp}'
                if normalize_by_std and normalize_by_std > 0:
                    y = y / normalize_by_std
                    label += ' (normalized)'

                ax.plot(data_subset['timedelta_days'], y, 
                       marker='o', linewidth=2.5, label=label)
    
    def _plot_weather_predictions(self, ax, show_total: bool, show_variables: bool, use_single_axis: bool = False):
        """Plot weather prediction RMSE lines.

        If use_single_axis is True, the plotting will be done on the provided single axis
        (used when energy RMSE is normalized to the same units).
        """
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        linestyles = ['--', '-.', ':']

        for mlwp in self.weather_data['mlwp_model'].unique():
            data_subset = self.weather_data[self.weather_data['mlwp_model'] == mlwp]

            if show_total:
                # Plot total RMSE with prominent styling
                ax.plot(data_subset['timedelta_days'], data_subset['total_rmse'], 
                       marker='s', linewidth=3, linestyle='--', alpha=0.8,
                       color='green', label=f'{mlwp} (Total Weather RMSE)')

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
    
    def _plot_baselines(self, ax, show_persistence: bool, show_climatology: bool, persistence_lead_times: List[int], region_filter=None, normalize_by_std: float = None):
        """Plot baseline comparisons."""
        if show_persistence:
            logger.info("\n=== Calculating Persistence Baselines ===")
            
            # Use processed labels data directly to ensure timestamp alignment with model predictions
            logger.info("Loading processed labels data for persistence baseline...")
            
            setup_name = self.config.get('setup_name', 'default_setup')
            labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
            
            if labels_file.exists():
                # Load processed labels (already at 12:00 timestamps)
                test_gt_all = pd.read_parquet(labels_file)
                logger.info(f"Loaded processed labels with {len(test_gt_all)} timestamps")
                
                # Get timestamps that overlap with model predictions to ensure fair comparison
                if not self.energy_data.empty:
                    # Get model prediction timestamps from the actual predictions files
                    eval_dir = self.study_dir / self.model_name / "mlwp_evaluation"
                    
                    # Find any prediction file to get the timestamp pattern
                    pred_files = list(eval_dir.glob('**/predictions_*.csv'))
                    if pred_files:
                        sample_pred_df = pd.read_csv(pred_files[0], index_col=0, parse_dates=True)
                        model_timestamps = sample_pred_df.index
                        
                        # Filter ground truth to only timestamps that exist in model predictions
                        overlap_timestamps = test_gt_all.index.intersection(model_timestamps)
                        test_gt = test_gt_all.loc[overlap_timestamps]
                        
                        logger.info(f"Using {len(test_gt)} overlapping timestamps for persistence baseline (was {len(test_gt_all)} total)")
                    else:
                        # Fallback: filter by test years
                        test_years = self.config['split_years']['test']
                        test_gt = test_gt_all[test_gt_all.index.year.isin(test_years)]
                        logger.warning("No prediction files found, using test years from processed labels")
                else:
                    # Fallback: filter by test years
                    test_years = self.config['split_years']['test']
                    test_gt = test_gt_all[test_gt_all.index.year.isin(test_years)]
            else:
                # Fallback to original method if processed labels not found
                logger.warning(f"Processed labels not found at {labels_file}, using raw data")
                test_years = self.config['split_years']['test']
                test_gt = load_ground_truth_data(self.config, test_years)

            # Now plot persistence baselines either per-region or overall
            if self.per_region:
                # Calculate persistence baseline per-region efficiently
                regions_to_plot = []
                for region in test_gt.columns:
                    main_region_code = region[:3]
                    if region_filter is None or main_region_code in region_filter:
                        regions_to_plot.append(region)

                colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(regions_to_plot))))  # Different colormap for baselines
                for i, region in enumerate(regions_to_plot):
                    # Calculate persistence for this region for all lead times at once
                    region_data = test_gt[[region]]
                    # Use None logger to reduce verbosity for per-region calculations
                    region_persistence_results = calculate_persistence_baseline(region_data, persistence_lead_times, None)
                    region_persistence_values = [region_persistence_results[td]['rmse'] for td in persistence_lead_times]

                    # If normalization requested, divide persistence values by std
                    if normalize_by_std and normalize_by_std > 0:
                        region_persistence_values = [v / normalize_by_std for v in region_persistence_values]

                    ax.plot(persistence_lead_times, region_persistence_values, 
                           color=colors[i], linestyle=':', alpha=0.7, linewidth=1.5, 
                           marker='s', markersize=3, markerfacecolor=colors[i], markeredgecolor=colors[i])
            else:
                # Calculate persistence baselines for all lead times at once
                persistence_results = calculate_persistence_baseline(test_gt, persistence_lead_times, logger)
                persistence_rmse_values = [persistence_results[td]['rmse'] for td in persistence_lead_times]
                
                label_p = "Persistence Baseline"
                # Normalize persistence if requested
                if normalize_by_std and normalize_by_std > 0:
                    persistence_rmse_values = [v / normalize_by_std for v in persistence_rmse_values]
                    label_p += " (normalized)"
                ax.plot(persistence_lead_times, persistence_rmse_values, 
                       color='red', linestyle=':', alpha=0.7, linewidth=2, 
                       marker='s', markersize=4, markerfacecolor='red', markeredgecolor='red',
                       label=label_p)

        if show_climatology:
            climatology_rmse = calculate_climatology_baseline(self.config)
            ax.axhline(climatology_rmse, color='green', linestyle=':', alpha=0.7,
                      label=f"Climatology Baseline ({climatology_rmse:.4f})")


def load_energy_prediction_data_seasonal(study_dir: Path, config: dict, model_name: str = "best_model") -> pd.DataFrame:
    """
    Loads energy prediction RMSE data with seasonal breakdown computed from predictions and ground truth.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'season', 'rmse']
    """
    metric_files = _load_metrics_files(study_dir, model_name)
    if not metric_files:
        return pd.DataFrame()
    
    # Load ground truth data
    setup_name = config.get('setup_name', 'default_setup')
    
    # Try to load processed labels
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    if not labels_file.exists():
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
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
    for file_path in metric_files:
        processed = _process_metrics_file(file_path)
        if processed is None:
            continue
            
        # Load predictions
        pred_file = file_path.parent / f"predictions_{processed['timedelta_str']}.csv"
        if not pred_file.exists():
            continue
            
        pred_df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        
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
                    'mlwp_model': processed['mlwp_name'],
                    'timedelta_days': processed['timedelta_days'],
                    'season': season,
                    'rmse': seasonal_rmse
                })
    
    return pd.DataFrame(results)


def calculate_seasonal_persistence_baseline(config: dict, persistence_lead_times: List[int], logger=None) -> pd.DataFrame:
    """Calculate persistence baseline RMSE for each season."""
    
    setup_name = config.get('setup_name', 'default_setup')
    
    # Load processed data
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    if not labels_file.exists():
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
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
    
    def __init__(self, config: dict, study_dir: Path, mlwp_names: list = None, model_name: str = "best_model"):
        self.config = config
        self.study_dir = study_dir
        self.mlwp_names = mlwp_names
        self.model_name = model_name
        self.energy_data = load_energy_prediction_data_seasonal(study_dir, config, model_name)
        
        # Filter to only the requested MLWP models
        if not self.energy_data.empty and 'mlwp_model' in self.energy_data.columns:
            self.energy_data = self.energy_data[self.energy_data['mlwp_model'].isin(self.mlwp_names)]
        
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
        
        mlwp_names_str = ", ".join([name.upper() for name in self.mlwp_names])
        plt.title(f'Seasonal Performance: {mlwp_names_str} Model vs Persistence Baseline')
        plt.xlabel('Forecast Lead Time (Days)')
        plt.ylabel('Solar Capacity Factor RMSE')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-ticks
        plt.xticks(sorted(self.energy_data['timedelta_days'].unique()))
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            mlwp_names_str = "_".join(self.mlwp_names)
            filename = f"{mlwp_names_str}_cf_rmse_seasonal.png"
            model_subdir = self.model_name
            plot_dir = self.study_dir / model_subdir
            plot_dir.mkdir(exist_ok=True)
            save_path = plot_dir / filename
        plt.savefig(save_path)
        logger.info(f"Seasonal plot saved to: {save_path}")


def create_mlwp_seasonal_plot(config_name: str, 
                             study_name: str, 
                             persistence_lead_times: Optional[List[int]] = None,
                             figsize: tuple = (16, 8),
                             mlwp_names: list = None,
                             model_name: str = "best_model") -> None:
    """Create MLWP seasonal evaluation plot."""
    logger.info(f"--- Creating MLWP Seasonal Plot for study '{study_name}' ---")
    
    try:
        config, study_dir = _setup_study_directory(config_name, study_name)
    except FileNotFoundError:
        return

    # Create seasonal plotter
    plotter = MLWPSeasonalPlotter(config, study_dir, mlwp_names=mlwp_names, model_name=model_name)
    
    # Create the plot
    plotter.plot_seasonal_results(
        persistence_lead_times=persistence_lead_times,
        figsize=figsize
    )


def load_energy_prediction_rmse_mae_data(study_dir: Path, model_name: str = "best_model") -> pd.DataFrame:
    """
    Loads RMSE and MAE data from MLWP evaluation results.
    
    Returns:
        DataFrame with columns: ['mlwp_model', 'timedelta_days', 'rmse', 'mae']
    """
    metric_files = _load_metrics_files(study_dir, model_name)
    if not metric_files:
        return pd.DataFrame()
    
    results = []
    for file_path in metric_files:
        processed = _process_metrics_file(file_path)
        if processed is not None:
            # Extract RMSE and MAE
            rmse = processed['data'].get('rmse', 0.0)
            mae = processed['data'].get('mae', 0.0)
            
            results.extend([
                {
                    'mlwp_model': processed['mlwp_name'],
                    'timedelta_days': processed['timedelta_days'],
                    'metric_type': 'RMSE',
                    'value': rmse
                },
                {
                    'mlwp_model': processed['mlwp_name'],
                    'timedelta_days': processed['timedelta_days'],
                    'metric_type': 'MAE',
                    'value': mae
                }
            ])
    
    return pd.DataFrame(results)


class MLWPRmseMaePlotter:
    """Plotting class for RMSE vs MAE comparison."""
    
    def __init__(self, config: dict, study_dir: Path, mlwp_names: list = None, model_name: str = "best_model"):
        self.config = config
        self.study_dir = study_dir
        self.mlwp_names = mlwp_names or ['pangu']
        self.model_name = model_name
        self.data = load_energy_prediction_rmse_mae_data(study_dir, model_name)
        
        # Filter to only the requested MLWP models
        if not self.data.empty and 'mlwp_model' in self.data.columns:
            self.data = self.data[self.data['mlwp_model'].isin(self.mlwp_names)]
        
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
        
        mlwp_names_str = ", ".join([name.upper() for name in self.mlwp_names])
        plt.title(f'Model Performance Comparison: {mlwp_names_str} RMSE vs MAE')
        plt.xlabel('Forecast Lead Time (Days)')
        plt.ylabel('Solar Capacity Factor Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set x-ticks
        plt.xticks(sorted(rmse_data['timedelta_days'].unique()))
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            mlwp_names_str = "_".join(self.mlwp_names)
            filename = f"{mlwp_names_str}_cf_rmse_mae.png"
            model_subdir = self.model_name
            plot_dir = self.study_dir / model_subdir
            plot_dir.mkdir(exist_ok=True)
            save_path = plot_dir / filename
        plt.savefig(save_path)
        logger.info(f"RMSE vs MAE plot saved to: {save_path}")


def create_mlwp_rmse_mae_plot(config_name: str, 
                             study_name: str, 
                             figsize: tuple = (16, 8),
                             mlwp_names: list = None,
                             model_name: str = "best_model") -> None:
    """Create MLWP RMSE vs MAE comparison plot."""
    logger.info(f"--- Creating MLWP RMSE vs MAE Plot for study '{study_name}' ---")
    
    try:
        config, study_dir = _setup_study_directory(config_name, study_name)
    except FileNotFoundError:
        return

    # Create RMSE vs MAE plotter
    plotter = MLWPRmseMaePlotter(config, study_dir, mlwp_names=mlwp_names, model_name=model_name)
    
    # Create the plot
    plotter.plot_rmse_mae_comparison(figsize=figsize)


def create_mlwp_plot(config_name: str, 
                     study_name: str, 
                     weather_rmse_file: Optional[str] = None,
                     show_persistence: bool = True,
                     show_climatology: bool = True, 
                     show_weather_total: bool = True,
                     show_weather_variables: bool = False,
                     persistence_lead_times: Optional[List[int]] = None,
                     figsize: tuple = (16, 8),
                     per_region: bool = False,
                     mlwp_names: list = None,
                     model_name: str = "best_model") -> None:
    """
    High-level function to create MLWP evaluation plots.
    
    Args:
        config_name: Configuration file name
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
    
    try:
        config, study_dir = _setup_study_directory(config_name, study_name)
    except FileNotFoundError:
        return

    # Create plotter instance
    plotter = MLWPPlotter(config, study_dir, per_region=per_region, mlwp_names=mlwp_names, model_name=model_name)
    
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