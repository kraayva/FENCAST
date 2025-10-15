#!/usr/bin/env python3
"""
Persistence Baseline Script for FENCAST

This script implements a persistence baseline model for solar capacity factor forecasting.
The persistence model assumes that future values will be the same as the current observed value.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fencast.utils.paths import load_config, PROCESSED_DATA_DIR, PROJECT_ROOT
from fencast.utils.tools import setup_logger, calculate_persistence_baseline


# calculate_persistence_baseline is now imported from fencast.utils.tools


def calculate_persistence_by_region(data: pd.DataFrame, lead_times: List[int],
                                  logger=None) -> Dict[int, pd.DataFrame]:
    """
    Calculate persistence baseline metrics for each region separately.
    
    Args:
        data: DataFrame with datetime index and capacity factor columns
        lead_times: List of lead times in days
        logger: Optional logger instance
        
    Returns:
        Dictionary with lead times as keys and DataFrames with region metrics as values
    """
    if logger:
        logger.info("Calculating persistence baseline by region")
    
    results = {}
    
    for lt in lead_times:
        region_results = []
        
        for region in data.columns:
            # Get data for this region only
            region_data = data[[region]].copy()
            
            # Calculate persistence for this region
            region_metrics = calculate_persistence_baseline(region_data, [lt], logger=None)
            
            if lt in region_metrics:
                metrics = region_metrics[lt]
                region_results.append({
                    'region': region,
                    'mse': metrics['mse'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'samples': metrics['samples']
                })
        
        # Convert to DataFrame
        results[lt] = pd.DataFrame(region_results)
        
        if logger:
            avg_rmse = results[lt]['rmse'].mean()
            logger.info(f"Lead time {lt}d: Average RMSE across regions = {avg_rmse:.6f}")
    
    return results


def run_persistence_analysis(config_name: str = 'datapp_de', 
                           lead_times: Optional[List[int]] = None,
                           save_results: bool = True,
                           logger=None) -> Dict:
    """
    Run complete persistence baseline analysis.
    
    Args:
        config_name: Configuration file name
        lead_times: List of lead times in days (uses config default if None)
        save_results: Whether to save results to file
        logger: Optional logger instance
        
    Returns:
        Dictionary with all results
    """
    # Load configuration
    config = load_config(config_name)
    setup_name = config.get('setup_name', 'default_setup')
    
    if logger:
        logger.info(f"Running persistence analysis for setup: {setup_name}")
    
    # Use default lead times if not provided
    if lead_times is None:
        lead_times = [1, 2, 3, 7, 14, 21, 30]  # Default lead times in days
        
    if logger:
        logger.info(f"Lead times: {lead_times}")
    
    # Load processed data
    labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_cnn.parquet"
    if not labels_file.exists():
        # Try FFNN labels if CNN labels don't exist
        labels_file = PROCESSED_DATA_DIR / f"{setup_name}_labels_ffnn.parquet"
        
    if not labels_file.exists():
        raise FileNotFoundError(f"No processed labels found for setup {setup_name}")
    
    if logger:
        logger.info(f"Loading data from: {labels_file}")
    
    data = pd.read_parquet(labels_file)
    if logger:
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Regions: {list(data.columns)}")
    
    # Filter to test years only (for fair comparison with ML models)
    test_years = config.get('split_years', {}).get('test', [])
    if test_years:
        test_mask = data.index.year.isin(test_years)
        data = data[test_mask]
        if logger:
            logger.info(f"Filtered to test years {test_years}: {data.shape[0]} samples")
    
    # Calculate overall persistence metrics
    if logger:
        logger.info("\\n=== Overall Persistence Metrics ===")
    overall_results = calculate_persistence_baseline(data, lead_times, logger)
    
    # Calculate per-region persistence metrics
    if logger:
        logger.info("\\n=== Per-Region Persistence Metrics ===")
    region_results = calculate_persistence_by_region(data, lead_times, logger)
    
    # Compile all results
    all_results = {
        'config': config_name,
        'setup_name': setup_name,
        'lead_times': lead_times,
        'test_years': test_years,
        'data_shape': data.shape,
        'date_range': [str(data.index.min()), str(data.index.max())],
        'overall_metrics': overall_results,
        'region_metrics': {lt: df.to_dict('records') for lt, df in region_results.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results if requested
    if save_results:
        results_dir = PROJECT_ROOT / "results" / setup_name / "persistence_baseline"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"persistence_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=str)
        
        if logger:
            logger.info(f"Results saved to: {results_file}")
        
        # Also save summary table
        summary_data = []
        for lt in lead_times:
            if lt in overall_results:
                summary_data.append({
                    'lead_time_days': lt,
                    'rmse': overall_results[lt]['rmse'],
                    'mse': overall_results[lt]['mse'],
                    'mae': overall_results[lt]['mae'],
                    'samples': overall_results[lt]['samples']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = results_dir / f"persistence_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        if logger:
            logger.info(f"Summary table saved to: {summary_file}")
    
    return all_results


def plot_persistence_results(region_results: Dict[int, pd.DataFrame], 
                           lead_times: List[int],
                           metric: str = 'rmse',
                           save_path: Optional[Path] = None,
                           show_plot: bool = True,
                           logger=None) -> plt.Figure:
    """
    Plot persistence baseline results showing individual regions and overall mean.
    
    Args:
        region_results: Dictionary with lead times as keys and region DataFrames as values
        lead_times: List of lead times to plot
        metric: Metric to plot ('rmse', 'mse', 'mae')
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        logger: Optional logger instance
        
    Returns:
        matplotlib Figure object
    """
    if logger:
        logger.info(f"Creating persistence baseline plot for metric: {metric}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for all regions across lead times
    plot_data = {}
    mean_values = []
    
    for lt in lead_times:
        if lt in region_results:
            df = region_results[lt]
            
            # Plot each region as a thin line
            for _, row in df.iterrows():
                region = row['region']
                value = row[metric]
                
                if region not in plot_data:
                    plot_data[region] = {'lead_times': [], 'values': []}
                
                plot_data[region]['lead_times'].append(lt)
                plot_data[region]['values'].append(value)
            
            # Calculate mean for this lead time
            mean_val = df[metric].mean()
            mean_values.append(mean_val)
        else:
            mean_values.append(np.nan)
    
    # Plot individual regions with thin, light lines
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    
    for i, (region, data) in enumerate(plot_data.items()):
        ax.plot(data['lead_times'], data['values'], 
               color=colors[i], alpha=0.6, linewidth=1, 
               label=region if i < 10 else "")  # Only show first 10 in legend
    
    # Plot mean with thick line
    ax.plot(lead_times, mean_values, 
           color='black', linewidth=3, marker='o', markersize=6,
           label=f'Mean across all regions', zorder=10)
    
    # Formatting
    ax.set_xlabel('Lead Time (days)', fontsize=12)
    ax.set_ylabel(f'{metric.upper()}', fontsize=12)
    ax.set_title(f'Persistence Baseline: {metric.upper()} vs Lead Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(lead_times) - 0.5, max(lead_times) + 0.5)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Keep only the mean line and first few regions in legend
    if len(handles) > 11:
        # Keep mean (last) + first 10 regions
        selected_handles = handles[-1:] + handles[:10]
        selected_labels = labels[-1:] + labels[:10]
        ax.legend(selected_handles, selected_labels, 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add statistics text box
    mean_rmse = np.nanmean(mean_values)
    std_rmse = np.nanstd(mean_values)
    
    stats_text = f'Mean {metric.upper()}: {mean_rmse:.4f} ± {std_rmse:.4f}\n'
    stats_text += f'Lead times: {min(lead_times)}-{max(lead_times)} days\n'
    stats_text += f'Regions: {len(plot_data)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def run_persistence_analysis_with_plot(config_name: str = 'datapp_de',
                                     lead_times: Optional[List[int]] = None,
                                     save_results: bool = True,
                                     create_plot: bool = True,
                                     show_plot: bool = True,
                                     logger=None) -> Dict:
    """
    Run persistence analysis and create visualization.
    
    Args:
        config_name: Configuration file name
        lead_times: List of lead times in days
        save_results: Whether to save results to file
        create_plot: Whether to create visualization
        show_plot: Whether to display the plot
        logger: Optional logger instance
        
    Returns:
        Dictionary with all results
    """
    # Run the analysis
    results = run_persistence_analysis(
        config_name=config_name, 
        lead_times=lead_times, 
        save_results=save_results,
        logger=logger
    )
    
    # Create plot if requested
    if create_plot:
        if logger:
            logger.info("Creating persistence baseline visualization")
        
        region_results = {}
        for lt in results['lead_times']:
            if lt in results['region_metrics']:
                region_results[lt] = pd.DataFrame(results['region_metrics'][lt])
        
        if region_results:
            # Determine save path
            plot_save_path = None
            if save_results:
                results_dir = PROJECT_ROOT / "results" / results['setup_name'] / "persistence_baseline"
                results_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_save_path = results_dir / f"persistence_plot_{timestamp}.png"
            
            # Create plot
            fig = plot_persistence_results(
                region_results=region_results,
                lead_times=results['lead_times'],
                metric='rmse',
                save_path=plot_save_path,
                show_plot=show_plot,
                logger=logger
            )
            
            results['plot_figure'] = fig
        else:
            if logger:
                logger.warning("No region results available for plotting")
    
    return results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Calculate persistence baseline for solar capacity factor forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', '-c', default='datapp_de',
                       help='Configuration file name')
    parser.add_argument('--lead-times', '-lt', nargs='+', type=int,
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       help='Lead times in days')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--plot', action='store_true',
                       help='Create and show visualization plot')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plot (only save)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("persistence_baseline")
    logger.info("Starting persistence baseline analysis")
    
    try:
        if args.plot:
            # Run analysis with plotting
            results = run_persistence_analysis_with_plot(
                config_name=args.config,
                lead_times=args.lead_times,
                save_results=not args.no_save,
                create_plot=True,
                show_plot=not args.no_display,
                logger=logger
            )
        else:
            # Run analysis without plotting
            results = run_persistence_analysis(
                config_name=args.config,
                lead_times=args.lead_times,
                save_results=not args.no_save,
                logger=logger
            )
        
        # Print summary
        print("\\n" + "="*60)
        print("PERSISTENCE BASELINE RESULTS")
        print("="*60)
        
        overall = results['overall_metrics']
        for lt in args.lead_times:
            if lt in overall:
                metrics = overall[lt]
                print(f"Lead time {lt:2d}d: RMSE={metrics['rmse']:.6f}, "
                      f"MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
                      f"Samples={metrics['samples']}")
        
        print("="*60)
        logger.info("Persistence baseline analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
