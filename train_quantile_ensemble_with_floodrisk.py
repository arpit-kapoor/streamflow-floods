#!/usr/bin/env python
# coding: utf-8
"""
Quantile Ensemble Training with Flood Risk Indicator

This script trains quantile ensemble models for streamflow prediction and
generates flood risk indicators for each station. It combines:
- Quantile ensemble training (q05, median, q95)
- Flood threshold calculation based on historical data
- Flood risk classification (High/Moderate/Low/Unlikely)
- Comprehensive visualization of predictions and risk levels

The flood risk indicator uses the quantile predictions to classify flood
probability into four categories based on exceedance of historical thresholds.
"""

import os
import argparse
import pandas as pd
import numpy as np
import tqdm

from src.data import PrepareData, read_data_from_file
from src.window import MultiWindow
from src.model import QuantileEnsemble
from src.flood_risk import (
    calc_flood_threshold,
    calculate_flood_alerts,
    save_flood_outputs,
    calculate_alert_statistics,
    print_alert_summary,
    calculate_prediction_quality
)
from src.visualization import plot_flood_risk_analysis


# ==============================================================================
# CONSTANTS
# ==============================================================================

LABEL_COLUMN = 'streamflow_MLd_inclInfilled'


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_and_prepare_data(data_dir):
    """
    Load CAMELS dataset and initialize data preparation object.
    
    Parameters:
    -----------
    data_dir : str
        Path to the CAMELS dataset directory
    
    Returns:
    --------
    camels_data : PrepareData
        Initialized data preparation object
    """
    timeseries_data, summary_data = read_data_from_file(data_dir)
    camels_data = PrepareData(timeseries_data, summary_data)
    return camels_data


def select_stations(camels_data, state):
    """
    Select and filter stations by state.
    
    Parameters:
    -----------
    camels_data : PrepareData
        Data preparation object with station metadata
    state : str
        State/territory code to filter by
    
    Returns:
    --------
    tuple : (selected_stations, station_names, characteristics)
        List of station IDs, station names, and full characteristics DataFrame
    """
    # Remove duplicate stations from metadata
    camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T
    
    # Extract relevant catchment characteristics
    characteristics = camels_data.summary_data[[
        'state_outlet', 'station_name', 'river_region',
        'Q5', 'Q95', 'catchment_area',
        'lat_outlet', 'long_outlet', 'runoff_ratio'
    ]]
    
    # Filter by state and sort by runoff ratio
    char_selected = characteristics[characteristics.state_outlet == state]
    selected_stations = char_selected.sort_values(
        'runoff_ratio', 
        ascending=False
    ).index.tolist()
    
    station_names = char_selected.station_name
    
    return selected_stations, station_names, characteristics





def train_station_model(station, camels_data, variable_ts, args):
    """
    Train quantile ensemble model for a single station.
    
    Parameters:
    -----------
    station : str
        Station identifier
    camels_data : PrepareData
        Data preparation object
    variable_ts : list
        List of timeseries variable names
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    tuple : (quantile_ensemble, multi_window, test_min, test_scale) or None
        Trained model, window object, and scaling parameters, or None if failed
    """
    # Load train/test data for this station
    res = camels_data.get_train_val_test(
        source=variable_ts, 
        stations=[station]
    )
    
    if res is None:
        print(f'Warning: No data available for station {station}, skipping...')
        return None
    
    train_df_single, test_df_single = res
    print(f'Data loaded: Train shape={train_df_single.shape}, Test shape={test_df_single.shape}')
    
    # Create time series windows
    multi_window = MultiWindow(
        input_width=args.input_width,
        label_width=args.output_width,
        shift=args.shift,
        train_df=train_df_single,
        test_df=test_df_single,
        stations=[station],
        label_columns=[LABEL_COLUMN],
        batch_size=32
    )
    
    # Train quantile ensemble models
    print(f'Training QuantileEnsemble models (q05, median, q95)...')
    quantile_model = QuantileEnsemble(
        window=multi_window,
        conv_width=args.output_width
    )
    print(f'✓ Models trained successfully')
    
    return quantile_model, multi_window


def evaluate_station_model(station, quantile_model, test_min, test_scale):
    """
    Evaluate model performance for a station.
    
    Parameters:
    -----------
    station : str
        Station identifier
    quantile_model : QuantileEnsemble
        Trained quantile ensemble model
    test_min : float
        Minimum value for inverse scaling
    test_scale : float
        Scale value for inverse scaling
    
    Returns:
    --------
    tuple : (summary_regular, summary_q05, summary_q95, conf_metrics)
        Performance summaries for each quantile and confidence metrics dictionary
    """
    summary_regular, summary_q05, summary_q95, conf_metrics = quantile_model.summary(
        station, 
        _min=test_min, 
        _scale=test_scale, 
        conf_score=True
    )
    
    print(f'Performance - MSE: {summary_regular["MSE"]:.4f}, NSE: {summary_regular["NSE"]:.4f}')
    print(f'Confidence Metrics:')
    print(f'  ISSS (R²-like):  {conf_metrics["ISSS"]:.4f}  (1.0=perfect, 0.0=baseline)')
    print(f'  Raw IS:          {conf_metrics["raw_IS"]:.4f}  (lower is better)')
    print(f'  Coverage:        {conf_metrics["coverage"]:.2%}  (target: 90%)')
    
    return summary_regular, summary_q05, summary_q95, conf_metrics





def process_flood_risk_for_station(station, station_name, quantile_model, multi_window,
                                   test_min, test_scale, threshold, args, output_dir):
    """
    Calculate flood risk indicators and generate visualizations for a station.
    
    Parameters:
    -----------
    station : str
        Station identifier
    station_name : str
        Human-readable station name
    quantile_model : QuantileEnsemble
        Trained quantile ensemble model
    multi_window : MultiWindow
        Window object containing test data
    test_min : float
        Minimum value for inverse scaling
    test_scale : float
        Scale value for inverse scaling
    threshold : float
        Flood threshold value
    args : argparse.Namespace
        Command line arguments
    output_dir : str
        Base output directory
    """
    print(f'\nCalculating flood threshold...')
    print(f'✓ Flood threshold: {threshold:.4f}')
    
    # Get predictions and actual values
    predictions = (quantile_model.predictions(station) - test_min) / test_scale
    actual_values = (multi_window.test_windows(station) - test_min) / test_scale
    
    # Debug: Print shapes and sample values
    print(f'\n[DEBUG] Shapes before max:')
    print(f'  predictions.shape: {predictions.shape}')
    print(f'  actual_values.shape: {actual_values.shape}')
    print(f'  test_min: {test_min:.4f}, test_scale: {test_scale:.4f}')
    
    # Use maximum values across forecast horizon for flood detection
    pred_max = predictions.max(axis=1)
    actual_max = actual_values.max(axis=1)
    
    # Debug: Print shapes after max
    print(f'\n[DEBUG] Shapes after max:')
    print(f'  pred_max.shape: {pred_max.shape}')
    print(f'  actual_max.shape: {actual_max.shape}')
    print(f'  pred_max sample (first 3): {pred_max[:3] if len(pred_max.shape) == 1 else pred_max[:3, :]}')
    print(f'  actual_max sample (first 3): {actual_max[:3]}')
    
    # Calculate and display MSE for each quantile
    print(f'\nPrediction Quality:')
    print(f'  MSE (q05):    {np.mean((pred_max[:, 0] - actual_max)**2):.4f}')
    print(f'  MSE (median): {np.mean((pred_max[:, 1] - actual_max)**2):.4f}')
    print(f'  MSE (q95):    {np.mean((pred_max[:, 2] - actual_max)**2):.4f}')
    
    # Extract and align date values
    df_date = multi_window.test_df[station].reset_index()
    date_values = df_date['date'][args.input_width + args.shift - 1:]
    
    # Ensure date_values matches prediction length
    if len(date_values) > len(pred_max):
        date_values = date_values[:len(pred_max)]
    elif len(date_values) < len(pred_max):
        pred_max = pred_max[:len(date_values)]
        actual_max = actual_max[:len(date_values)]
    
    # Calculate flood alerts
    alert = calculate_flood_alerts(pred_max, threshold)
    
    # Display alert distribution
    alert_counts = {
        'High': np.sum(alert == 2),
        'Moderate': np.sum(alert == 1),
        'Low': np.sum(alert == 0),
        'Unlikely': np.sum(alert == -1)
    }
    print(f'\nFlood Alert Distribution:')
    print(f'  High Chance:      {alert_counts["High"]:4d} days ({alert_counts["High"]/len(alert)*100:.1f}%)')
    print(f'  Moderate Chance:  {alert_counts["Moderate"]:4d} days ({alert_counts["Moderate"]/len(alert)*100:.1f}%)')
    print(f'  Low Chance:       {alert_counts["Low"]:4d} days ({alert_counts["Low"]/len(alert)*100:.1f}%)')
    print(f'  Flood Unlikely:   {alert_counts["Unlikely"]:4d} days ({alert_counts["Unlikely"]/len(alert)*100:.1f}%)')
    
    # Save outputs
    station_output_dir = f'{output_dir}/{station}'
    save_flood_outputs(station_output_dir, pred_max, actual_max, alert, threshold)
    
    # Generate visualization
    print(f'\nGenerating flood risk visualization...')
    plot_path = f'{station_output_dir}/flood_risk_analysis.png'
    plot_flood_risk_analysis(
        station, station_name, date_values, actual_max,
        pred_max, alert, threshold, plot_path,
        label_column=LABEL_COLUMN
    )
    print(f'✓ Plot saved to: {plot_path}')


def export_results(output_dir, summary_data_reg, summary_data_q05, 
                  summary_data_q95, summary_conf_score, flood_thresholds, summary_cols,
                  individual_station_results=None):
    """
    Export aggregated results to CSV files.
    
    Parameters:
    -----------
    output_dir : str
        Output directory for results
    summary_data_reg : list
        Median prediction results
    summary_data_q05 : list
        Lower quantile results
    summary_data_q95 : list
        Upper quantile results
    summary_conf_score : list
        Confidence metrics (list of dictionaries with ISSS, raw_IS, coverage)
    flood_thresholds : dict
        Flood thresholds by station
    summary_cols : list
        Column names for summary metrics
    individual_station_results : list, optional
        List of dictionaries containing individual station results per run
    """
    # Convert to DataFrames
    results_reg_df = pd.DataFrame(summary_data_reg)
    results_q05_df = pd.DataFrame(summary_data_q05)
    results_q95_df = pd.DataFrame(summary_data_q95)
    results_conf_score_df = pd.DataFrame(summary_conf_score)
    
    # Display results
    print('\n' + '='*80)
    print('FINAL RESULTS ACROSS ALL RUNS')
    print('='*80)
    print('\nMedian Predictions:')
    print(results_reg_df)
    print('\nLower Quantile (Q05) Predictions:')
    print(results_q05_df)
    print('\nUpper Quantile (Q95) Predictions:')
    print(results_q95_df)
    print('\nConfidence Metrics:')
    print(results_conf_score_df)
    print('='*80)
    
    # Export to CSV
    results_reg_df.to_csv(f'{output_dir}/results_reg.csv', index=True)
    results_q05_df.to_csv(f'{output_dir}/results_q05.csv', index=True)
    results_q95_df.to_csv(f'{output_dir}/results_q95.csv', index=True)
    results_conf_score_df.to_csv(f'{output_dir}/results_conf_metrics.csv', index=True)
    
    # Save flood thresholds
    threshold_df = pd.DataFrame.from_dict(
        flood_thresholds, 
        orient='index', 
        columns=['threshold']
    )
    threshold_df.to_csv(f'{output_dir}/flood_thresholds.csv')
    
    # Export individual station results if provided
    if individual_station_results:
        individual_df = pd.DataFrame(individual_station_results)
        individual_df.to_csv(f'{output_dir}/results_individual_stations.csv', index=False)
        
        print('\n' + '='*80)
        print('INDIVIDUAL STATION RESULTS')
        print('='*80)
        print(individual_df.to_string())
        print('='*80)
    
    print(f'\nResults exported to: {output_dir}')


# ==============================================================================
# COMMAND LINE ARGUMENTS CONFIGURATION
# ==============================================================================

parser = argparse.ArgumentParser(
    description='Train quantile ensemble models with flood risk indicators'
)

# Data configuration
parser.add_argument('--data-dir', type=str, 
                    default='/srv/scratch/z5370003/projects/data/camels/dropbox',
                    help='Path to the data directory')
parser.add_argument('--state', type=str, default='SA',
                    help='State to train the model on (ignored if --station-ids is provided)')
parser.add_argument('--station-ids', type=str, nargs='+', default=None,
                    help='List of station IDs to process (e.g., 108003A 108004A). If provided, --state is ignored.')
parser.add_argument('--n-stations', type=int, default=10,
                    help='Number of stations to process')

# Model configuration
parser.add_argument('--input-width', type=int, default=5,
                    help='Input width for the window')
parser.add_argument('--output-width', type=int, default=5,
                    help='Output width for the window')
parser.add_argument('--shift', type=int, default=5,
                    help='Shift for the window')
parser.add_argument('--num-runs', type=int, default=1,
                    help='Number of training runs')

# Flood risk configuration  
parser.add_argument('--flood-threshold-percentile', type=float, default=0.90,
                    help='Percentile for flood threshold (0-1)')

args = parser.parse_args()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function for training quantile ensemble models with flood risk analysis.
    """
    print('='*80)
    print('QUANTILE ENSEMBLE TRAINING WITH FLOOD RISK INDICATORS')
    print('='*80)
    
    # Load CAMELS data
    print(f'\nLoading CAMELS data from: {args.data_dir}')
    timeseries_data, summary_data = read_data_from_file(args.data_dir)
    camels_data = PrepareData(timeseries_data, summary_data)
    print('✓ Data loaded successfully')
    
    # Select stations: use provided station IDs or select by state
    if args.station_ids:
        print(f'\nUsing manually specified station IDs: {args.station_ids}')
        selected_stations = args.station_ids
        
        # Remove duplicate stations from metadata
        camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T
        
        # Get station names for the specified IDs
        station_names = camels_data.summary_data.loc[selected_stations, 'station_name']
        characteristics = camels_data.summary_data[[
            'state_outlet', 'station_name', 'river_region',
            'Q5', 'Q95', 'catchment_area',
            'lat_outlet', 'long_outlet', 'runoff_ratio'
        ]]
        
        # Validate that all station IDs exist
        missing_stations = [s for s in selected_stations if s not in camels_data.summary_data.index]
        if missing_stations:
            print(f'Warning: The following station IDs were not found in the dataset: {missing_stations}')
            selected_stations = [s for s in selected_stations if s in camels_data.summary_data.index]
            station_names = station_names[station_names.index.isin(selected_stations)]
        
        print(f'✓ Using {len(selected_stations)} manually specified stations')
    else:
        print(f'\nSelecting stations from state: {args.state}')
        selected_stations, station_names, characteristics = select_stations(
            camels_data, 
            args.state
        )
        print(f'✓ Selected {len(selected_stations)} stations in {args.state}')
    
    # Define time series variables
    variable_ts = [
        LABEL_COLUMN,
        'precipitation_deficit',
        'year_sin', 'year_cos',
        'tmax_AWAP',
        'tmin_AWAP'
    ]
    
    # Performance metrics to track
    summary_cols = ['MSE', 'NSE']
    
    # Initialize results storage
    summary_data_q05 = []
    summary_data_q95 = []
    summary_data_reg = []
    summary_conf_score = []
    quantile_ensemble = {}
    flood_thresholds = {}
    individual_station_results = []  # Track individual station results
    
    # Create output directory
    if args.station_ids:
        output_dir = f'results/Stage_FloodRisk/manual-{args.output_width}'
    else:
        output_dir = f'results/Stage_FloodRisk/{args.state}-{args.output_width}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all training data once for threshold calculation
    print('\nLoading training/test data for all stations...')
    train_df, test_df = camels_data.get_train_val_test(
        source=variable_ts, 
        stations=selected_stations
    )
    print('✓ Data loaded')

    # Main training loop
    for n in tqdm.trange(args.num_runs):
        print(f'\nStarting run {n+1} of {args.num_runs}.')
        
        # Initialize result collectors for this run
        results_regular = []
        results_q05 = []
        results_q95 = []
        results_conf_score = []
        
        # Process each station
        for idx, station in enumerate(selected_stations):
            # Stop if we've processed the requested number of stations
            if idx >= args.n_stations:
                break
            
            print(f'\n{"="*80}')
            print(f'Processing Station {idx+1}/{min(args.n_stations, len(selected_stations))}: {station}')
            station_display_name = station_names[station] if station in station_names.index else "Unknown"
            print(f'Station Name: {station_display_name}')
            print(f'{"="*80}')
            
            # Train model for this station
            model_result = train_station_model(station, camels_data, variable_ts, args)
            if model_result is None:
                continue
            
            quantile_model, multi_window = model_result
            quantile_ensemble[station] = quantile_model
            
            # Get scaling parameters for individual station
            # scaler_test has only 6 features (one station), streamflow is at index 1
            test_min = camels_data.scaler_test.min_[1]
            test_scale = camels_data.scaler_test.scale_[1]
            
            # Evaluate model
            summary_regular, summary_q05, summary_q95, conf_metrics = evaluate_station_model(
                station, quantile_model, test_min, test_scale
            )
            
            # Calculate flood threshold (compute for all runs to ensure consistency)
            threshold_prob, threshold, threshold_year = calc_flood_threshold(
                station, train_df, test_min, test_scale,
                top=args.flood_threshold_percentile
            )
            # Store threshold only on first run to avoid duplicates
            if n == 0:
                flood_thresholds[station] = threshold
            
            # Store individual station results
            station_result = {
                'run': n + 1,
                'station_id': station,
                'station_name': station_display_name,
                'MSE_median': summary_regular['MSE'],
                'NSE_median': summary_regular['NSE'],
                'MSE_q05': summary_q05['MSE'],
                'NSE_q05': summary_q05['NSE'],
                'MSE_q95': summary_q95['MSE'],
                'NSE_q95': summary_q95['NSE'],
                'ISSS': conf_metrics['ISSS'],
                'raw_IS': conf_metrics['raw_IS'],
                'coverage': conf_metrics['coverage'],
                'flood_threshold': threshold,
                'threshold_percentile': args.flood_threshold_percentile
            }
            individual_station_results.append(station_result)
            
            # Store results for aggregation
            results_regular.append(summary_regular)
            results_q05.append(summary_q05)
            results_q95.append(summary_q95)
            results_conf_score.append(conf_metrics)
            
            # Process flood risk visualization and outputs (only on first run)
            if n == 0:
                process_flood_risk_for_station(
                    station, station_display_name, quantile_model, multi_window,
                    test_min, test_scale, threshold, args, output_dir
                )
        
        # Aggregate results for this run, handling NaN values appropriately
        if len(results_regular) > 0:
            results_reg = pd.DataFrame(results_regular)[summary_cols].mean(skipna=True).to_dict()
            results_q05_agg = pd.DataFrame(results_q05)[summary_cols].mean(skipna=True).to_dict()
            results_q95_agg = pd.DataFrame(results_q95)[summary_cols].mean(skipna=True).to_dict()
            
            # Calculate mean confidence metrics from list of dictionaries
            conf_metrics_df = pd.DataFrame(results_conf_score)
            conf_metrics_mean = conf_metrics_df.mean(skipna=True).to_dict()
            
            # Store aggregated results
            summary_data_reg.append(results_reg)
            summary_data_q05.append(results_q05_agg)
            summary_data_q95.append(results_q95_agg)
            summary_conf_score.append(conf_metrics_mean)
            
            print(f'\n✓ Run {n+1} complete - MSE (median): {results_reg["MSE"]:.4f}, ISSS: {conf_metrics_mean["ISSS"]:.4f}')
    
    # Export all results
    if summary_data_reg:
        export_results(
            output_dir, summary_data_reg, summary_data_q05,
            summary_data_q95, summary_conf_score, flood_thresholds, summary_cols,
            individual_station_results=individual_station_results
        )
        print('\nTraining and flood risk analysis complete!')
    else:
        print('\nNo valid results obtained. Please check your data and configuration.')



if __name__ == '__main__':
    main()
