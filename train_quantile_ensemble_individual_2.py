#!/usr/bin/env python
# coding: utf-8
"""
Quantile Ensemble Training for Individual Stations (Stage 7)

This script trains quantile ensemble models for streamflow prediction across
multiple catchment stations. It generates predictions at three quantile levels:
- Regular (median): 50th percentile
- Lower bound (q05): 5th percentile
- Upper bound (q95): 95th percentile

The script processes data by state, trains models for selected stations,
and evaluates performance using MSE and NSE metrics.
"""

import os
import argparse
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm

from src.data import PrepareData
from src.data import plot_catchments, read_data_from_file
from src.window import MultiNumpyWindow, MultiWindow, WindowGenerator
from src.model import Switch_Model, QuantileEnsemble



# ==============================================================================
# COMMAND LINE ARGUMENTS CONFIGURATION
# ==============================================================================

parser = argparse.ArgumentParser(
    description='Train quantile ensemble models for streamflow prediction (Stage 7)'
)

# Data configuration
parser.add_argument(
    '--data-dir', 
    type=str, 
    default='/srv/scratch/z5370003/projects/data/camels/dropbox/', 
    help='Path to the CAMELS dataset directory'
)

# Training configuration
parser.add_argument(
    '--num-runs', 
    type=int, 
    default=1, 
    help='Number of training runs for averaging results'
)

parser.add_argument(
    '--state', 
    type=str, 
    default='SA', 
    help='Australian state/territory to train models on (e.g., SA, NSW, QLD)'
)

# Model window configuration
parser.add_argument(
    '--input-width', 
    type=int, 
    default=5, 
    help='Number of historical timesteps used as input'
)

parser.add_argument(
    '--output-width', 
    type=int, 
    default=5, 
    help='Number of future timesteps to predict'
)

parser.add_argument(
    '--shift', 
    type=int, 
    default=1, 
    help='Temporal shift between input and output windows'
)

parser.add_argument(
    '--n-stations', 
    type=int, 
    default=5, 
    help='Maximum number of stations to train and evaluate'
)



# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

# Parse command line arguments
args = parser.parse_args()
data_dir = args.data_dir
num_runs = args.num_runs

# Load CAMELS dataset (timeseries and catchment metadata)
timeseries_data, summary_data = read_data_from_file(data_dir)

# Initialize data preparation object with scaling and preprocessing
camels_data = PrepareData(timeseries_data, summary_data)

# Optional: Visualize catchment locations on map
# plot_catchments(camels_data, data_dir)


# ==============================================================================
# STATION SELECTION AND FEATURE CONFIGURATION
# ==============================================================================

# Remove duplicate stations from metadata
camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T

# Extract relevant catchment characteristics for station selection
characteristics = camels_data.summary_data[[
    'state_outlet',      # State/territory location
    'station_name',      # Station identifier
    'river_region',      # River basin/region
    'Q5', 'Q95',        # Flow quantiles (5th and 95th percentile)
    'catchment_area',    # Catchment size (km²)
    'lat_outlet',        # Latitude of outlet
    'long_outlet',       # Longitude of outlet
    'runoff_ratio'       # Ratio of runoff to precipitation
]]

# Filter stations by selected state/territory
char_selected = characteristics[(characteristics.state_outlet == args.state)]

# Sort stations by runoff ratio (descending) to prioritize more responsive catchments
selected_stations = char_selected.sort_values(
    'runoff_ratio', 
    ascending=False
).index.tolist()

# Define time series variables for model input
variable_ts = [
    'streamflow_MLd_inclInfilled',          # Target: streamflow in mm/day
    'precipitation_deficit',    # Precipitation minus evapotranspiration
    'year_sin', 'year_cos',    # Seasonal encoding (cyclical time features)
    'tmax_AWAP',               # Maximum temperature from AWAP dataset
    'tmin_AWAP'                # Minimum temperature from AWAP dataset
]

print(f'Training on stations: {selected_stations}')

# Performance metrics to track
summary_cols = ['MSE', 'NSE']  # Mean Squared Error, Nash-Sutcliffe Efficiency

# ==============================================================================
# RESULTS STORAGE INITIALIZATION
# ==============================================================================

# Initialize lists to store results across multiple runs
summary_data_q05 = []   # Results for lower quantile (5th percentile)
summary_data_q95 = []   # Results for upper quantile (95th percentile)
summary_data_reg = []   # Results for regular/median prediction
summary_conf_score = [] # Confidence interval coverage scores

# Dictionary to store trained models for each station
quantile_ensemble = {}

# Create output directory for all results (plots and CSV files)
output_dir = f'results/Stage 8 Revision/{args.state}-{args.output_width}'
os.makedirs(output_dir, exist_ok=True)


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

for n in tqdm.trange(args.num_runs):
    print(f'Starting run {n+1} of {args.num_runs}.')

    # Initialize result collectors for this run
    results_regular = []     # Median predictions
    results_q05 = []         # Lower bound predictions
    results_q95 = []         # Upper bound predictions
    results_conf_score = []  # Confidence interval scores

    print(f"Selected stations: {selected_stations}")
    
    # Iterate through stations up to the maximum specified
    for idx, station in enumerate(selected_stations):

        # Stop if we've processed the requested number of stations
        if idx >= args.n_stations:
            break

        print(f'\nTraining Quantile Ensembles for station {station}.')

        # Load train/test data for this station
        res = camels_data.get_train_val_test(
            source=variable_ts, 
            stations=[station]
        )
        print(f"\n\nCamels data res: {res}\n\n")
        
        # Skip station if data is unavailable or insufficient
        if res is None:
            print(f'Warning: No data available for station {station}, skipping...')
            continue
        else:
            train_df, test_df = res
            print(f'\nStation {station} data loaded successfully.')

        # Create time series windows for training
        multi_window = MultiWindow(
            input_width=args.input_width,      # Historical timesteps
            label_width=args.output_width,     # Future timesteps to predict
            shift=args.shift,                  # Offset between input and output
            train_df=train_df,
            test_df=test_df,
            stations=[station],
            label_columns=['streamflow_MLd_inclInfilled'],  # Target variable
            batch_size=32                       # Batch size for training
        )
        
        # Train quantile ensemble models (3 quantiles: 0.05, 0.5, 0.95)
        print(f'Training QuantileEnsemble models (q05, median, q95) - this may take several minutes...')
        quantile_ensemble[station] = QuantileEnsemble(
            window=multi_window,
            conv_width=args.output_width
        )
        print(f'QuantileEnsemble models trained successfully for station {station}.')
        

        # Extract scaling parameters for inverse transformation
        # Index [1] corresponds to streamflow_MLd_inclInfilled in the feature array
        _min = camels_data.scaler.min_[1]
        _scale = camels_data.scaler.scale_[1]

        # Evaluate model performance on test set for all three quantiles
        summary_regular, summary_q05, summary_q95, conf_score = quantile_ensemble[station].summary(
            station, 
            _min=_min, 
            _scale=_scale, 
            conf_score=True
        )
        
        # Store results for this station
        results_regular.append(summary_regular)
        results_q05.append(summary_q05)
        results_q95.append(summary_q95)
        results_conf_score.append(conf_score)

    
        # Generate visualization only for the first run to avoid redundancy
        if n == 0:
            # Define plotting window (last 2 years of test data)
            plot_start = -365 * 2  # Start 730 days from end
            plot_end = None        # Plot until end
            
            # Create figure for streamflow predictions
            fig = plt.figure(figsize=(20, 5))
            plt.title(station, fontsize=20)
            plt.ylabel('Streamflow (mm/day)', fontsize=18)
            
            # Get predictions and actual values, apply inverse scaling
            predictions = (quantile_ensemble[station].predictions(station) - _min) / _scale
            actual_values = (multi_window.test_windows(station) - _min) / _scale
            
            # Log prediction shapes and MSE for each quantile
            print(f"Station: {station}")
            print(f"Predictions shape: {predictions.shape}, Actual shape: {actual_values.shape}")
            print(f"MSE (q05):    {np.mean((predictions[:, :, 0] - actual_values)**2):.4f}")
            print(f"MSE (median): {np.mean((predictions[:, :, 1] - actual_values)**2):.4f}")
            print(f"MSE (q95):    {np.mean((predictions[:, :, 2] - actual_values)**2):.4f}")

            # Configure plot styling
            plt.rcParams.update({'font.size': 15})

            # Extract date values for x-axis
            df_date = multi_window.test_df[station].reset_index()
            date_values = df_date['date']

            # Plot median prediction and actual values
            ax1 = plt.plot(
                date_values[plot_start:plot_end], 
                predictions[plot_start:plot_end, 0, 1],  # Median prediction
                color='blue', 
                label='Predicted (Median)'
            )
            ax2 = plt.plot(
                date_values[plot_start:plot_end], 
                actual_values[plot_start:plot_end, 0], 
                color='red', 
                label='Actual'
            )
            
            # Add 90% confidence interval (q05 to q95)
            ax3 = plt.fill_between(
                date_values[plot_start:plot_end], 
                predictions[plot_start:plot_end, 0, 0],  # Lower bound (q05)
                predictions[plot_start:plot_end, 0, 2],  # Upper bound (q95)
                color='blue', 
                alpha=0.4, 
                label='90% Confidence Interval'
            )

            # Create legend patches
            red_patch = mpatches.Patch(color='red', label='Actual')
            blue_patch = mpatches.Patch(color='blue', label='Predicted')

            plt.legend()
            plt.tight_layout()
            
            # Save figure to results directory
            plot_path = f'{output_dir}/{station}-streamflow.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved to: {plot_path}")

    # Aggregate results across all stations for this run, handling NaN values
    results_reg = pd.DataFrame(results_regular)[summary_cols].mean(skipna=True).to_dict()
    results_q05 = pd.DataFrame(results_q05)[summary_cols].mean(skipna=True).to_dict()
    results_q95 = pd.DataFrame(results_q95)[summary_cols].mean(skipna=True).to_dict()

    # Calculate average confidence score across all stations
    summary_conf_score.append(np.nanmean(results_conf_score))

    # Store aggregated results for this run
    summary_data_reg.append(results_reg)
    summary_data_q05.append(results_q05)
    summary_data_q95.append(results_q95)


# ==============================================================================
# RESULTS AGGREGATION AND EXPORT
# ==============================================================================

# Convert results lists to DataFrames for analysis
results_reg_df = pd.DataFrame(summary_data_reg)
results_q05_df = pd.DataFrame(summary_data_q05)
results_q95_df = pd.DataFrame(summary_data_q95)
results_conf_score_df = pd.DataFrame(summary_conf_score, columns=['conf_score'])

# Display final aggregated results
print('\n' + '='*80)
print('FINAL RESULTS ACROSS ALL RUNS')
print('='*80)
print('\nMedian Predictions:')
print(results_reg_df)
print('\nLower Quantile (Q05) Predictions:')
print(results_q05_df)
print('\nUpper Quantile (Q95) Predictions:')
print(results_q95_df)
print('\nConfidence Score:')
print(results_conf_score_df)
print('='*80)

# Export results to CSV files
results_reg_df.to_csv(f'{output_dir}/results_reg.csv', index=True)
results_q05_df.to_csv(f'{output_dir}/results_q05.csv', index=True)
results_q95_df.to_csv(f'{output_dir}/results_q95.csv', index=True)
results_conf_score_df.to_csv(f'{output_dir}/results_conf_score.csv', index=True)

print(f'\nResults exported to: {output_dir}')
print('Training complete!')



