#!/usr/bin/env python
# coding: utf-8
"""Evaluate different training strategies for streamflow prediction models.

This script compares three training approaches:
1. Individual LSTM - Train separate models for each station
2. Batch-Temporal LSTM - Train on multiple stations with temporal batching
3. Batch-Static LSTM - Train on multiple stations with static catchment attributes
"""

import itertools
import argparse
import pandas as pd
from datetime import datetime

from src.data import PrepareData, plot_catchments, read_data_from_file
from src.window import WindowGenerator, MultiNumpyWindow, MultiWindow
from src.model import Base_Model

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(
    description='Evaluate training strategies for streamflow prediction'
)
parser.add_argument(
    '--data-dir',
    type=str,
    default='/srv/scratch/z5370003/projects/data/camels/dropbox',
    help='Path to the CAMELS data directory'
)
parser.add_argument(
    '--num-runs',
    type=int,
    default=3,
    help='Number of training runs for averaging results'
)

args = parser.parse_args()

data_dir = args.data_dir
num_runs = args.num_runs

# ==============================================================================
# DATA LOADING
# ==============================================================================

print(f'Loading data from: {data_dir}')
timeseries_data, summary_data = read_data_from_file(data_dir)

# Initialize data preparation with feature engineering
camels_data = PrepareData(timeseries_data, summary_data)

# Visualize catchment locations (optional)
plot_catchments(camels_data, data_dir)

# ==============================================================================
# STATION SELECTION
# ==============================================================================

# Remove duplicate station metadata
camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T

# Select all stations in South Australia
selected_stations = list(
    camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index
)
print(f'Selected {len(selected_stations)} stations in SA')

# ==============================================================================
# STRATEGY EVALUATION
# ==============================================================================

# Available models: 'multi-LSTM', 'multi-linear', 'multi-CNN', 'multi-Bidirectional-LSTM'

# ==============================================================================
# STRATEGY 1: INDIVIDUAL LSTM (One model per station)
# ==============================================================================
print('\n' + '='*80)
print('STRATEGY 1: INDIVIDUAL LSTM')
print('='*80)

combined = []
for i in range(num_runs):
    print(f'\nRun {i+1}/{num_runs}')
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base = list(itertools.product(*[input_widths, label_widths, selected_stations, models, variables]))

    results_baseModels_variables = []
    models_baseModels_variables = []
    errors_baseModels_variables = []

    for input_width, label_width, station, model_name, variable in permutations_base:
        if input_width < label_width:
            continue

        train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)

        try:
            print('input_width:{}, label_width:{}, station:{}, model:{}, variables:{}'.format(input_width, label_width, station, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=label_width,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=station,
                                     label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=window, CONV_WIDTH=label_width)

            results_baseModels_variables.append(model.summary())

            pd.DataFrame(results_baseModels_variables).to_csv('results_files/results_ensemble_all_1.csv')

        except Exception as e:
            print(f'Error training {model_name} for {station}: {e}')
            errors_baseModels_variables.append([input_width, label_width, station, model_name])
        
        # Note: break here limits evaluation to first station only (for testing)
        break

    Individual_SA = pd.DataFrame(results_baseModels_variables)
    # Results aggregation commented out for individual analysis
    # Individual_SA = Individual_SA.mean()
    # Individual_SA = Individual_SA.to_dict()
    # combined.append(Individual_SA)


# ==============================================================================
# STRATEGY 2: BATCH-TEMPORAL LSTM (Multiple stations, temporal features only)
# ==============================================================================
print('\n' + '='*80)
print('STRATEGY 2: BATCH-TEMPORAL LSTM')
print('='*80)

combined = []
for i in range(num_runs):
    print(f'\nRun {i+1}/{num_runs}')
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations, discard=0.5)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_all_1.csv')
        except Exception as e:
            print(f'Error in batch training: {e}')
            errors_baseModels_batch.append([input_width, label_width, model_name])
            
    Batch_SA = pd.DataFrame(results_baseModels_batch)
    Batch_SA = Batch_SA.mean()
    Batch_SA = Batch_SA.to_dict()
    combined.append(Batch_SA)

print(f'\nBatch-Temporal results: {combined}')



# ==============================================================================
# STRATEGY 3: BATCH-STATIC LSTM (Multiple stations with static attributes)
# ==============================================================================
print('\n' + '='*80)
print('STRATEGY 3: BATCH-STATIC LSTM')
print('='*80)

combined = []
for i in range(num_runs):
    print(f'\nRun {i+1}/{num_runs}')
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP', 'q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_static_all_1.csv')
        except Exception as e:
            print(f'Error in batch-static training: {e}')
            errors_baseModels_batch.append([input_width, label_width, model_name])
            
    Batch_SA = pd.DataFrame(results_baseModels_batch)
    Batch_SA = Batch_SA.mean()
    Batch_SA = Batch_SA.to_dict()
    combined.append(Batch_SA)

print(f'\nBatch-Static results: {combined}')
print('\nEvaluation complete!')