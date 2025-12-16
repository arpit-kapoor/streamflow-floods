#!/usr/bin/env python
# coding: utf-8
"""Evaluate different model architectures for streamflow prediction.

This script compares various neural network architectures:
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- Bidirectional LSTM
- Linear models
"""

import itertools
import argparse
import pandas as pd
from datetime import datetime

from src.data import PrepareData, plot_catchments, read_data_from_file
from src.window import WindowGenerator
from src.model import Base_Model

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(
    description='Evaluate model architectures for streamflow prediction (Stage 2)'
)
parser.add_argument('--data-dir', type=str, default='/srv/scratch/z5370003/projects/data/camels/dropbox', help='Path to the data directory')
parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')

args = parser.parse_args()

data_dir = args.data_dir
num_runs = args.num_runs

# ==============================================================================
# DATA LOADING
# ==============================================================================

print(f'Loading data from: {data_dir}')
timeseries_data, summary_data = read_data_from_file(data_dir)

# Initialize data preparation
camels_data = PrepareData(timeseries_data, summary_data)

# Visualize catchments (optional)
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

# Available architectures: 'multi-LSTM', 'multi-linear', 'multi-CNN', 'multi-Bidirectional-LSTM'

# ==============================================================================
# ARCHITECTURE EVALUATION
# ==============================================================================

combined = []
for i in range(0, num_runs):
    print('RUN', i)
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
        
        # Note: break limits evaluation to first station (for testing)
        break

    Individual_SA = pd.DataFrame(results_baseModels_variables)
    # Results aggregation commented out for individual analysis
    # Individual_SA = Individual_SA.mean()
    # Individual_SA = Individual_SA.to_dict()
    # combined.append(Individual_SA)

print('\nArchitecture evaluation complete!')