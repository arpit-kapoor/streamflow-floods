"""
Flood Risk Assessment Module

This module provides functions for calculating flood thresholds and classifying
flood risk levels based on quantile predictions.
"""

import numpy as np
import pandas as pd
import os


def calc_flood_threshold(station, train_df, scaler_min, scaler_scale, top=0.20):
    """
    Calculate flood threshold for a station based on historical data.
    
    Uses annual maximum streamflow values and exceedance probability to
    determine a threshold that represents significant flood events.
    
    Parameters:
    -----------
    station : str
        Station identifier
    train_df : pd.DataFrame
        Training data containing streamflow timeseries
    scaler_min : float
        Minimum value from scaler for inverse transformation
    scaler_scale : float
        Scale value from scaler for inverse transformation
    top : float
        Exceedance probability threshold (default 0.20 = top 20% of events)
    
    Returns:
    --------
    tuple : (threshold_prob, threshold, threshold_year)
        Probability, streamflow threshold value, and representative year
    """
    # Extract and scale streamflow data
    streamflow_data_train = train_df[station][['streamflow_MLd_inclInfilled']]
    streamflow_data_train = (streamflow_data_train - scaler_min) / scaler_scale
    
    # Calculate annual maximum streamflow
    streamflow_data = streamflow_data_train.copy()
    streamflow_data.loc[:, 'year'] = streamflow_data.index.year
    streamflow_data = streamflow_data.groupby('year').agg({
        'streamflow_MLd_inclInfilled': 'max'
    })
    n_years = len(streamflow_data)
    
    # Calculate exceedance probability
    streamflow_data = streamflow_data.sort_values(
        by='streamflow_MLd_inclInfilled', 
        ascending=False
    )
    streamflow_data.loc[:, 'exceedance_prob'] = (
        (np.arange(n_years) + 1) / (n_years + 2)
    )
    
    # Find threshold corresponding to specified exceedance probability
    # Filter for events exceeding the threshold probability
    threshold_candidates = streamflow_data.loc[
        streamflow_data.exceedance_prob > top
    ]
    
    # Check if any candidates exist
    if len(threshold_candidates) == 0:
        # If no candidates, use the highest streamflow value (rarest event)
        threshold_point = streamflow_data.reset_index().values[0]
    else:
        # Use the first candidate (lowest streamflow among those exceeding threshold)
        threshold_point = threshold_candidates.reset_index().values[0]
    
    threshold_year = threshold_point[0]
    threshold = threshold_point[1]
    threshold_prob = threshold_point[2]
    
    return threshold_prob, threshold, threshold_year


def calculate_flood_alerts(pred_max, threshold):
    """
    Classify flood risk levels based on quantile predictions and threshold.
    
    Alert levels are determined by which quantile predictions exceed the threshold:
    - High (2): Even the lower quantile (q05) exceeds threshold
    - Moderate (1): Median exceeds threshold
    - Low (0): Only upper quantile (q95) exceeds threshold
    - Unlikely (-1): No quantiles exceed threshold
    
    Parameters:
    -----------
    pred_max : np.ndarray
        Maximum predictions across forecast horizon, shape (n_days, 3)
        Columns: [q05, median, q95]
    threshold : float
        Flood threshold value
    
    Returns:
    --------
    np.ndarray : Alert levels
        -1 = Unlikely, 0 = Low, 1 = Moderate, 2 = High
    """
    alert = np.array([-1] * len(pred_max))
    
    alert[pred_max[:, 2] > threshold] = 0  # q95 exceeds
    alert[pred_max[:, 1] > threshold] = 1  # median exceeds
    alert[pred_max[:, 0] > threshold] = 2  # q05 exceeds (all quantiles exceed)
    
    return alert


def save_flood_outputs(station_output_dir, pred_max, actual_max, alert, threshold):
    """
    Save prediction outputs and flood alerts to text files.
    
    Parameters:
    -----------
    station_output_dir : str
        Output directory for station files
    pred_max : np.ndarray
        Maximum predictions, shape (n_days, 3) with columns [q05, median, q95]
    actual_max : np.ndarray
        Actual maximum values
    alert : np.ndarray
        Alert level classifications (-1, 0, 1, 2)
    threshold : float
        Flood threshold value
    """
    os.makedirs(station_output_dir, exist_ok=True)
    
    np.savetxt(f'{station_output_dir}/pred_q05.txt', pred_max[:, 0])
    np.savetxt(f'{station_output_dir}/pred_median.txt', pred_max[:, 1])
    np.savetxt(f'{station_output_dir}/pred_q95.txt', pred_max[:, 2])
    np.savetxt(f'{station_output_dir}/actual.txt', actual_max)
    np.savetxt(f'{station_output_dir}/alert_levels.txt', alert, fmt='%d')
    np.savetxt(f'{station_output_dir}/threshold.txt', [threshold])


def calculate_alert_statistics(alert):
    """
    Calculate distribution of flood alert levels.
    
    Parameters:
    -----------
    alert : np.ndarray
        Alert level classifications
    
    Returns:
    --------
    dict : Alert counts and percentages
        Dictionary with keys for each alert level
    """
    total = len(alert)
    
    return {
        'High': {
            'count': int(np.sum(alert == 2)),
            'percentage': float(np.sum(alert == 2) / total * 100)
        },
        'Moderate': {
            'count': int(np.sum(alert == 1)),
            'percentage': float(np.sum(alert == 1) / total * 100)
        },
        'Low': {
            'count': int(np.sum(alert == 0)),
            'percentage': float(np.sum(alert == 0) / total * 100)
        },
        'Unlikely': {
            'count': int(np.sum(alert == -1)),
            'percentage': float(np.sum(alert == -1) / total * 100)
        }
    }


def print_alert_summary(alert_stats):
    """
    Print formatted summary of flood alert distribution.
    
    Parameters:
    -----------
    alert_stats : dict
        Alert statistics from calculate_alert_statistics()
    """
    print(f'\nFlood Alert Distribution:')
    for level in ['High', 'Moderate', 'Low', 'Unlikely']:
        count = alert_stats[level]['count']
        pct = alert_stats[level]['percentage']
        print(f'  {level} Chance:      {count:4d} days ({pct:.1f}%)')


def calculate_prediction_quality(pred_max, actual_max):
    """
    Calculate MSE for each quantile prediction.
    
    Parameters:
    -----------
    pred_max : np.ndarray
        Maximum predictions, shape (n_days, 3)
    actual_max : np.ndarray
        Actual maximum values
    
    Returns:
    --------
    dict : MSE values for each quantile
    """
    return {
        'q05': float(np.mean((pred_max[:, 0] - actual_max)**2)),
        'median': float(np.mean((pred_max[:, 1] - actual_max)**2)),
        'q95': float(np.mean((pred_max[:, 2] - actual_max)**2))
    }
