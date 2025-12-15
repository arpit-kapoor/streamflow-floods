"""
Visualization Module for Streamflow and Flood Risk

This module provides plotting functions for visualizing streamflow predictions,
flood risk analysis, and confidence intervals.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_flood_risk_analysis(station, station_name, date_values, actual_max, 
                             pred_max, alert, threshold, output_path):
    """
    Create comprehensive flood risk analysis visualization with three panels:
    1. Actual streamflow with flood threshold
    2. Predicted streamflow with 90% confidence interval
    3. Flood probability indicator (color-coded risk levels)
    
    Parameters:
    -----------
    station : str
        Station identifier
    station_name : str
        Human-readable station name
    date_values : pd.Series
        Date values for x-axis
    actual_max : np.ndarray
        Actual streamflow values
    pred_max : np.ndarray
        Predicted streamflow values, shape (n_days, 3)
        Columns: [q05, median, q95]
    alert : np.ndarray
        Alert level classifications (-1=Unlikely, 0=Low, 1=Moderate, 2=High)
    threshold : float
        Flood threshold value (mm/day)
    output_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    plt.rcParams.update({'font.size': 15})
    
    # Panel 1: Actual streamflow with threshold
    ax[0].plot(date_values, actual_max, color='red', label='Actual', linewidth=1.5)
    ax[0].hlines(
        y=threshold, 
        xmin=date_values.min(), 
        xmax=date_values.max(), 
        linestyle='dashed', 
        label='Flood Threshold', 
        color='orange',
        linewidth=2
    )
    ax[0].set_ylabel('Streamflow (mm/day)', fontsize=18)
    ax[0].set_ylim(0, max(actual_max.max(), threshold) * 1.1)
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f'{station_name} - Flood Risk Analysis', fontsize=20)
    
    # Panel 2: Predicted streamflow with confidence interval
    ax[1].plot(
        date_values, 
        pred_max[:, 1],
        color='blue', 
        label='Predicted (Median)',
        linewidth=1.5
    )
    ax[1].fill_between(
        date_values, 
        pred_max[:, 0],
        pred_max[:, 2],
        color='magenta', 
        alpha=0.4, 
        label='90% Confidence Interval'
    )
    ax[1].hlines(
        y=threshold, 
        xmin=date_values.min(), 
        xmax=date_values.max(), 
        linestyle='dashed', 
        label='Flood Threshold', 
        color='orange',
        linewidth=2
    )
    ax[1].set_ylabel('Streamflow (mm/day)', fontsize=18)
    ax[1].set_ylim(0, max(actual_max.max(), threshold) * 1.1)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, alpha=0.3)
    
    # Panel 3: Flood probability indicator
    ax[2].scatter(
        date_values[alert == 2], alert[alert == 2], 
        c='red', s=25, marker='x', label='High Chance', alpha=0.8
    )
    ax[2].scatter(
        date_values[alert == 1], alert[alert == 1], 
        c='darkorange', s=20, marker='x', label='Moderate Chance', alpha=0.8
    )
    ax[2].scatter(
        date_values[alert == 0], alert[alert == 0], 
        c='gold', s=20, marker='x', label='Low Chance', alpha=0.8
    )
    ax[2].scatter(
        date_values[alert == -1], alert[alert == -1], 
        c='green', s=5, label='Flood Unlikely', alpha=0.6
    )
    
    ax[2].set_ylabel('Flood Probability Indicator', fontsize=18)
    ax[2].set_ylim(-1.5, 2.5)
    ax[2].set_yticks([-1, 0, 1, 2])
    ax[2].set_yticklabels(['Unlikely', 'Low', 'Moderate', 'High'])
    ax[2].legend(loc='upper right')
    ax[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_streamflow_predictions(date_values, predictions, actuals, station_name,
                                output_path, pred_range=None):
    """
    Create streamflow prediction plot with confidence intervals.
    
    Parameters:
    -----------
    date_values : pd.Series
        Date values for x-axis
    predictions : np.ndarray
        Predicted values, shape (n_days, 3) for [q05, median, q95]
    actuals : np.ndarray
        Actual values
    station_name : str
        Station name for plot title
    output_path : str
        Path to save the plot
    pred_range : tuple, optional
        (start, end) indices for plotting subset. If None, plots all data.
    """
    if pred_range is not None:
        start, end = pred_range
        date_values = date_values[start:end]
        predictions = predictions[start:end]
        actuals = actuals[start:end]
    
    fig, ax = plt.subplots(figsize=(20, 5))
    
    # Plot predictions and actual
    ax.plot(date_values, predictions[:, 1], color='blue', 
            label='Predicted (Median)', linewidth=1.5)
    ax.plot(date_values, actuals, color='red', 
            label='Actual', linewidth=1.5)
    
    # Add confidence interval
    ax.fill_between(
        date_values,
        predictions[:, 0],
        predictions[:, 2],
        color='magenta',
        alpha=0.3,
        label='90% Confidence Interval'
    )
    
    ax.set_title(f'{station_name} - Streamflow Prediction', fontsize=20)
    ax.set_ylabel('Streamflow (mm/day)', fontsize=18)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_simple_comparison_plot(date_values, predicted, actual, 
                                  title, ylabel, output_path):
    """
    Create simple comparison plot between predicted and actual values.
    
    Parameters:
    -----------
    date_values : pd.Series
        Date values for x-axis
    predicted : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
    title : str
        Plot title
    ylabel : str
        Y-axis label
    output_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    
    ax.plot(date_values, predicted, color='blue', label='Predicted', linewidth=1.5)
    ax.plot(date_values, actual, color='red', label='Actual', linewidth=1.5)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
