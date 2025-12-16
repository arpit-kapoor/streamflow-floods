"""Data loading and preprocessing utilities for CAMELS-AUS streamflow data."""

import geopandas as gpd
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Constants
MISSING_VALUE_INDICATOR = -99.99
MIN_YEAR = 1990
DATA_START_DATE = dt.datetime(1980, 1, 1)
DATA_END_DATE = dt.datetime(2015, 1, 1)
MAX_MISSING_DATA_THRESHOLD = 0.1  # 10%
YEAR_SECONDS = 365.2425 * 24 * 60 * 60
FLOOD_PROBABILITY_THRESHOLD = 0.05
DAYS_IN_YEAR = 365

STATE_COLOR_MAPPING = {
    'QLD': 'darkblue',
    'NSW': 'black',
    'SA': 'purple',
    'VIC': 'darkred',
    'ACT': 'darkgreen',
    'WA': 'darkorange',
    'NT': 'brown',
    'TAS': 'blue'
}


def read_data_from_file(data_dir):
    """Read timeseries and summary data from CSV files in the data directory.
    
    Args:
        data_dir: Path to the directory containing CAMELS-AUS data files
        
    Returns:
        tuple: (timeseries_data, summary_data) as pandas DataFrames
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no valid data files are found
    """
    timeseries_dfs = []
    summary_dfs = []
    
    # Validate data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please check the --data-dir argument or set the correct path."
        )
    
    # Read all CSV files recursively from directory
    csv_pattern = os.path.join(data_dir, '**', '*.csv')
    for file_path in glob.glob(csv_pattern, recursive=True):
        # Skip checkpoint and temporary files
        if '.ipynb_checkpoints' in file_path or 'checkpoint' in file_path.lower():
            continue
            
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue
    
        # Skip quality code files
        if file_name in ['streamflow_QualityCodes']:
            continue
    
        # Categorize as timeseries or summary data
        if 'year' in df.columns:
            df['source'] = file_name
            df = df[df['year'] > MIN_YEAR]
            df = df.drop_duplicates(['year', 'month', 'day'])
            timeseries_dfs.append(df)
        else:
            df = df.rename({'ID': 'station_id'}, axis=1)
            df = df.set_index('station_id')
            summary_dfs.append(df)
    
    # Validate that data was loaded
    if not timeseries_dfs:
        raise ValueError(
            f"No timeseries data found in {data_dir}\n"
            f"Expected CSV files with 'year' column. "
            f"Please verify the data directory contains CAMELS-AUS data."
        )
    
    if not summary_dfs:
        raise ValueError(
            f"No summary data found in {data_dir}\n"
            f"Expected CSV files with station metadata."
        )
    
    # Combine all timeseries data and convert date columns
    timeseries_data = pd.concat(timeseries_dfs, axis=0, ignore_index=True)
    timeseries_data['date'] = pd.to_datetime(timeseries_data[['year', 'month', 'day']])
    timeseries_data = timeseries_data.drop(['year', 'month', 'day'], axis=1)
    
    # Combine all summary data
    summary_data = pd.concat(summary_dfs, axis=1)

    return timeseries_data, summary_data


def plot_catchments(camels_data, data_dir):
    """Plot catchment locations across Australia.
    
    Args:
        camels_data: PrepareData object containing summary data
        data_dir: Path to data directory containing shapefiles
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    # Get unique columns from summary data
    summary_data = camels_data.summary_data.loc[:, ~camels_data.summary_data.columns.duplicated()]

    # Extract station location information
    cities = summary_data['station_name']
    lats = summary_data['lat_outlet']
    longs = summary_data['long_outlet']
    states = summary_data['state_outlet']
    
    # Generate random priority values (placeholder for future use)
    priority = np.random.randint(1, 6, size=len(cities))
    
    # Create DataFrame with station data
    df = pd.DataFrame({
        'cityname': cities,
        'lats': lats,
        'longs': longs,
        'States': states,
        'priority': priority
    })

    state_mapping = {'QLD': 1, 'NSW': 2, 'SA': 3, 'VIC': 4, 'ACT': 5, 'WA': 6, 'NT': 7, 'TAS': 8}
    df['state_num'] = df['States'].map(state_mapping)

    # Load Australia boundary shapefile
    shape_file = os.path.join(data_dir, '02_location_boundary_area/shp/bonus data/Australia_boundaries.shp')
    australia = gpd.read_file(shape_file)
    
    # Set coordinate reference system (CRS) - GDA2020
    australia.crs = 'epsg:7844'
    
    # Create GeoDataFrame from station coordinates
    gdf_cities = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longs, df.lats))
    
    # Set CRS to GDA2020 (https://epsg.io/7844)
    gdf_cities.crs = 'epsg:7844'
    
    # Reproject to match shapefile CRS
    gdf_cities = gdf_cities.to_crs(australia.crs)
    
    # Spatial join to link stations to their state polygons
    gdf_cities = gpd.sjoin(gdf_cities, australia, predicate='within')
    
    # Set up the plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create color palette for states
    custom_palette = sns.color_palette(
        ['darkblue', 'black', 'purple', 'darkred', 'darkgreen', 'darkorange', 'brown', 'blue'],
        n_colors=len(df['state_num'].unique())
    )
    
    # Plot stations colored by state
    sns.scatterplot(
        ax=ax, data=gdf_cities, x='longs', y='lats', hue='States',
        s=15, palette=custom_palette, edgecolor='black',
        alpha=0.8, legend='full', zorder=2
    )
    
    # Add Australia boundary as background
    australia.plot(ax=ax, color='lightgrey', edgecolor='white', zorder=1)
    
    # Set plot limits and labels
    ax.set_xlim(110, 160)
    ax.set_title('Catchments across Australia')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return fig






class PrepareData:
    """Data preparation and feature engineering for CAMELS-AUS streamflow data.
    
    This class handles data cleaning, feature engineering, and dataset preparation
    for training and evaluation of streamflow models.
    """
    
    def __init__(self, timeseries_data, summary_data):
        """Initialize PrepareData with timeseries and summary data.
        
        Args:
            timeseries_data: DataFrame containing timeseries data for all stations
            summary_data: DataFrame containing summary/static attributes for stations
        """
        # Clean missing value indicators
        self.timeseries_data = timeseries_data.replace(MISSING_VALUE_INDICATOR, np.nan)
        
        # Feature Engineering: Calculate precipitation deficit
        actual_evap_data = self._get_source_data('et_morton_actual_SILO')
        precipitation_data = self._get_source_data('precipitation_AWAP')
         
        # Align dates between precipitation and evapotranspiration
        actual_evap_data = actual_evap_data[actual_evap_data['date'].isin(precipitation_data['date'])].reset_index(drop=True)
        precipitation_data = precipitation_data[precipitation_data['date'].isin(actual_evap_data['date'])].reset_index(drop=True)
        
        # Calculate precipitation deficit (P - ET)
        self.precipitation_deficit = precipitation_data.drop(['date'], axis=1).subtract(actual_evap_data.drop(['date'], axis=1))
        self.precipitation_deficit['source'] = 'precipitation_deficit'
        self.precipitation_deficit['date'] = precipitation_data['date']
        
        # Extract streamflow data
        self.streamflow_data = self._get_source_data('streamflow_MLd_inclInfilled')
        self.streamflow_data = self.streamflow_data.set_index('date')
        
        # Calculate flood probabilities using empirical CDF
        self.flood_probabilities = self.streamflow_data.apply(self._calculate_flood_extent, axis=0)
        self.flood_probabilities['source'] = 'flood_probabilities'
        self.flood_probabilities['date'] = self.streamflow_data.index

        # Calculate flow acceleration (rate of change)
        self.flow_acceleration = np.abs(self.streamflow_data - self.streamflow_data.shift(1))
        self.flood_prob_acc = self.flow_acceleration.apply(self._calculate_flood_extent, axis=0)
        self.flood_prob_acc['source'] = 'flood_prob_acc'
        self.flood_prob_acc['date'] = self.streamflow_data.index
        
        # Create binary flood indicator (1 if flood probability < 5%)
        self.flood_indicator = self.flood_probabilities.map(
            lambda x: int(x < FLOOD_PROBABILITY_THRESHOLD) if pd.notna(x) and isinstance(x, float) else x
        )
        self.flood_indicator['source'] = 'flood_indicator'
        self.flood_indicator['date'] = self.flood_probabilities['date']        
        
        # Create temporal features using sin/cos transformation for cyclical encoding
        date_min = self.flood_probabilities['date'].min()
        year_sin = self.flood_probabilities['date'].apply(
            lambda x: np.sin((x - date_min).total_seconds() * (2 * np.pi / YEAR_SECONDS))
        )
        year_cos = self.flood_probabilities['date'].apply(
            lambda x: np.cos((x - date_min).total_seconds() * (2 * np.pi / YEAR_SECONDS))
        )
        all_stations = list(self.flood_probabilities.drop(columns=['source', 'date'], axis=1).columns)
        
        # Create sin feature for all stations
        df_sin = pd.DataFrame([{k: value for k in all_stations} for value in year_sin])
        df_sin['source'] = 'year_sin'
        df_sin['date'] = self.flood_probabilities['date']
 
        # Create cos feature for all stations
        df_cos = pd.DataFrame([{k: value for k in all_stations} for value in year_cos])
        df_cos['source'] = 'year_cos'
        df_cos['date'] = self.flood_probabilities['date']
            
        # Combine all engineered features with original timeseries data
        self.timeseries_data = pd.concat([
            self.timeseries_data,
            self.precipitation_deficit,
            self.flood_probabilities,
            self.flood_prob_acc,
            df_sin,
            df_cos,
            self.flood_indicator
        ], axis=0).reset_index(drop=True)
        
        self.summary_data = summary_data
    
    def _get_source_data(self, source_name):
        """Extract data for a specific source and remove the source column."""
        return self.timeseries_data[self.timeseries_data['source'] == source_name].drop(['source'], axis=1)
    
    def get_timeseries_data(self, source, stations):
        """Get timeseries data for specified sources and stations.
        
        Args:
            source: List of data sources to include
            stations: List of station IDs to filter
            
        Returns:
            DataFrame with pivoted timeseries data, or None if data quality is insufficient
        """
        # Filter by source
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(source)]

        # Filter by date range
        self.data_filtered = self.data_filtered.loc[
            (self.data_filtered['date'] >= DATA_START_DATE) &
            (self.data_filtered['date'] < DATA_END_DATE)
        ]
        
        # Pivot data by station and source
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(
            index='date', columns='source', values=stations
        )

        # Check if too much data is missing
        if self.data_filtered.isnull().any(axis=1).sum() > MAX_MISSING_DATA_THRESHOLD * self.data_filtered.shape[0]:
            print(f"Station {stations} has more than {MAX_MISSING_DATA_THRESHOLD*100:.0f}% missing data. Skipping...")
            return None

        # Fill missing values using previous year's data, then interpolate
        for col in self.data_filtered.columns:
            missing_count = self.data_filtered[col].isna().sum()
            print(f"Column {col[1]} for station {stations} has {missing_count} missing values. Filling with previous year data...")
            
            # Find null values after 1980
            null_data = self.data_filtered[col][
                (self.data_filtered[col].isna()) &
                (self.data_filtered.index > dt.datetime(1980, 12, 31))
            ]

            # Identify indices where previous year data exists
            year_offset = pd.offsets.Day(DAYS_IN_YEAR)
            missing_index = [ix for ix in null_data.index - year_offset if ix not in self.data_filtered.index]
            null_indices_to_fill = null_data.index[(null_data.index - year_offset).isin(self.data_filtered.index)]

            print(f"Missing index: {len(missing_index)}/{len(null_data.index)}")

            # Fill with previous year's values
            self.data_filtered.loc[null_indices_to_fill, col] = self.data_filtered.loc[
                null_indices_to_fill - year_offset, col
            ].values

            remaining_missing = self.data_filtered[col].isna().sum()
            print(f"Column {col[1]} has {remaining_missing} missing values after filling with previous year data.")

        # Interpolate remaining missing values
        self.data_filtered.interpolate(method='linear', inplace=True)
        print(f"Interpolated remaining missing values for station {stations}.")
        print(self.data_filtered.info())
        
        return self.data_filtered
        
    def get_data(self, source, stations):
        """Get combined timeseries and summary data for specified sources and stations.
        
        Args:
            source: List of data sources (both timeseries and summary)
            stations: List of station IDs
            
        Returns:
            DataFrame with combined timeseries and summary data
        """
        # Separate timeseries and summary sources
        summary_source = [s for s in source if s in self.summary_data.columns]
        timeseries_source = [s for s in source if s not in self.summary_data.columns]
     
        # Filter and pivot timeseries data
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(timeseries_source)]

        # Pivot data by station and source
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(
            index='date', columns='source', values=stations
        )
        
        # Remove rows with any missing values
        self.data_filtered = self.data_filtered[~self.data_filtered.isnull().any(axis=1)]
        
        # Add summary data as static features
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                self.data_filtered[station, variable] = value
        
        return self.data_filtered.sort_index(axis=1)

    def get_train_val_test(self, source, stations,
                          scaled=True, target=['streamflow_MLd_inclInfilled'],
                          start=None, end=None,
                          discard=0.05, train=0.6, test=0.4):
        """Prepare training and test datasets with optional scaling.
        
        Args:
            source: List of data sources to include
            stations: List of station IDs
            scaled: Whether to apply MinMax scaling
            target: Target variable(s) for prediction
            start: Start date for data filtering
            end: End date for data filtering
            discard: Fraction of data to discard from the beginning (warmup period)
            train: Fraction of data to use for training
            test: Fraction of data to use for testing
            
        Returns:
            tuple: (train_df, test_df) or None if data quality is insufficient
        """
        assert 0 <= discard <= 1, "Discard fraction must be between 0 and 1"
        assert train + test == 1, "Train and test fractions must sum to 1"
     
        # Separate timeseries and summary sources
        summary_source = [s for s in source if s in self.summary_data.columns]
        timeseries_source = [s for s in source if s not in self.summary_data.columns]
        
        # Get timeseries data
        all_data = self.get_timeseries_data(timeseries_source, stations)
        if all_data is None:
            return None

        # Filter by date range if specified
        all_data = all_data.loc[start:end]
        n_rows_all = len(all_data)
        
        # Discard warmup period
        all_data_discarded = all_data.iloc[int(n_rows_all * discard):]
        n_rows_discarded = len(all_data_discarded)
        
        # Split into train and test sets
        train_df = all_data_discarded[:int(n_rows_discarded * train)]
        test_df = all_data_discarded[-int(n_rows_discarded * test):]
        
        # Apply scaling if requested
        if scaled:
            scaler = MinMaxScaler()
            scaler.fit(train_df)
            
            train_df = pd.DataFrame(
                scaler.transform(train_df),
                index=train_df.index,
                columns=train_df.columns
            )
            test_df = pd.DataFrame(
                scaler.transform(test_df),
                index=test_df.index,
                columns=test_df.columns
            )

            self.scaler = scaler
     
        # Add summary data as static features
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                train_df[station, variable] = value
                test_df[station, variable] = value
                                  
        return train_df.sort_index(axis=1), test_df.sort_index(axis=1) 

    def _calculate_flood_extent(self, streamflow_ts):
        """Calculate empirical flood probability using CDF.
        
        Computes the empirical cumulative distribution function (CDF) for
        streamflow values to estimate flood probabilities. Higher flows
        have lower probability values (rare events).
        
        Args:
            streamflow_ts: Series of streamflow values for a single station
            
        Returns:
            Series of flood probabilities corresponding to each streamflow value
        """
        station_name = streamflow_ts.name
    
        flow_data = pd.DataFrame(streamflow_ts)
        flow_data = flow_data.sort_values(by=station_name, ascending=True)
    
        # Assign ranking indices to non-null values
        non_null_mask = ~flow_data[station_name].isnull()
        flow_data.loc[non_null_mask, 'idx'] = np.arange(non_null_mask.sum())
        
        # Calculate empirical probability (Weibull plotting position)
        flow_data.loc[:, 'prob'] = (flow_data['idx'] + 1) / (1 + len(flow_data))
        
        # Extract probability column and restore original order
        flood_prob = flow_data['prob']
        flood_prob.name = station_name
        flood_prob = flood_prob.sort_index()
    
        return flood_prob.reset_index(drop=True)


if __name__ == '__main__':
    # Example usage
    data_dir = '/srv/scratch/z5370003/data/camels-dropbox/'
    
    # Read timeseries and summary data
    timeseries_data, summary_data = read_data_from_file(data_dir)

    # Prepare dataset with feature engineering
    camels_data = PrepareData(timeseries_data, summary_data)

    # Visualize catchment locations
    fig = plot_catchments(camels_data, data_dir)
    plt.show()


