
# Evaluation of Deep Learning Models for Extreme Floods in Australia

This project assesses deep learning techniques for multi-step ahead streamflow forecasting using hydrological data from 222 Australian catchments. The study enhances predictions for extreme flood events using a novel Quantile LSTM approach that integrates time series and static catchment features.

## Dataset

**CAMELS-AUS**: Australian Catchment Attributes and Meteorology for Large-sample Studies  
- **Source**: [DOI:10.1594/PANGAEA.921850](https://doi.pangaea.de/10.1594/PANGAEA.921850)
- **Catchments**: 222 stations across Australia  
- **Features**: Time series meteorological/hydrological data + static catchment attributes
- **Data Access**: [Dropbox Link](https://www.dropbox.com/home/Evaluation-of-Deep-Learning-Models-for-Extreme-Floods-in-Australia%3A%20Data)

### Data Organization
```
data/
├── 01_id_name_metadata/    # Station IDs and names (ordered by BoM division, then ID)
├── 02_location_boundary_area/  # Geographic information
└── 04_attributes/          # Catchment characteristics (geology, topography, land cover, etc.)
```

## Project Structure

```
src/
├── data.py          # Data loading and preprocessing
├── model.py         # Deep learning model architectures
├── window.py        # Time series windowing utilities
├── flood_risk.py    # Flood threshold and alert classification
└── visualization.py # Plotting functions for analysis
```

## Training Scripts

### Core Training
- **`train_quantile_ensemble_individual.py`** - Train quantile models for individual stations (top N by runoff ratio)
- **`train_quantile_ensemble_batch.py`** - Batch training across all catchments in a state
- **`train_quantile_ensemble_with_floodrisk.py`** - Training with integrated flood risk indicators

### Model Evaluation
- **`evaluate_strategy.py`** - Compare catchment selection strategies (Multi-LSTM)
- **`evaluate_architecture.py`** - Compare model architectures (individual strategy)

### Switching Models
- **`train_switching_model.py`** - Train hybrid models for flood probability and streamflow

**Usage Example:**
```bash
python train_quantile_ensemble_individual.py \
    --state SA \
    --n-stations 5 \
    --num-runs 1 \
    --input-width 5 \
    --output-width 5

# For help on all arguments
python train_quantile_ensemble_individual.py --help
```

## Key Features

### Quantile Regression Models
- **Three quantiles**: q05 (lower bound), q50 (median), q95 (upper bound)
- **Uncertainty quantification**: Probabilistic streamflow predictions
- **Ensemble approach**: Combined predictions across multiple model runs

### Flood Risk Assessment
- **Automated threshold calculation**: Based on historical percentiles (default: 95th)
- **Four-tier classification**: High / Moderate / Low / Unlikely flood risk
- **Visualization**: Multi-panel plots showing predictions, alerts, and statistics
- **Export**: CSV outputs for predictions, alerts, and thresholds

### Model Architectures
- **Quantile LSTM**: Multi-quantile streamflow prediction
- **Mixed Models**: Combine time series and static features
- **Switch Models**: Hybrid flood probability + streamflow prediction
- **Ensemble Static**: Integration of catchment attributes

## Recent Improvements (2024)

### Code Quality
- ✅ Fixed critical bugs in `model.py` (@staticmethod decorators, None checks)
- ✅ Standardized parameter naming (PEP 8 compliant: `conv_width` not `CONV_WIDTH`)
- ✅ Improved NaN handling in aggregation (using `skipna=True`, `np.nanmean()`)
- ✅ Replaced sentinel values (-99) with `np.nan` for missing data

### Modularization
- ✅ Extracted flood risk functions to `src/flood_risk.py` (~170 lines)
- ✅ Extracted visualization functions to `src/visualization.py` (~120 lines)
- ✅ Reduced main training script size by 31% (750+ → 520 lines)
- ✅ Improved code reusability and maintainability

## Notebooks

- **`notebooks/data_processing.ipynb`** - Data exploration and preprocessing
- **`notebooks/generate_flood_risk_indicator.ipynb`** - Flood risk methodology
- **`notebooks/results.ipynb`** - Results analysis and visualization
- **`notebooks/plotting_catchments.ipynb`** - Geographic catchment visualization

## Results

Training outputs are organized by stage in `results/`:
- **Stage 1-2**: Model architecture comparisons
- **Stage 3**: State-level quantile predictions
- **Stage 4-8**: Progressive refinements and flood risk integration
- **Stage_FloodRisk**: Latest flood risk assessment results

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- tqdm

## Citation

If you use this code or methodology, please cite the original CAMELS-AUS dataset:
> Fowler, K. J. A., et al. (2020). CAMELS-AUS: Hydrometeorological time series and landscape attributes for 222 catchments in Australia. PANGAEA. https://doi.org/10.1594/PANGAEA.921850

## License

This project is for research purposes. Please refer to the original CAMELS-AUS dataset license for data usage terms.
