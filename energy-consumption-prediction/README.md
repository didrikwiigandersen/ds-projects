# Building Energy Consumption Prediction

## Project Overview

This project implements a machine learning pipeline to predict the energy consumption of buildings based on various building characteristics and environmental factors. 

### Problem Statement

The goal is to build a predictive model that can accurately estimate the energy consumption of residential buildings based on various building features such as surface area, wall area, roof area, orientation, and glazing characteristics.

### Dataset

We use the **Energy Efficiency Dataset**, which includes:
- **Target Variables**: 
  - Heating Load (kWh/m² per year)
  - Cooling Load (kWh/m² per year)
- **Features**: 8 numerical features including:
  - X1: Relative Compactness
  - X2: Surface Area
  - X3: Wall Area
  - X4: Roof Area
  - X5: Overall Height
  - X6: Orientation (2-5)
  - X7: Glazing Area
  - X8: Glazing Area Distribution (0-5)

## Project Structure

```
energy-consumption-prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── energy_consumption_prediction.ipynb # Main Jupyter notebook with full pipeline
├── images/                            # Generated visualizations and plots
├── models/                            # Saved trained models
└── data/                              # Data files (if any)
```

## Methodology

The project follows the following methodology:

1. **Data Exploration**: Exploratory data analysis (EDA) with visualizations
2. **Data Preprocessing**: Train-test splitting, outlier handling, feature scaling
3. **Feature Engineering**: Creation of interaction features and transformations
4. **Model Selection**: Training and comparison of 7 different algorithms
5. **Hyperparameter Tuning**: Grid search optimization for best models
6. **Model Evaluation**: Evaluation using multiple metrics (RMSE, MAE, R²)
7. **Visualization**: Plots demonstrating findings and model performance

## Installation

1. Navigate to the project directory:
```bash
cd energy-consumption-prediction
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

1. Navigate to the project directory:
```bash
cd energy-consumption-prediction
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

### Generated Outputs

The notebook generates several visualization files in the `images/` folder:
- `images/target_distribution.png`: Distribution of heating and cooling loads
- `images/feature_distributions.png`: Distribution of all features
- `images/correlation_matrix.png`: Feature correlation heatmap
- `images/feature_target_relationships.png`: Scatter plots of top features vs. target
- `images/model_comparison.png`: Comparison of baseline models
- `images/prediction_vs_actual.png`: Prediction accuracy visualization
- `images/residual_analysis.png`: Residual plots for model diagnostics
- `images/feature_importance.png`: Feature importance for tree-based models
- `images/all_models_comparison.png`: Comprehensive model comparison

## Limitations

1. **Dataset Size**: Limited number of samples may affect model generalization
2. **Feature Limitations**: Could benefit from additional features (weather data, occupancy patterns)
3. **Outliers**: Some extreme values may affect model performance
4. **Generalization**: Performance on new buildings may vary

## Technical Details

### Feature Engineering

Created new features:
- `SurfaceToVolumeRatio`: Surface area to volume ratio
- `WallToRoofRatio`: Wall area to roof area ratio
- `GlazingRatio`: Glazing area relative to surface area
- `Compactness_Squared`: Non-linear compactness transformation
- `Height_Glazing`: Interaction between height and glazing area

### Hyperparameter Tuning

Grid search with 5-fold cross-validation on:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Gradient Boosting: n_estimators, learning_rate, max_depth, min_samples_split

## Dependencies

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.3.0
- Jupyter >= 1.0.0
- Joblib >= 1.3.0
