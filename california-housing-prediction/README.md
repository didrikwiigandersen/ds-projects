# California Housing Price Prediction

## Project Overview

This project implements a machine learning pipeline to predict median house values in California districts. 

### Problem Statement

The goal is to build a predictive model that can accurately estimate the median house value for California districts based on various demographic and geographic features. 

### Dataset

We use the **California Housing Dataset** from the 1990 California census, which includes:
- **Target Variable**: Median house value (in $100,000s)
- **Features**: 8 numerical features including:
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude

## Project Structure

```
california-housing-prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── housing_price_prediction.ipynb     # Main Jupyter notebook with full pipeline
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
cd california-housing-prediction
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  #
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

1. Navigate to the project directory:
```bash
cd california-housing-prediction
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```


4. Open `housing_price_prediction.ipynb` from the Jupyter interface

3. Run all cells sequentially (Cell → Run All)

The notebook will:
- Load and explore the data
- Preprocess and engineer features
- Train multiple models
- Perform hyperparameter tuning
- Generate comprehensive visualizations
- Save the best model

### Generated Outputs

The notebook generates several visualization files in the `images/` folder:
- `images/target_distribution.png`: Distribution of house values
- `images/feature_distributions.png`: Distribution of all features
- `images/correlation_matrix.png`: Feature correlation heatmap
- `images/feature_target_relationships.png`: Scatter plots of top features vs. target
- `images/model_comparison.png`: Comparison of baseline models
- `images/prediction_vs_actual.png`: Prediction accuracy visualization
- `images/residual_analysis.png`: Residual plots for model diagnostics
- `images/feature_importance.png`: Feature importance for tree-based models
- `images/all_models_comparison.png`: Comprehensive model comparison

## Evaluation Metrics

The project uses multiple evaluation metrics:
- **RMSE (Root Mean Squared Error)**: Primary metric for regression
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Proportion of variance explained
- **Cross-Validation**: 5-fold CV for robust performance estimation

## Limitations

1. **Data Age**: Dataset from 1990 may not reflect current market conditions
2. **Feature Limitations**: Could benefit from additional features (crime rates, school quality)
3. **Outliers**: Some extreme values may affect model performance
4. **Generalization**: Performance on new data may vary

## Future Work

1. Collect more recent data
2. Include additional features (crime rates, school ratings, proximity to amenities)
3. Experiment with deep learning models (neural networks)

## Technical Details

### Models Evaluated

1. **Linear Regression**: Baseline linear model
2. **Ridge Regression**: L2 regularization
3. **Lasso Regression**: L1 regularization with feature selection
4. **Elastic Net**: Combination of L1 and L2 regularization
5. **Random Forest**: Ensemble of decision trees
6. **Gradient Boosting**: Sequential ensemble method
7. **Support Vector Regression**: Non-linear regression with RBF kernel

### Feature Engineering

Created 5 new features:
- `RoomsPerHousehold`: Average rooms per household
- `BedroomsPerRoom`: Ratio of bedrooms to rooms
- `PopulationPerHousehold`: Population density metric
- `MedInc_Squared`: Non-linear income transformation
- `Income_Rooms`: Interaction between income and rooms

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
