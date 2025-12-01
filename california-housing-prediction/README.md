# California Housing Price Prediction: An End-to-End Machine Learning Pipeline

## Project Overview

This project implements a comprehensive, academically rigorous machine learning pipeline to predict median house values in California districts. The project follows research-grade standards with thorough documentation, extensive visualizations, and rigorous methodology.

### Problem Statement

The goal is to build a predictive model that can accurately estimate the median house value for California districts based on various demographic and geographic features. This is a **regression problem** where we predict continuous values (house prices in hundreds of thousands of dollars).

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

The project follows a rigorous, end-to-end machine learning pipeline:

1. **Data Exploration**: Comprehensive exploratory data analysis (EDA) with visualizations
2. **Data Preprocessing**: Train-test splitting, outlier handling, feature scaling
3. **Feature Engineering**: Creation of interaction features and transformations
4. **Model Selection**: Training and comparison of 7 different algorithms
5. **Hyperparameter Tuning**: Grid search optimization for best models
6. **Model Evaluation**: Rigorous evaluation using multiple metrics (RMSE, MAE, R²)
7. **Visualization**: Extensive plots demonstrating findings and model performance

## Key Features

- **Multiple Algorithms**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, SVR
- **Comprehensive Evaluation**: Cross-validation, train/test metrics, residual analysis
- **Feature Engineering**: 5 new engineered features to improve model performance
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Rich Visualizations**: 8+ publication-quality figures
- **Model Persistence**: Saved models for future predictions

## Installation

1. Navigate to the project directory:
```bash
cd california-housing-prediction
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter "externally-managed-environment" errors, make sure you're using a virtual environment as shown above.

## Usage

### Running the Notebook

1. Navigate to the project directory:
```bash
cd california-housing-prediction
```

2. Activate the virtual environment (if not already active):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

Alternatively, you can use:
```bash
python3 -m jupyter notebook
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

Saved models are stored in the `models/` folder:
- `models/best_housing_model.pkl`: Best trained model
- `models/feature_scaler.pkl`: Feature scaler for preprocessing

### Using the Saved Model

The best model is saved in the `models/` folder. To use it:

```python
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('models/best_housing_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Prepare new data (must have same features as training data)
new_data = pd.DataFrame({
    'MedInc': [8.0],
    'HouseAge': [41.0],
    'AveRooms': [6.0],
    'AveBedrms': [1.0],
    'Population': [322.0],
    'AveOccup': [2.5],
    'Latitude': [37.88],
    'Longitude': [-122.23]
})

# Scale and engineer features (use the same functions from notebook)
# ... apply feature engineering ...

# Predict
prediction = model.predict(new_data)
print(f"Predicted house value: ${prediction[0]:.2f} (in $100,000s)")
```

## Research Questions Addressed

1. **Which features are most predictive of house prices?**
   - Answer: Median income (MedInc) is the strongest predictor, followed by location (Latitude/Longitude) and house characteristics.

2. **How do different machine learning algorithms compare in performance?**
   - Answer: Tree-based ensemble methods (Random Forest, Gradient Boosting) outperform linear models, with tuned models achieving R² > 0.8.

3. **What is the impact of feature engineering on model performance?**
   - Answer: Engineered features (e.g., RoomsPerHousehold, Income_Rooms) contribute to improved model performance.

4. **Can we achieve high prediction accuracy with this dataset?**
   - Answer: Yes, the best model achieves R² > 0.8 with RMSE < 0.5, indicating strong predictive power.

## Key Findings

1. **Best Model**: Tuned Gradient Boosting or Random Forest achieves the lowest RMSE
2. **Feature Importance**: Median Income is the most important predictor
3. **Model Performance**: R² > 0.8 indicates strong predictive power
4. **Hyperparameter Tuning**: Provides significant improvements over baseline models
5. **Feature Engineering**: Contributes meaningfully to model performance

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
4. Implement model deployment pipeline
5. Create interactive prediction tool/web app
6. Perform time series analysis if temporal data becomes available

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

## Reproducibility

The project uses random seeds (42) throughout to ensure reproducibility. All results should be consistent across runs.

## License

This project is for educational and research purposes.

## Author

Created as an academic research project demonstrating end-to-end machine learning pipeline development.

## Acknowledgments

- California Housing Dataset from scikit-learn
- Scikit-learn library for machine learning tools
- Matplotlib and Seaborn for visualizations
