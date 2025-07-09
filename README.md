# Time Series Forecasting with Ensemble Models

This project demonstrates time series forecasting using an ensemble of RandomForest, XGBoost, and a simple Neural Network model. It utilizes a walk-forward validation approach to evaluate performance and employs SHAP values for model interpretability.

## Features

- **Data Fetching:** Downloads historical stock data (AAPL by default) using `yfinance`.
- **Feature Engineering:** Calculates technical indicators like RSI, SMA, and Volatility.
- **Target Variable:** Creates a target variable representing the next day's percentage price change.
- **Walk-Forward Validation:** Splits the data into sequential training and testing sets for realistic evaluation.
- **Ensemble Modeling:** Trains and combines predictions from:
    - RandomForest Regressor
    - XGBoost Regressor
    - Simple Keras Sequential Neural Network
- **Weighted Ensemble:** Combines the predictions from the individual models using predefined weights.
- **Model Evaluation:** Reports the Mean Squared Error (MSE) for each fold and the average MSE across all folds.
- **Model Interpretability:** Uses SHAP (SHapley Additive exPlanations) to explain the predictions of the XGBoost model (SHAP is only supported for tree-based models like RandomForest and XGBoost).

## Prerequisites

- Python 3.7+
- Required libraries (automatically installed by the notebook):
    - `yfinance`
    - `pandas`
    - `ta`
    - `scikit-learn`
    - `xgboost`
    - `tensorflow`
    - `shap`

## How to Run

This code is designed to be run in a Google Colaboratory or Jupyter Notebook environment.

1.  **Open the Notebook:** Upload the notebook file to Google Colab or open it in your local Jupyter environment.
2.  **Run All Cells:** Execute all the cells in the notebook sequentially.

The notebook will perform the following steps:
1. Install necessary libraries.
2. Fetch the stock data.
3. Engineer the features and target variable.
4. Perform walk-forward cross-validation.
5. Train and evaluate each model within the ensemble for each fold.
6. Combine predictions and calculate the ensemble's MSE.
7. Print the results for each fold and the average MSE.
8. Generate a SHAP summary plot for the XGBoost model.

## Code Explanation

- **`fetch_data`:** Downloads historical stock data.
- **Feature Engineering:** Calculates `returns_1d`, `rsi`, `sma_10`, `volatility_10`, and the `target` variable.
- **`walk_forward_split`:** Implements the time series split for walk-forward validation.
- **`build_random_forest`, `build_xgboost`, `build_neural_net`:** Functions to create instances of the individual models.
- **`weighted_ensemble`:** Combines the predictions from the individual models.
- **Main Loop:** Iterates through the walk-forward splits, trains each model, makes predictions, combines them, and calculates the MSE.
- **SHAP Analysis:** Initializes SHAP and calculates SHAP values for the XGBoost model's test predictions to visualize feature importance.

## Customization

- **Ticker Symbol:** Modify the `ticker` parameter in `fetch_data` to analyze a different stock.
- **Date Range:** Adjust the `start` and `end` dates in `fetch_data`.
- **Ensemble Weights:** Change the `weights` tuple in `weighted_ensemble` to experiment with different weighting schemes.
- **Model Hyperparameters:** Modify the hyperparameters of the individual models (`n_estimators`, `max_depth`, `learning_rate`, `epochs`, etc.).
- **Features:** Add or remove technical indicators or other relevant features to the data.

## Results

The output will show the Mean Squared Error for each walk-forward fold and the average MSE across all folds, providing an evaluation of the ensemble model's performance. The SHAP plot will visualize the impact of different features on the XGBoost model's predictions.

## License

[Specify your license here, e.g., MIT License]
