#!/usr/bin/python3

"""
This is a utility module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from prettytable import PrettyTable, TableStyle


def fill_missing_with_linear_regression(group):
    """
    For a single county (group), fit a simple linear regression model
    Year vs. each numeric column. Use that model to fill missing values.
    """
    # Sort by Year for clarity
    group = group.sort_index(level="Year")

    # Iterate over each column
    for col in group.columns:
        # Only process numeric columns
        if pd.api.types.is_numeric_dtype(group[col]):
            # Extract the known data points (drop missing)
            valid_data = group[col].dropna()

            # If there aren't at least two valid points, we can't fit a regression
            if len(valid_data) < 2:
                continue

            # Prepare X (Year) and y (column values)
            X = valid_data.index.get_level_values("Year").values.reshape(-1, 1)
            y = valid_data.values

            # Fit the linear regression model
            model = LinearRegression().fit(X, y)

            # Predict for all years in this county
            X_all = group.index.get_level_values("Year").values.reshape(-1, 1)
            y_pred = model.predict(X_all)

            # Fill only missing values with the predictions
            missing_mask = group[col].isna()
            group.loc[missing_mask, col] = y_pred[missing_mask]

    return group


def detect_imputation_strategy(dataframe, threshold=0.5):
    """
    Categorize columns for imputation based on their data type and missing values.

    Args:
        dataframe (pd.DataFrame): Dataset to analyze.
        threshold (float, optional): Skewness threshold for distinguishing
                                     mean vs. median imputation (default: 0.5).

    Returns:
        dict: Contains column categorization:
            - 'mean': Columns suitable for mean imputation.
            - 'median': Columns suitable for median imputation.
            - 'untouched': Columns left unchanged, including string columns and
                           numeric columns without missing values.

    Example:
        >>> df = pd.DataFrame({
        ...     "feature1": [1, 2, 3, np.nan],
        ...     "feature2": [1, 100, 2, np.nan],
        ...     "feature3": [5, 5, 5, 5],
        ...     "feature4": ["a", "b", "c", "d"]
        ... })
        >>> strategies = detect_imputation_strategy(df)
        >>> print(strategies)

    Notes:
        - Skewness is computed for numeric columns with missing values.
        - String columns are always left unchanged.
    """
    mean_columns = []
    median_columns = []
    untouched_columns = []

    for column in dataframe.columns:
        if dataframe[column].dtype in [np.float64, np.int64]:  # Numeric columns
            if dataframe[column].isnull().any():
                col_skewness = skew(dataframe[column].dropna())
                if abs(col_skewness) <= threshold:
                    mean_columns.append(column)
                else:
                    median_columns.append(column)
            else:
                untouched_columns.append(column)
        elif dataframe[column].dtype == object:  # String columns
            untouched_columns.append(column)

    return {
        "mean": mean_columns,
        "median": median_columns,
        "untouched": untouched_columns,
    }


def impute_data(dataframe, column_groups):
    """
    Apply preprocessing to the given DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_groups (dict): A dictionary where the key is the strategy (e.g., 'mean')
                              and the value is a list of columns for that strategy.

    Returns:
        pd.DataFrame: Imputed DataFrame with untouched columns added back.
    """
    untouched_columns = column_groups.pop("untouched", [])

    # Create transformers for each group
    def create_transformer(strategy, columns):
        transformer = Pipeline([("imputer", SimpleImputer(strategy=strategy))])
        return (strategy, transformer, columns)

    transformers = [
        create_transformer(strategy, columns)
        for strategy, columns in column_groups.items()
    ]

    # Combine transformations into a ColumnTransformer
    impute_preprocessor = ColumnTransformer(transformers)

    # Apply transformation
    imputed_data = impute_preprocessor.fit_transform(dataframe)

    # Convert back to a DataFrame
    imputed_df = pd.DataFrame(
        imputed_data,
        columns=[col for cols in column_groups.values() for col in cols],
    )

    # Add untouched columns back
    return pd.concat(
        [dataframe[untouched_columns].reset_index(drop=True), imputed_df], axis=1
    )


def display_model_metrics(models_stats, region):
    """
    Generates and displays a PrettyTable for model performance metrics.

    Args:
        models_stats (dict): A dictionary containing performance statistics for models.
                             Each key is the model name, and the value is another dictionary
                             with keys 'train' and 'test', each containing a list of metrics
                             [MSE, RMSE, r2 Score].

    Example:
        models_stats = {
            "Ridge": {
                "train": [10.5, 3.24, 0.85],
                "test": [12.3, 3.5, 0.80]
            },
            "Lasso": {
                "train": [9.8, 3.13, 0.87],
                "test": [11.7, 3.42, 0.82]
            }
        }
        display_model_metrics(models_stats)
    """
    # Create the PrettyTable instance
    table = PrettyTable()
    table.field_names = ["Model", "Type", "MSE", "RMSE", "r2 Score"]

    # Populate the table with rows
    for key, stat in models_stats.items():
        table.add_rows(
            [
                [
                    key,
                    "train data",
                    stat["train"][0],
                    stat["train"][1],
                    stat["train"][2],
                ],
                ["", "test data", stat["test"][0], stat["test"][1], stat["test"][2]],
            ],
            divider=True,
        )

    # Set table title and styling
    table.title = f"Model Performance Metrics for {region}"
    table.set_style(TableStyle.DOUBLE_BORDER)

    # Display the table
    print(table)


def display_cross_validation_metrics(metrics_dict, region):
    """
    Generates and displays a PrettyTable for comparing multiple models' metrics.

    Args:
        metrics_dict (dict): A dictionary containing performance metrics for multiple models.
                             Each key is the model name, and its value is another dictionary
                             with metrics (e.g., MSE, RMSE, r2Score) containing 'Mean' and 'Std' values.

    Example:
        metrics_dict = {
            "LinearRegression": {
                "MSE": {"Mean MSE": 1.363147e-06, "Std MSE": 3.169104e-07},
                "RMSE": {"Mean RMSE": 0.001158, "Std MSE": 0.000145},
                "r2Score": {"Mean r2Score": 0.576404, "Std r2Score": 0.070375},
            },
            "Ridge": {
                "MSE": {"Mean MSE": 1.359505e-06, "Std MSE": 3.105469e-07},
                "RMSE": {"Mean RMSE": 0.001157, "Std MSE": 0.000143},
                "r2Score": {"Mean r2Score": 0.577083, "Std r2Score": 0.070440},
            },
        }
        display_model_comparison(metrics_dict)
    """
    # Create the PrettyTable instance
    table = PrettyTable()
    table.field_names = [
        "Model",
        "Mean MSE",
        "Std MSE",
        "Mean RMSE",
        "Std RMSE",
        "Mean R²",
        "Std R²",
    ]

    # Populate the table with data for each model
    for model_name, metrics in metrics_dict.items():
        table.add_row(
            [
                model_name,
                f"{metrics['MSE']['Mean MSE']:.6e}",
                f"{metrics['MSE']['Std MSE']:.6e}",
                f"{metrics['RMSE']['Mean RMSE']:.6e}",
                f"{metrics['RMSE']['Std MSE']:.6e}",
                f"{metrics['r2Score']['Mean r2Score']:.6f}",
                f"{metrics['r2Score']['Std r2Score']:.6f}",
            ]
        )

    # Set table title and styling
    table.title = f"Cross Validation: Performance Metrics for {region}"
    table.set_style(TableStyle.DOUBLE_BORDER)

    # Display the table
    print(table)


def residual_scatter_plot(
    model_type,
    test_pred,
    residuals_test,
    train_pred,
    residuals_train,
    filename,
):
    """
    Generates a scatter plot showing residuals vs predicted values for both
    the test and train sets. Saves the figure as an image file.

    Parameters:
        model_type (str): Name of the model.
        test_pred (array-like): Predicted values for the test set.
        residuals_test (array-like): Residuals for the test set.
        train_pred (array-like): Predicted values for the train set.
        residuals_train (array-like): Residuals for the train set.
        filename (str): Path to save the plot as an image file.

    Saves:
        A figure with two subplots displaying residuals for test and train sets.
    """

    # Create a figure and two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # First subplot

    axes[0].scatter(test_pred, residuals_test, color="blue", label="Residuals(test)")
    axes[0].axhline(0, color="red", linestyle="--")
    # axes[0].plot([min(y_test), max(y_test)], [min(mlr_test_residuals), max(mlr_test_residuals)])
    axes[0].set_title(f"Residuals vs Predicted for test set ({model_type})")
    axes[0].set_xlabel("Predicted Crime Rate")
    axes[0].set_ylabel("Residuals")
    axes[0].legend()

    # Second subplot
    axes[1].scatter(train_pred, residuals_train, color="blue", label="Residuals(train)")
    axes[1].axhline(0, color="red", linestyle="--")
    # axes[0].plot([min(y_test), max(y_test)], [min(mlr_test_residuals), max(mlr_test_residuals)])
    axes[1].set_title(f"Residuals vs Predicted for train set ({model_type})")
    axes[1].set_xlabel("Predicted Crime Rate")
    axes[1].set_ylabel("Residuals")
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def residual_hist_plot(
    model_type,
    residuals_test,
    residuals_train,
    filename,
):
    """
    Generates histograms of residuals for both the test and train sets.
    Saves the figure as an image file.

    Parameters:
        model_type (str): Name of the model.
        residuals_test (array-like): Residuals for the test set.
        residuals_train (array-like): Residuals for the train set.
        filename (str): Path to save the plot as an image file.

    Saves:
        A figure with two subplots displaying histograms of residuals for test and train sets.
    """

    # Create a figure and two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].hist(residuals_test, bins=30, edgecolor="black")
    axes[0].set_title(f"Residual Histogram for test set ({model_type})")
    axes[0].set_xlabel("Residuals")
    axes[0].set_ylabel("Frequency")

    # Second subplot
    axes[1].hist(residuals_train, bins=30, edgecolor="black")
    axes[1].set_title(f"Residual Histogram for train set ({model_type})")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    # Adjust layout
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
