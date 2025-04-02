"""
This module is the entry point of the program
"""

import os
import pandas as pd
from crime_prediction_project.pipeline import CrimeRatePipeline
from crime_prediction_project.utils import (
    display_model_metrics,
    display_cross_validation_metrics,
    fill_missing_with_linear_regression,
)


def load_data(option: str = "1985-2023") -> pd.DataFrame:
    """
    Load a dataset based on the specified time range.

    This function reads an Excel file containing the dataset for the specified time range
    and returns it as a pandas DataFrame. Two options are currently available:
    "1985-2023" or "2010-2023".

    Args:
        option (str, optional): A string indicating the time range for the dataset.
            Defaults to "1985-2023".
            - "1985-2023": Loads data from 'tentative_final_with_NaN.xlsx'.
            - "2010-2023": Loads data from 'tentative_final.xlsx'.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset for the specified time range.

    Raises:
        FileNotFoundError: If the specified Excel file is not found.
        ValueError: If an invalid option is provided.

    Example:
        >>> df = load_data(option="2010-2023")
        >>> print(df.head())
    """
    filepath = ""
    df = pd.DataFrame()
    if option == "1985-2023":
        filepath = os.path.join("dataset", "crime_data_1985-2023.xlsx")
        dataframe = pd.read_excel(filepath)
        dataframe = dataframe.set_index(["County", "Year"])
        df = fill_missing_with_linear_regression(dataframe)
        # strategies = detect_imputation_strategy(da)
        # df = impute_data(df, strategies)
    elif option == "2010-2023":
        filepath = os.path.join("dataset", "crime_data_2010-2023.xlsx")
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Invalid option. Choose either '1985-2023' or '2010-2023'.")
    return df


if __name__ == "__main__":
    PERFORM_CV = True
    FINE_TUNE = True
    PCA_COMPONENTS = 11

    data = load_data(option="1985-2023")
    # Print dataset dimensions and preview
    print(f"\nDataset created with {data.shape[0]} rows and {data.shape[1]} columns.\n")
    # print(f"{data.head()}\n")

    # Initialize pipeline
    pipeline = CrimeRatePipeline(data)

    model_types = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "RandomForest",
        "XGBoost",
    ]  # Models to run

    evaluated_metrics = {}  # To store metrics for each model
    saved_evals = {}
    saved_cv_scores = {}
    for model_type in model_types:
        print("\n---------------------------------------------------")
        print(f"Running pipeline for {model_type}...")
        (evaluations, region) = pipeline.run_pipeline(
            category="Urban",
            model_type=model_type,
            pca_components=PCA_COMPONENTS,
            perform_cv=PERFORM_CV,
            k_folds=5,
            fine_tune=FINE_TUNE,
        )
        evaluated_metrics[model_type] = evaluations

        # Metrics are a
        # load_evals = pipeline.load_model_artifacts(
        #    model_type=model_type, artifact="evaluations"
        # )
        # saved_evals[model_type] = load_evals

        if PERFORM_CV:
            load_cv_scores = pipeline.load_model_artifacts(
                model_type=model_type, artifact="cv_scores"
            )
            saved_cv_scores[model_type] = load_cv_scores

    TITLE = ""
    if region == "all":
        TITLE = "All counties"
    elif region in ["Urban", "Suburban", "Rural"]:
        TITLE = region + " counties"
    else:
        TITLE = region + " county"

    print("\n")
    display_model_metrics(evaluated_metrics, TITLE)
    print("\n\n")
    if PERFORM_CV:
        display_cross_validation_metrics(saved_cv_scores, TITLE)
