"""
This module contains the class model manager.
"""

import os
from scipy.stats import shapiro
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from crime_prediction_project.utils import residual_scatter_plot, residual_hist_plot


class ModelManager:
    """
    Class to manage a model
    """

    def __init__(self, state_manager):
        self.state_manager = state_manager

    def get_pipeline(
        self,
        model_type,
        model_params,
        use_scaler=True,
        use_pca=False,
        pca_components=None,
    ):
        """
        Creates a pipeline with preprocessing steps (e.g., scaling, PCA) and the selected model.

        Parameters:
            model_type (str): The type of model (e.g., 'Ridge', 'RandomForest', 'XGBoost').
            model_params (dict): Hyperparameters for the selected model.
            use_scaler (bool): Whether to include a scaler in the pipeline.
            use_pca (bool): Whether to include PCA for dimensionality reduction.
            pca_components (int): Number of PCA components (optional).

        Returns:
            Pipeline: The constructed pipeline.
        """
        steps = []

        # Add scaler if required
        if use_scaler:
            steps.append(("scaler", StandardScaler()))

        # Add PCA if required
        if use_pca:
            if not pca_components:
                raise ValueError(
                    "You must specify the number of PCA components when using PCA."
                )
            steps.append(("pca", PCA(n_components=pca_components)))

        supported_models = [
            "LinearRegression",
            "Ridge",
            "XGBoost",
            "RandomForest",
            "Lasso",
        ]

        if model_type not in supported_models:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported models:{supported_models}"
            )

        # Filter model parameters based on the selected model type
        filtered_params = self.filter_model_params(model_type, model_params)

        self.state_manager.save_model_artifacts(
            model_type=model_type, obj=filtered_params, artifact_type="default_params"
        )

        # Dynamically create the model instance
        model = ""
        if model_type == "Ridge":
            model = Ridge(**(filtered_params or {}))
        if model_type == "RandomForest":
            model = RandomForestRegressor(**(filtered_params or {}))
        if model_type == "XGBoost":
            model = XGBRegressor(**(filtered_params or {}))
        if model_type == "LinearRegression":
            model = LinearRegression(**(filtered_params or {}))
        if model_type == "Lasso":
            model = Lasso(**(filtered_params or {}))

        # Add the model to the pipeline
        steps.append(("model", model))

        return Pipeline(steps)

    def filter_model_params(self, model_type, model_params):
        """
        Filters `model_params` to include only the parameters relevant for the selected model.

        Parameters:
            model_type (str): The type of model being used.
            model_params (dict): All available hyperparameters.

        Returns:
            dict: Filtered parameters that match the model's expected parameters.
        """
        # Define valid parameters for each model
        valid_params = {
            "Ridge": {"alpha", "random_state"},
            "Lasso": {"alpha", "max_iter", "tol", "fit_intercept", "selection"},
            "LinearRegression": {"fit_intercept", "n_jobs", "positive"},
            "RandomForest": {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "random_state",
            },
            "XGBoost": {
                "learning_rate",
                "max_depth",
                "n_estimators",
                "colsample_bytree",
                "random_state",
            },
        }

        # Filter parameters dynamically
        if model_type not in valid_params:
            raise ValueError(f"Model type '{model_type}' is not recognized.")
        return {
            key: value
            for key, value in model_params.items()
            if key in valid_params[model_type]
        }

    def cross_validate_model(self, features, target, pipeline, random_state, kfolds=5):
        """
        Performs cross-validation on the pipeline and returns average scores.

        Parameters:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            pipeline (Pipeline): The pipeline to cross-validate.
            cv (int): Number of cross-validation folds.

        Returns:
            dict: Cross-validation scores (mean and standard deviation).
        """
        # Perform cross-validation using negative mean squared error
        evals = {"mse": [], "rmse": [], "r2score": []}

        kfold = KFold(n_splits=kfolds, random_state=random_state, shuffle=True)
        for train_index, test_index in kfold.split(features):
            x_tt, y_tt = features.iloc[train_index], target.iloc[train_index]
            x_ho, y_ho = features.iloc[test_index], target.iloc[test_index]

            # Fit the pipeline
            pipeline.fit(x_tt, y_tt)
            predictions = pipeline.predict(x_ho)

            # Compute metrics
            mse = mean_squared_error(y_ho, predictions)
            evals["mse"].append(mse)

            rmse = np.sqrt(mean_squared_error(y_ho, predictions))
            evals["rmse"].append(rmse)

            r2 = r2_score(y_ho, predictions)
            evals["r2score"].append(r2)

        # Log the results
        cross_val_metrics = {
            "MSE": {
                "Mean MSE": np.mean(evals["mse"]),
                "Std MSE": np.std(evals["mse"]),
            },
            "RMSE": {
                "Mean RMSE": np.mean(evals["rmse"]),
                "Std MSE": np.std(evals["rmse"]),
            },
            "r2Score": {
                "Mean r2Score": np.mean(evals["r2score"]),
                "Std r2Score": np.std(evals["r2score"]),
            },
        }
        self.state_manager.update_state(
            "cv_scores",
            cross_val_metrics,
        )

        return cross_val_metrics

    def train_and_evaluate(self, x_train, x_test, y_train, y_test, pipeline):
        """
        Trains the pipeline and evaluates its performance.

        Parameters:
            x_train (pd.DataFrame): Training features.
            x_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training target.
            y_test (pd.Series): Test target.
            pipeline (Pipeline): The pipeline object.

        Returns:
            dict: Evaluation metrics (e.g., MSE).
        """
        # Train the pipeline
        pipeline.fit(x_train, y_train)
        self.state_manager.update_state("trained_model", pipeline)

        # Make predictions on the training and test sets
        train_pred = pipeline.predict(x_train)
        test_pred = pipeline.predict(x_test)

        # Find the mse on the training set
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = root_mean_squared_error(y_train, train_pred)
        train_r2score = r2_score(y_train, train_pred)

        # Find the mse on the test set
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = root_mean_squared_error(y_test, test_pred)
        test_r2score = r2_score(y_test, test_pred)

        # y_pred = model.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "train": [train_mse, train_rmse, train_r2score],
            "test": [test_mse, test_rmse, test_r2score],
        }
        predictions = {"train": train_pred, "test": test_pred}
        return (metrics, predictions)

    def fine_tune_model(
        self,
        model_type,
        estimator_model,
        model_params_grid,
        x_train,
        y_train,
        scoring="r2",  # Default scoring method is R²
        use_random_search=False,
        n_iter=10,
    ):
        """
        Fine-tunes a model's hyperparameters using GridSearchCV or RandomizedSearchCV.

        Parameters:
            model_type (str): The type of model to fine-tune (e.g., 'Ridge', 'RandomForest', 'XGBoost').
            model_params_grid (dict): A dictionary of hyperparameters and their possible values.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            scoring (str): Scoring method for evaluation (default is 'r2').
            use_random_search (bool): Whether to use RandomizedSearchCV instead of GridSearchCV.
            n_iter (int): Number of iterations for RandomizedSearchCV (ignored for GridSearchCV).

        Returns:
            best_model: The best model found after fine-tuning.
            best_params: The best combination of hyperparameters.
        """
        # Define base models
        base_models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
        }

        # Ensure the model type is supported
        if model_type not in base_models:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize the base model
        # model = base_models[model_type]

        # Choose search method: GridSearchCV or RandomizedSearchCV
        if use_random_search:
            search = RandomizedSearchCV(
                estimator=estimator_model,
                param_distributions=model_params_grid,
                n_iter=n_iter,
                scoring=scoring,  # Dynamic scoring
                cv=5,
                random_state=42,
                verbose=1,
            )
        else:
            search = GridSearchCV(
                estimator=estimator_model,
                param_grid=model_params_grid,
                scoring=scoring,  # Dynamic scoring
                cv=5,
                verbose=1,
            )

        # Perform the search
        search.fit(x_train, y_train)

        # Return the best model and parameters
        return search.best_estimator_, search.best_params_

    def perform_residual_analysis(
        self, y_train, y_test, train_pred, test_pred, model_type
    ):
        """
        Performs residual analysis and saves plots using StateManager.

        Parameters:
            model: Trained model or pipeline.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target values.
            model_type (str): The model type (used for naming saved plots).

        Returns:
            dict: Residual analysis results including saved plot paths.
        """
        # Get the directory from StateManager
        model_folder = os.path.join(
            self.state_manager.get_directory("residual_analysis"), model_type
        )
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Generate predictions and calculate residuals
        # y_pred = model.predict(x_test)
        # residuals = y_test - y_pred

        residuals_train = y_train - train_pred
        residuals_test = y_test - test_pred

        # Define file paths for plots
        residual_hist_file = os.path.join(
            model_folder, f"{model_type}_residual_hist.png"
        )
        residual_scatter_file = os.path.join(
            model_folder, f"{model_type}_residual_scatter.png"
        )

        # Residual plots
        residual_scatter_plot(
            model_type=model_type,
            test_pred=test_pred,
            residuals_test=residuals_test,
            train_pred=train_pred,
            residuals_train=residuals_train,
            filename=residual_scatter_file,
        )

        residual_hist_plot(
            model_type=model_type,
            residuals_test=residuals_test,
            residuals_train=residuals_train,
            filename=residual_hist_file,
        )
        # Perform normality test (Shapiro-Wilk)
        stat_test, p_test = shapiro(residuals_test)
        normality_result_test = (
            "Residuals are normally distributed."
            if p_test > 0.05
            else "Residuals are not normally distributed."
        )

        # Perform normality test (Shapiro-Wilk)
        stat_train, p_train = shapiro(residuals_train)
        normality_result_train = (
            "Residuals are normally distributed."
            if p_train > 0.05
            else "Residuals are not normally distributed."
        )

        # Return residual analysis details
        return {
            "residuals": {"train": residuals_train, "test": residuals_test},
            "shapiro_test": {
                "statistic": {"train": stat_train, "test": stat_test},
                "p_value": {"train": p_train, "test": p_test},
                "result": {
                    "train": normality_result_train,
                    "test": normality_result_test,
                },
            },
            "saved_plots": {
                "histogram": residual_hist_file,
                "scatter_plot": residual_scatter_file,
            },
        }
