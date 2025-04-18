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
        category,
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
            model_type=model_type,
            obj=filtered_params,
            artifact_type="default_params",
            category=category,
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
            features (pd.DataFrame): Feature matrix.
            target (pd.Series): Target variable.
            pipeline (Pipeline): The pipeline to cross-validate.
            kfolds (int): Number of cross-validation folds.

        Returns:
            dict: Cross-validation scores (mean and standard deviation).
        """
        # Perform cross-validation using negative mean squared error
        evals_train = {"mse": [], "rmse": [], "r2score": []}
        evals_test = {"mse": [], "rmse": [], "r2score": []}

        kfold = KFold(n_splits=kfolds, random_state=random_state, shuffle=True)
        for train_index, test_index in kfold.split(features):
            features_train, features_test = (
                features.iloc[train_index],
                features.iloc[test_index],
            )
            target_train, target_test = (
                target.iloc[train_index],
                target.iloc[test_index],
            )

            # Fit the pipeline (model)
            pipeline.fit(features_train, target_train)

            # Prediction for train set.
            predictions_train = pipeline.predict(features_train)

            # Compute metrics for train set and append
            evals_train["mse"].append(
                mean_squared_error(target_train, predictions_train)
            )
            evals_train["rmse"].append(
                root_mean_squared_error(target_train, predictions_train)
            )
            evals_train["r2score"].append(r2_score(target_train, predictions_train))

            # Prediction for test set.
            predictions_test = pipeline.predict(features_test)

            # Compute metrics for test set and append
            evals_test["mse"].append(mean_squared_error(target_test, predictions_test))
            evals_test["rmse"].append(
                root_mean_squared_error(target_test, predictions_test)
            )
            evals_test["r2score"].append(r2_score(target_test, predictions_test))

        # Compute average and standard deviations of metrics
        avg_std_metrics_train = {
            "MSE": {
                "Avg MSE": np.mean(evals_train["mse"]),
                "Std MSE": np.std(evals_train["mse"]),
            },
            "RMSE": {
                "Avg RMSE": np.mean(evals_train["rmse"]),
                "Std MSE": np.std(evals_train["rmse"]),
            },
            "r2Score": {
                "Avg r2Score": np.mean(evals_train["r2score"]),
                "Std r2Score": np.std(evals_train["r2score"]),
            },
        }

        avg_std_metrics_test = {
            "MSE": {
                "Avg MSE": np.mean(evals_test["mse"]),
                "Std MSE": np.std(evals_test["mse"]),
            },
            "RMSE": {
                "Avg RMSE": np.mean(evals_test["rmse"]),
                "Std MSE": np.std(evals_test["rmse"]),
            },
            "r2Score": {
                "Avg r2Score": np.mean(evals_test["r2score"]),
                "Std r2Score": np.std(evals_test["r2score"]),
            },
        }

        # Log the results
        cross_val_metrics = {
            "train": avg_std_metrics_train,
            "test": avg_std_metrics_test,
        }

        self.state_manager.update_state(
            "cv_scores",
            cross_val_metrics,
        )

        return cross_val_metrics

    def fine_tune_model(
        self,
        model_type,
        estimator_model,
        model_params_grid,
        x_train,
        y_train,
        scoring="r2",
        search_method="grid",  # Options: "grid", "random", "hybrid"
        n_iter=10,
        kfolds=5,
        perform_cross_validation=False,
        random_state=42,
        verbose=0,
    ):
        """
        Fine-tunes a model using Grid Search and Random Search (Hybrid approach).

        Parameters:
            - model_type (str): Model type (e.g., 'Ridge', 'RandomForest').
            - estimator_model (object): Model instance to optimize.
            - model_params_grid (dict): Hyperparameter ranges.
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - scoring (str): Scoring metric.
            - search_method (str): "grid", "random", or "hybrid" (Grid + Random combination).
            - n_iter (int): Iterations for RandomizedSearchCV.
            - kfolds (int): Number of cross-validation folds.
            - perform_cross_validation (bool): Whether to perform cross-validation
            - random_state (int): Random seed.
            - verbose (int)

        Returns:
            - Best tuned model.
            - Best hyperparameters.
            - Cross-validation scores (train/test averages).
        """

        best_params = {}

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

        # Grid Search
        if search_method == "grid":
            search = GridSearchCV(
                estimator=estimator_model,
                param_grid=model_params_grid,
                scoring=scoring,
                cv=kfolds,
                verbose=verbose,
            )
            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

        # Randomized Search
        elif search_method == "random":
            search = RandomizedSearchCV(
                estimator=estimator_model,
                param_distributions=model_params_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=kfolds,
                random_state=random_state,
                verbose=verbose,
            )
            search.fit(x_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

        # Hybrid (Random + Grid)
        elif search_method == "hybrid":
            # First: Randomized Search (Wide search space)
            random_search = RandomizedSearchCV(
                estimator=estimator_model,
                param_distributions=model_params_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=kfolds,
                random_state=random_state,
                verbose=verbose,
            )
            random_search.fit(x_train, y_train)
            best_random_params = random_search.best_params_

            # Narrow down search space for Grid Search based on Random Search results
            refined_grid = {
                key: [best_random_params[key]] for key in best_random_params
            }

            # Second: Grid Search (Precise tuning)
            grid_search = GridSearchCV(
                estimator=estimator_model,
                param_grid=refined_grid,
                scoring=scoring,
                cv=kfolds,
                verbose=verbose,
            )
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        else:
            raise ValueError(f"Unsupported search method: {search_method}")

        # Perform Final Cross-Validation when enabled.
        cross_val_scores = None
        if perform_cross_validation:
            print(
                f"Now performing {kfolds}-Fold Cross-Validation for {model_type}...\n"
            )
            cross_val_scores = self.cross_validate_model(
                features=x_train,
                target=y_train,
                pipeline=best_model,
                random_state=random_state,
                kfolds=kfolds,
            )

        return best_model, best_params, cross_val_scores

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

    def perform_residual_analysis(
        self,
        y_train,
        y_test,
        train_pred,
        test_pred,
        model_type,
        category,
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
            model_folder, f"{model_type}_residual_hist_{category}.png"
        )
        residual_scatter_file = os.path.join(
            model_folder, f"{model_type}_residual_scatter_{category}.png"
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
