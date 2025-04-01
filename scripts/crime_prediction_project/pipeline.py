"""
This module contains the calss CrimeRratePipeline.
"""

from sklearn.model_selection import train_test_split
from .data_handler import DataHandler
from .state_manager import StateManager
from .model_manager import ModelManager
from .config import CONFIG


class CrimeRatePipeline:
    """
    The class represents CrimeRatePipeline
    """

    def __init__(self, data):
        self.config = CONFIG
        self.data_handler = DataHandler(data, self.config)
        self.state_manager = StateManager()
        self.model_manager = ModelManager(self.state_manager)

    def run_pipeline(
        self,
        category,
        model_type,
        use_scaler=None,
        use_pca=None,
        pca_components=None,
        perform_cv=False,
        k_folds=5,
        fine_tune=False,
        **default_model_params,
    ):
        """
        Executes the full pipeline: preprocessing, feature engineering, fine-tuning (optional),
        model training, and evaluation.

        Parameters:
            category (str): Data category (e.g., 'Urban', 'Rural', 'all').
            model_type (str): Type of model (e.g., 'Ridge', 'XGBoost').
            use_scaler (bool): Whether to include a scaler.
            use_pca (bool): Whether to include PCA.
            pca_components (int): Number of PCA components (optional).
            perform_cv (bool): Whether to perform K-Fold cross-validation.
            k_folds (int): Number of folds for K-Fold cross-validation.
            fine_tune (bool): Whether to fine-tune the model before training and evaluation.
            model_params (dict): Additional model hyperparameters.
        """

        # Dynamically determine whether to use scaler or PCA based on model type
        if use_scaler is None:
            use_scaler = model_type in ["LinearRegression", "Ridge", "Lasso"]

        if use_pca is None:
            use_pca = model_type in ["LinearRegression", "Ridge", "Lasso"]

        # Update state
        self.state_manager.update_state("category", category)
        self.state_manager.update_state("model_type", model_type)

        # Preprocess data
        preprocessed_data = self.data_handler.preprocess_data(category)
        # print(preprocessed_data.head())

        # Separate features (x) and target (y)
        x = preprocessed_data[self.config["non_categorical_features"]]
        y = preprocessed_data[self.config["target_column"]]

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.config["random_seed"]
        )

        # Initialize pipeline with optional PCA and scaler
        default_model_params = CONFIG["default_model_params"]
        pipeline = self.model_manager.get_pipeline(
            model_type,
            default_model_params,
            use_scaler,
            use_pca,
            pca_components=pca_components if use_pca else None,
        )

        # Fine-tune the model before training (if enabled)
        if fine_tune:
            print(f"Fine-tuning hyperparameters for {model_type}...")
            best_model, best_params = self.model_manager.fine_tune_model(
                model_type=model_type,
                estimator_model=pipeline,
                model_params_grid=self.config["param_grids"][model_type],
                x_train=x_train,
                y_train=y_train,
                scoring="r2",
                use_random_search=False,  # Default to GridSearchCV
            )

            self.state_manager.save_model_artifacts(
                model_type=model_type, obj=best_params, artifact_type="best_params"
            )
            print(f"Best hyperparameters for {model_type}: {best_params}")

            # Replace pipeline model with the fine-tuned version
            pipeline = best_model
            print(pipeline)

        # Perform K-Fold Cross-Validation (if requested)
        if perform_cv:
            print(f"\nPerforming {k_folds}-Fold Cross-Validation for {model_type}...")
            cv_scores = self.model_manager.cross_validate_model(
                x_train,
                y_train,
                pipeline,
                random_state=self.config["random_seed"],
                kfolds=k_folds,
            )
            # print(f"K-Fold Cross-Validation Scores: {cv_scores}")

            # Save artifacts: Cross-validation scores
            self.state_manager.save_model_artifacts(
                model_type=model_type, obj=cv_scores, artifact_type="cv_scores"
            )

        # Train and evaluate the fine-tuned pipeline
        print(f"\nTraining and evaluating the {model_type} model...")
        metrics, predictions = self.model_manager.train_and_evaluate(
            x_train, x_test, y_train, y_test, pipeline
        )
        # print(f"Performance Metrics for {model_type}: {evaluations}")

        # Save artifacts: Evaluation metrics
        self.state_manager.save_model_artifacts(
            model_type=model_type, obj=metrics, artifact_type="metrics"
        )

        # Save artifacts: Fine-tuned pipeline
        self.state_manager.save_model_artifacts(
            model_type=model_type, obj=pipeline, artifact_type="pipeline"
        )

        # Perform residual analysis and save residual plots
        print(f"\nPerforming residual analysis for {model_type}...")
        residual_analysis = self.model_manager.perform_residual_analysis(
            y_train=y_train,
            y_test=y_test,
            train_pred=predictions["train"],
            test_pred=predictions["test"],
            model_type=model_type,
        )
        print(f"Residual Analysis Results for {model_type}: {residual_analysis}")
        print(f"Saved Plots: {residual_analysis['saved_plots']}")

        return metrics, category

    def load_model_artifacts(self, model_type, artifact):
        """
        Loads saved evaluation metrics.

        Returns:
            dict: Loaded evaluation metrics.
        """

        return self.state_manager.load_model_artifacts(
            model_type=model_type, artifact_type=artifact
        )
