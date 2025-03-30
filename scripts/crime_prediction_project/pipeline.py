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
        use_scaler=True,
        use_pca=False,
        pca_components=None,
        perform_cv=False,
        k_folds=5,
        **model_params,
    ):
        """
        Executes the full pipeline: preprocessing, feature engineering, model training, and evaluation.

        Parameters:
            category (str): Data category (e.g., 'Urban', 'Rural', 'all').
            model_type (str): Type of model (e.g., 'Ridge', 'XGBoost').
            use_scaler (bool): Whether to include a scaler.
            use_pca (bool): Whether to include PCA.
            pca_components (int): Number of PCA components (optional).
            perform_cv (bool): Whether to perform K-Fold cross-validation.
            k_folds (int): Number of folds for K-Fold cross-validation.
            model_params (dict): Additional model hyperparameters.
        """
        # Update state
        self.state_manager.update_state("category", category)
        self.state_manager.update_state("model_type", model_type)
        self.state_manager.update_state("model_params", model_params)

        # Preprocess data
        preprocessed_data = self.data_handler.preprocess_data(category)
        # print(preprocessed_data.columns)

        # Separate features (x) and target (y)
        x = preprocessed_data[self.config["non_categorical_features"]]
        y = preprocessed_data[self.config["target_column"]]

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.config["random_seed"]
        )

        # Create the pipeline
        pipeline = self.model_manager.get_pipeline(
            model_type, model_params, use_scaler, use_pca, pca_components
        )

        # Perform K-Fold Cross-Validation (if requested)
        if perform_cv:
            cv_scores = self.model_manager.cross_validate_model(
                x_train,
                y_train,
                pipeline,
                random_state=self.config["random_seed"],
                kfolds=k_folds,
            )
            # print(f"K-Fold Cross-Validation Scores: {cv_scores}")

            # Save artifacts: cv_scores
            self.state_manager.save_model_artifacts(
                model_type=model_type, obj=cv_scores, artifact_type="cv_scores"
            )

        # Train and evaluate the pipeline
        evaluations = self.model_manager.train_and_evaluate(
            x_train, x_test, y_train, y_test, pipeline
        )
        # Save artifacts: metrics
        self.state_manager.save_model_artifacts(
            model_type=model_type, obj=evaluations, artifact_type="evaluations"
        )

        # Save artifacts: pipeline
        self.state_manager.save_model_artifacts(
            model_type=model_type, obj=pipeline, artifact_type="pipeline"
        )

        # print(f"Performance Metrics: {evaluations}")
        return evaluations, category

    def load_model_artifacts(self, model_type, artifact):
        """
        Loads saved evaluation metrics.

        Returns:
            dict: Loaded evaluation metrics.
        """

        return self.state_manager.load_model_artifacts(
            model_type=model_type, artifact_type=artifact
        )
