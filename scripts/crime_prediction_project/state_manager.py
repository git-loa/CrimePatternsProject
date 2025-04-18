"""
This module contains the class StateManager
"""

import os
import joblib


class StateManager:
    """
    Class to manage state for the pipeline and persist important artifacts.
    """

    def __init__(self, base_dir="models_directory"):
        """
        Initializes the state manager with default values for the pipeline state
        and sets up a directory for saving/loading artifacts.

        Parameters:
            state_dir (str): Directory to save/load pipeline artifacts.
        """
        self.directories = {
            "residual_analysis": os.path.join(base_dir, "residual_analysis"),
            "models": os.path.join(base_dir, "models"),
            "metrics": os.path.join(base_dir, "metrics"),
            "cv_scores": os.path.join(base_dir, "cross_validations"),
            "best_params": os.path.join(base_dir, "best_params"),
        }
        self.state = {}

        for _, directory in self.directories.items():
            if not os.path.exists(directory):
                os.makedirs(directory)

    def get_directory(self, key):
        """
        Retrieves the directory path for a given key.

        Parameters:
            key (str): The key of the desired directory (e.g., 'residual_analysis').

        Returns:
            str: The path of the directory.
        """
        return self.directories.get(key, None)

    # ===============================
    # State Management
    # ===============================
    def update_state(self, key, value):
        """
        Updates a specific state variable.

        Parameters:
            key (str): The state variable to update.
            value: The new value for the state variable.
        """
        self.state[key] = value

    def get_state(self, key):
        """
        Retrieves a specific state variable.

        Parameters:
            key (str): The state variable to retrieve.

        Returns:
            The value of the specified state variable.
        """
        return self.state.get(key)

    # ===============================
    # Artifact Persistence
    # ===============================
    def save_object(self, obj, filename, directory_key):
        """
        Saves an object to the specified directory within StateManager.

        Parameters:
            obj: Object to save (e.g., trained model, metrics).
            filename (str): Filename for the saved object.
            directory_key (str): Key representing the directory where the object should be saved.
        """
        directory = self.get_directory(directory_key)
        if directory is None:
            raise ValueError(f"Invalid directory key: {directory_key}")

        filepath = os.path.join(directory, filename)
        joblib.dump(obj, filepath)
        print(f"Object saved to {filepath}")

    def load_object(self, filename, directory_key):
        """
        Loads an object from the specified directory.

        Parameters:
            filename (str): Filename of the object to load.
            directory_key (str): Key representing the directory where the object is stored.

        Returns:
            Loaded object.
        """
        directory = self.get_directory(directory_key)
        if directory is None:
            raise ValueError(f"Invalid directory key: {directory_key}")

        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No object found at {filepath}")

        print(f"Object loaded from {filepath}")
        return joblib.load(filepath)

    def save_model_artifacts(
        self,
        model_type,
        obj,
        artifact_type,
        category,
    ):
        """
        Saves model-specific artifacts (trained_models, metrics, or cross-vals scores).

        Parameters:
            model_type (str): Type of model (e.g., "Ridge").
            obj: Object to save (e.g., trained model, metrics, CV scores).
            artifact_type (str): Type of artifact ("trained_model", "metrics", "cv_scores").
            category (str): Data category (e.g., 'Urban', 'Rural', 'all').
        """
        directory_key = "models"
        if artifact_type == "metrics":
            directory_key = "metrics"
        if artifact_type == "cv_scores":
            directory_key = "cv_scores"
        if artifact_type == "best_params":
            directory_key = "best_params"

        filename = f"{model_type}_{artifact_type}_{category}.pkl"
        self.save_object(obj, filename, directory_key)
        self.update_state(f"{model_type}_{artifact_type}", filename)

    def load_model_artifacts(
        self,
        model_type,
        artifact_type,
        category,
    ):
        """
        Loads model-specific artifacts.

        Parameters:
            model_type (str): Type of model (e.g., "Ridge").
            artifact_type (str): Type of artifact ("trained_model", "metrics", "cv_scores").
            category (str): Data category (e.g., 'Urban', 'Rural', 'all').

        Returns:
            Loaded object.
        """
        filename = f"{model_type}_{artifact_type}_{category}.pkl"
        return self.load_object(filename, artifact_type)
