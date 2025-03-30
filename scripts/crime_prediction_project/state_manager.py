"""
This module contains the class StateManager
"""

import os
import joblib


class StateManager:
    """
    Class to manage state for the pipeline and persist important artifacts.
    """

    def __init__(self, state_dir="state"):
        """
        Initializes the state manager with default values for the pipeline state
        and sets up a directory for saving/loading artifacts.

        Parameters:
            state_dir (str): Directory to save/load pipeline artifacts.
        """
        self.state_dir = state_dir
        self.state = {}

        # Ensure the state directory exists
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

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
    def save_object(self, obj, filename):
        """
        Saves an object to the state directory.

        Parameters:
            obj: Object to save (e.g., trained model, scaler).
            filename (str): Filename for the saved object.
        """
        filepath = os.path.join(self.state_dir, filename)
        joblib.dump(obj, filepath)
        print(f"Object saved to {filepath}")

    def load_object(self, filename):
        """
        Loads an object from the state directory.

        Parameters:
            filename (str): Filename of the object to load.

        Returns:
            Loaded object.
        """
        filepath = os.path.join(self.state_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No object found at {filepath}")
        print(f"Object loaded from {filepath}")
        return joblib.load(filepath)

    def save_model_artifacts(self, model_type, obj, artifact_type):
        """
        Saves model-specific artifacts (trained_models, metrics, or cross-vals scores).

        Parameters:
            model_type (str): Type of model (e.g., "Ridge").
            obj: Object to save (e.g., trained model, metrics, CV scores).
            artifact_type (str): Type of artifact ("trained_model", "metrics", "cv_scores").
        """
        filename = f"{model_type}_{artifact_type}.pkl"
        self.save_object(obj, filename)
        self.update_state(f"{model_type}_{artifact_type}", filename)

    def load_model_artifacts(self, model_type, artifact_type):
        """
        Loads model-specific artifacts.

        Parameters:
            model_type (str): Type of model (e.g., "Ridge").
            artifact_type (str): Type of artifact ("trained_model", "metrics", "cv_scores").

        Returns:
            Loaded object.
        """
        filename = f"{model_type}_{artifact_type}.pkl"
        return self.load_object(filename)
