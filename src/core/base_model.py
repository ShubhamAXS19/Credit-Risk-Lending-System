from abc import ABC, abstractmethod
import joblib
import os
from typing import Any, Dict

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models in the system.
    Enforces a consistent interface for training, prediction, and persistence.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X, y, **kwargs):
        """
        Train the model using the provided data.
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the trained model.
        """
        pass

    def save(self, directory: str = "models"):
        """
        Save the model to disk.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, directory: str = "models"):
        """
        Load the model from disk.
        """
        path = os.path.join(directory, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
