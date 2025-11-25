"""
Model training and prediction module for Wine Quality Classifier.

This module provides the WineQualityModel class for training a logistic regression
classifier and making predictions on wine quality.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional


class WineQualityModel:
    """
    Wine Quality Classification Model using Logistic Regression.
    
    This class encapsulates a scikit-learn LogisticRegression model for
    multi-class classification of wine quality scores.
    """
    
    def __init__(self, max_iter: int = 1000, solver: str = 'lbfgs'):
        """
        Initialize the Wine Quality Model.
        
        Args:
            max_iter: Maximum number of iterations for solver convergence (default: 1000)
            solver: Algorithm to use for optimization (default: 'lbfgs')
        """
        self.model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            multi_class='auto',
            random_state=42
        )
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the logistic regression model on training data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            
        Raises:
            ValueError: If training data is invalid or has single class
        """
        if len(X_train) == 0:
            raise ValueError("Training data is empty")
        
        if len(np.unique(y_train)) == 1:
            raise ValueError("Training data contains only one quality class. Cannot train classifier.")
        
        if len(X_train) < 100:
            print("Warning: Training set has fewer than 100 samples. Model performance may be poor.")
        
        # Fit the model
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
        except Exception as e:
            # If convergence fails, this will be caught
            raise RuntimeError(f"Model training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate quality score predictions for wine samples.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        
        Returns:
            Predicted quality scores of shape (n_samples,)
            
        Raises:
            RuntimeError: If model hasn't been trained yet
            ValueError: If input has wrong number of features
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions. Call train() first.")
        
        # Check feature count matches training data
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but model expects {self.model.n_features_in_} features"
            )
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability distributions over quality classes.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        
        Returns:
            Probability distributions of shape (n_samples, n_classes)
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions. Call train() first.")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def save_model(self, model_path: str, scaler_path: str, scaler: StandardScaler) -> None:
        """
        Serialize model and scaler to disk using joblib.
        
        Args:
            model_path: Path to save the trained model (e.g., 'models/trained_model.pkl')
            scaler_path: Path to save the scaler (e.g., 'models/scaler.pkl')
            scaler: StandardScaler object used for feature scaling
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model. Call train() first.")
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save the scaler
        joblib.dump(scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str) -> StandardScaler:
        """
        Load saved model and scaler from disk.
        
        Args:
            model_path: Path to the saved model file
            scaler_path: Path to the saved scaler file
        
        Returns:
            Loaded StandardScaler object
            
        Raises:
            FileNotFoundError: If model or scaler files don't exist
        """
        try:
            # Load the model
            self.model = joblib.load(model_path)
            self.is_trained = True
            
            # Load the scaler
            scaler = joblib.load(scaler_path)
            self.scaler = scaler
            
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            
            return scaler
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model or scaler file not found: {e}")
    
    @property
    def classes_(self) -> np.ndarray:
        """
        Get the unique quality classes the model was trained on.
        
        Returns:
            Array of unique quality scores
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.classes_
    
    @property
    def n_features_(self) -> int:
        """
        Get the number of features the model expects.
        
        Returns:
            Number of input features
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.n_features_in_
