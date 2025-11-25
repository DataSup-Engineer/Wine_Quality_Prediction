"""
Data loading and preprocessing module for Wine Quality Classifier.

This module provides functions to load wine quality datasets, check for missing values,
preprocess features, and split data into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict


def load_wine_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Load red and white wine datasets and combine them into a single DataFrame.
    
    Args:
        red_path: Path to the red wine CSV file
        white_path: Path to the white wine CSV file
    
    Returns:
        Combined DataFrame with shape (6497, 13) including wine_type column
        
    Raises:
        FileNotFoundError: If CSV files don't exist at specified paths
        ValueError: If CSV has wrong structure or missing expected columns
    """
    try:
        # Load red wine data (semicolon delimiter)
        red_wine = pd.read_csv(red_path, sep=';')
        red_wine['wine_type'] = 0  # 0 for red wine
        
        # Load white wine data (semicolon delimiter)
        white_wine = pd.read_csv(white_path, sep=';')
        white_wine['wine_type'] = 1  # 1 for white wine
        
        # Combine datasets
        combined_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
        
        # Verify expected columns exist
        expected_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality'
        ]
        
        missing_cols = [col for col in expected_features if col not in combined_data.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        return combined_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Check for missing values in the dataset.
    
    Args:
        df: DataFrame to check for missing values
    
    Returns:
        Dictionary mapping column names to count of missing values
    """
    missing_counts = df.isnull().sum().to_dict()
    return missing_counts


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Separate features from target and apply StandardScaler to features.
    
    Args:
        df: DataFrame containing wine data with 'quality' column
    
    Returns:
        Tuple of (scaled_features, target_labels, fitted_scaler)
        - scaled_features: numpy array of shape (n_samples, 11) with standardized features
        - target_labels: numpy array of shape (n_samples,) with quality scores
        - fitted_scaler: StandardScaler object fitted on the features
        
    Raises:
        ValueError: If dataset is empty or contains invalid values
        TypeError: If features contain non-numeric values
    """
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    # Separate features from target
    # Features: only the 11 physicochemical properties (exclude 'quality' and 'wine_type')
    feature_cols = [col for col in df.columns if col not in ['quality', 'wine_type']]
    X = df[feature_cols].values
    y = df['quality'].values
    
    # Check for non-numeric values
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("Features contain non-numeric values")
    
    # Check for infinite values
    if np.any(np.isinf(X)):
        raise ValueError("Features contain infinite values. Please clean the data.")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets with stratification.
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        test_size: Proportion of dataset to include in test split (default: 0.3)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If test_size is not between 0 and 1
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Use stratified split to maintain quality score distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE with targeted oversampling strategy to balance minority classes.
    
    This function uses a targeted approach that only oversamples severely
    underrepresented quality classes while leaving moderately represented
    classes unchanged.
    
    Targeted sampling strategy:
    - Quality 3: 30 → 500 samples
    - Quality 4: 216 → 800 samples
    - Quality 8: 193 → 500 samples
    - Quality 9: 5 → 300 samples
    - Quality 5, 6, 7: unchanged
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (X_resampled, y_resampled)
        - X_resampled: Resampled features with synthetic samples
        - y_resampled: Resampled labels with balanced distribution
        
    Raises:
        ValueError: If training data is empty or invalid
    """
    if len(X_train) == 0:
        raise ValueError("Training data is empty")
    
    # Get original class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    original_dist = dict(zip(unique, counts))
    
    print("\n" + "=" * 60)
    print("SMOTE - Class Imbalance Handling")
    print("=" * 60)
    print("\nOriginal class distribution:")
    for quality, count in sorted(original_dist.items()):
        print(f"  Quality {quality}: {count:,} samples")
    
    # Define targeted sampling strategy
    # Only oversample severely underrepresented classes
    sampling_strategy = {}
    
    # Add targets for minority classes that need oversampling
    if 3 in original_dist and original_dist[3] < 500:
        sampling_strategy[3] = 500
    if 4 in original_dist and original_dist[4] < 800:
        sampling_strategy[4] = 800
    if 8 in original_dist and original_dist[8] < 500:
        sampling_strategy[8] = 500
    if 9 in original_dist and original_dist[9] < 300:
        sampling_strategy[9] = 300
    
    # If no classes need oversampling, return original data
    if not sampling_strategy:
        print("\n✓ No oversampling needed - all classes sufficiently represented")
        print("=" * 60)
        return X_train, y_train
    
    print("\nTargeted oversampling strategy:")
    for quality, target in sorted(sampling_strategy.items()):
        original = original_dist.get(quality, 0)
        synthetic = target - original
        print(f"  Quality {quality}: {original} → {target} (+{synthetic} synthetic)")
    
    # Apply SMOTE
    try:
        # Determine k_neighbors based on smallest class size
        min_samples = min([original_dist[cls] for cls in sampling_strategy.keys()])
        k_neighbors = min(5, min_samples - 1)  # SMOTE needs k_neighbors < n_samples
        
        if k_neighbors < 1:
            print(f"\n✗ Cannot apply SMOTE: smallest class has only {min_samples} samples")
            print("  SMOTE requires at least 2 samples per class")
            print("Returning original training data")
            print("=" * 60)
            return X_train, y_train
        
        print(f"\nSMOTE configuration: k_neighbors={k_neighbors}")
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Get resampled class distribution
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        resampled_dist = dict(zip(unique_new, counts_new))
        
        print("\nResampled class distribution:")
        for quality, count in sorted(resampled_dist.items()):
            original = original_dist.get(quality, 0)
            change = count - original
            change_str = f"(+{change})" if change > 0 else ""
            print(f"  Quality {quality}: {count:,} samples {change_str}")
        
        total_original = len(y_train)
        total_resampled = len(y_resampled)
        total_synthetic = total_resampled - total_original
        
        print(f"\nTotal samples: {total_original:,} → {total_resampled:,} (+{total_synthetic:,} synthetic)")
        print("✓ SMOTE applied successfully")
        print("=" * 60)
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"\n✗ Error applying SMOTE: {e}")
        print("Returning original training data")
        print("=" * 60)
        return X_train, y_train
