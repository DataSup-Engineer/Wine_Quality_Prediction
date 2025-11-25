"""
Property-based tests for data_loader module using Hypothesis.

These tests verify universal properties that should hold across all valid inputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import pytest

from data_loader import preprocess_features


# Property 2: Feature scaling produces standardized distributions
@settings(max_examples=100)
@given(
    # Generate random wine datasets with varying feature ranges
    n_samples=st.integers(min_value=10, max_value=1000),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_feature_scaling_standardization(n_samples, seed):
    """
    **Feature: wine-quality-classifier, Property 2: Feature scaling produces standardized distributions**
    **Validates: Requirements 1.5, 5.2**
    
    For any dataset of wine samples, after applying StandardScaler, each feature should have 
    a mean approximately equal to 0 and standard deviation approximately equal to 1 
    (within tolerance of 0.1).
    """
    np.random.seed(seed)
    
    # Generate random wine data with 12 features (11 physicochemical + 1 wine_type)
    # Use realistic ranges for wine features
    data = {
        'fixed acidity': np.random.uniform(3.8, 15.9, n_samples),
        'volatile acidity': np.random.uniform(0.08, 1.58, n_samples),
        'citric acid': np.random.uniform(0.0, 1.66, n_samples),
        'residual sugar': np.random.uniform(0.6, 65.8, n_samples),
        'chlorides': np.random.uniform(0.009, 0.611, n_samples),
        'free sulfur dioxide': np.random.uniform(1, 289, n_samples),
        'total sulfur dioxide': np.random.uniform(6, 440, n_samples),
        'density': np.random.uniform(0.987, 1.039, n_samples),
        'pH': np.random.uniform(2.72, 4.01, n_samples),
        'sulphates': np.random.uniform(0.22, 2.0, n_samples),
        'alcohol': np.random.uniform(8.0, 14.9, n_samples),
        'wine_type': np.random.randint(0, 2, n_samples),
        'quality': np.random.randint(3, 10, n_samples)  # Quality scores 3-9
    }
    
    df = pd.DataFrame(data)
    
    # Apply preprocessing
    X_scaled, y, scaler = preprocess_features(df)
    
    # Property: Each feature should have mean ≈ 0 and std ≈ 1
    feature_means = X_scaled.mean(axis=0)
    feature_stds = X_scaled.std(axis=0)
    
    # Check all features have mean close to 0 (within tolerance of 0.1)
    assert np.all(np.abs(feature_means) < 0.1), \
        f"Feature means not close to 0: {feature_means}"
    
    # Check all features have std close to 1 (within tolerance of 0.1)
    assert np.all(np.abs(feature_stds - 1.0) < 0.1), \
        f"Feature stds not close to 1: {feature_stds}"
    
    # Verify output shapes are correct (11 features, wine_type excluded)
    assert X_scaled.shape == (n_samples, 11), \
        f"Expected shape ({n_samples}, 11), got {X_scaled.shape}"
    assert y.shape == (n_samples,), \
        f"Expected target shape ({n_samples},), got {y.shape}"


if __name__ == "__main__":
    # Run the property test
    test_feature_scaling_standardization()
    print("✅ Property test passed: Feature scaling produces standardized distributions")



# Property 3: Correlation coefficients are bounded
@settings(max_examples=100)
@given(
    # Generate random subsets of wine data
    n_samples=st.integers(min_value=20, max_value=1000),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_correlation_coefficients_bounded(n_samples, seed):
    """
    **Feature: wine-quality-classifier, Property 3: Correlation coefficients are bounded**
    **Validates: Requirements 2.4**
    
    For any computed correlation matrix between features and quality scores, 
    all correlation values should be in the range [-1, 1].
    """
    np.random.seed(seed)
    
    # Generate random wine data with 12 features (11 physicochemical + 1 wine_type)
    data = {
        'fixed acidity': np.random.uniform(3.8, 15.9, n_samples),
        'volatile acidity': np.random.uniform(0.08, 1.58, n_samples),
        'citric acid': np.random.uniform(0.0, 1.66, n_samples),
        'residual sugar': np.random.uniform(0.6, 65.8, n_samples),
        'chlorides': np.random.uniform(0.009, 0.611, n_samples),
        'free sulfur dioxide': np.random.uniform(1, 289, n_samples),
        'total sulfur dioxide': np.random.uniform(6, 440, n_samples),
        'density': np.random.uniform(0.987, 1.039, n_samples),
        'pH': np.random.uniform(2.72, 4.01, n_samples),
        'sulphates': np.random.uniform(0.22, 2.0, n_samples),
        'alcohol': np.random.uniform(8.0, 14.9, n_samples),
        'wine_type': np.random.randint(0, 2, n_samples),
        'quality': np.random.randint(3, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Compute correlation matrix
    correlation_matrix = df.corr()
    
    # Property: All correlation values must be in range [-1, 1]
    # Extract all correlation values
    corr_values = correlation_matrix.values.flatten()
    
    # Check that all correlations are bounded
    assert np.all(corr_values >= -1.0), \
        f"Found correlation values less than -1: {corr_values[corr_values < -1.0]}"
    
    assert np.all(corr_values <= 1.0), \
        f"Found correlation values greater than 1: {corr_values[corr_values > 1.0]}"
    
    # Additional check: diagonal should be exactly 1 (self-correlation)
    diagonal = np.diag(correlation_matrix)
    assert np.allclose(diagonal, 1.0), \
        f"Diagonal values (self-correlation) should be 1.0, got: {diagonal}"
    
    # Check that matrix is symmetric
    assert np.allclose(correlation_matrix, correlation_matrix.T), \
        "Correlation matrix should be symmetric"


if __name__ == "__main__":
    # Run all property tests
    test_feature_scaling_standardization()
    print("✅ Property test passed: Feature scaling produces standardized distributions")
    
    test_correlation_coefficients_bounded()
    print("✅ Property test passed: Correlation coefficients are bounded")
