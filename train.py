#!/usr/bin/env python
"""
Training script for Wine Quality Classifier.

This script orchestrates the full training pipeline:
1. Load and preprocess wine quality datasets
2. Split data into training and test sets
3. Train logistic regression model
4. Evaluate model performance
5. Save trained model and scaler
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from data_loader import load_wine_data, check_missing_values, preprocess_features, split_data, apply_smote
from model import WineQualityModel
from evaluation import evaluate_model

# Configuration
RED_WINE_PATH = 'wine+quality dataset/winequality-red.csv'
WHITE_WINE_PATH = 'wine+quality dataset/winequality-white.csv'
MODEL_SAVE_PATH = 'models/trained_model.pkl'
SCALER_SAVE_PATH = 'models/scaler.pkl'
TEST_SIZE = 0.3
RANDOM_STATE = 42
MAX_ITER = 1000


def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("WINE QUALITY CLASSIFIER - TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/6] Loading wine quality datasets...")
    try:
        df = load_wine_data(RED_WINE_PATH, WHITE_WINE_PATH)
        print(f"✓ Loaded {len(df)} wine samples")
        print(f"  - Red wines: {(df['wine_type'] == 0).sum()}")
        print(f"  - White wines: {(df['wine_type'] == 1).sum()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return 1
    
    # Check for missing values
    print("\n[2/6] Checking for missing values...")
    missing = check_missing_values(df)
    total_missing = sum(missing.values())
    if total_missing > 0:
        print(f"✗ Found {total_missing} missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"  - {col}: {count}")
        return 1
    else:
        print("✓ No missing values found")
    
    # Step 2: Preprocess features
    print("\n[3/6] Preprocessing features...")
    try:
        X_scaled, y, scaler = preprocess_features(df)
        print(f"✓ Preprocessed {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        print(f"✓ Feature scaling applied (mean ≈ 0, std ≈ 1)")
        print(f"  - Quality score range: {y.min()} to {y.max()}")
        print(f"  - Quality distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    except Exception as e:
        print(f"✗ Error preprocessing data: {e}")
        return 1
    
    # Step 3: Split data
    print("\n[4/7] Splitting data into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = split_data(
            X_scaled, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"✓ Train set: {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
        print(f"✓ Test set: {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")
        print(f"✓ Stratified split maintains quality distribution")
    except Exception as e:
        print(f"✗ Error splitting data: {e}")
        return 1
    
    # Step 4: Apply SMOTE to training data
    print("\n[5/7] Applying SMOTE to balance training data...")
    try:
        X_train_balanced, y_train_balanced = apply_smote(
            X_train, y_train,
            random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"✗ Error applying SMOTE: {e}")
        print("Continuing with original training data...")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Step 5: Train model
    print("\n[6/7] Training logistic regression model...")
    try:
        model = WineQualityModel(max_iter=MAX_ITER, solver='lbfgs')
        model.train(X_train_balanced, y_train_balanced)
        print(f"✓ Model trained successfully")
        print(f"  - Algorithm: Logistic Regression")
        print(f"  - Solver: lbfgs")
        print(f"  - Max iterations: {MAX_ITER}")
        print(f"  - Training samples (after SMOTE): {len(X_train_balanced):,}")
        print(f"  - Classes: {model.classes_}")
        print(f"  - Features: {model.n_features_}")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return 1
    
    # Step 6: Evaluate model
    print("\n[7/7] Evaluating model on test set (original, unbalanced)...")
    try:
        results = evaluate_model(model, X_test, y_test, plot_cm=False)
        print(f"\n✓ Model evaluation complete")
        print(f"  - Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"✗ Error evaluating model: {e}")
        return 1
    
    # Step 7: Save model and scaler
    print("\n[8/8] Saving trained model and scaler...")
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model.save_model(MODEL_SAVE_PATH, SCALER_SAVE_PATH, scaler)
        print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
        print(f"✓ Scaler saved to: {SCALER_SAVE_PATH}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel Summary:")
    print(f"  - Original training samples: {len(X_train)}")
    print(f"  - SMOTE-balanced training samples: {len(X_train_balanced)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Test accuracy: {results['accuracy']:.4f}")
    print(f"  - Model saved: {MODEL_SAVE_PATH}")
    print(f"  - Scaler saved: {SCALER_SAVE_PATH}")
    print("\nNext steps:")
    print("  1. Review the classification report above")
    print("  2. Run the Flask application: python src/app.py")
    print("  3. Make predictions via the web interface")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
