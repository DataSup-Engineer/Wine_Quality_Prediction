"""
Model evaluation module for Wine Quality Classifier.

This module provides functions to evaluate model performance including
accuracy, classification reports, and confusion matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Optional


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate overall classification accuracy.
    
    Args:
        y_true: True labels of shape (n_samples,)
        y_pred: Predicted labels of shape (n_samples,)
    
    Returns:
        Accuracy score in range [0, 1]
    """
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a classification report with precision, recall, and F1-score.
    
    Args:
        y_true: True labels of shape (n_samples,)
        y_pred: Predicted labels of shape (n_samples,)
    
    Returns:
        Formatted string containing classification metrics for each class
    """
    report = classification_report(y_true, y_pred)
    return report


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Generate confusion matrix showing predicted vs actual quality scores.
    
    Args:
        y_true: True labels of shape (n_samples,)
        y_pred: Predicted labels of shape (n_samples,)
    
    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix of shape (n_classes, n_classes)
        classes: List of class labels (quality scores)
        title: Title for the plot (default: 'Confusion Matrix')
        cmap: Colormap for the heatmap (default: 'Blues')
        figsize: Figure size as (width, height) tuple (default: (10, 8))
        save_path: Optional path to save the figure (default: None)
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Quality Score', fontsize=12)
    plt.xlabel('Predicted Quality Score', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def print_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Test"
) -> None:
    """
    Print comprehensive evaluation metrics in a readable format.
    
    Args:
        y_true: True labels of shape (n_samples,)
        y_pred: Predicted labels of shape (n_samples,)
        dataset_name: Name of the dataset being evaluated (default: "Test")
    """
    print("=" * 80)
    print(f"EVALUATION METRICS - {dataset_name} Set")
    print("=" * 80)
    
    # Overall accuracy
    accuracy = compute_accuracy(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\nClassification Report:")
    print("-" * 80)
    report = generate_classification_report(y_true, y_pred)
    print(report)
    
    # Confusion matrix summary
    cm = generate_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"Total Predictions: {cm.sum()}")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    print("-" * 80)
    for i, class_label in enumerate(np.unique(y_true)):
        if i < len(cm):
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            print(f"  Quality {int(class_label)}: {class_accuracy:.4f} ({class_correct}/{class_total})")
    
    print("=" * 80)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plot_cm: bool = True,
    save_cm_path: Optional[str] = None
) -> dict:
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features of shape (n_samples, n_features)
        y_test: Test labels of shape (n_samples,)
        plot_cm: Whether to plot confusion matrix (default: True)
        save_cm_path: Optional path to save confusion matrix plot (default: None)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = compute_accuracy(y_test, y_pred)
    report = generate_classification_report(y_test, y_pred)
    cm = generate_confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print_evaluation_metrics(y_test, y_pred, dataset_name="Test")
    
    # Plot confusion matrix
    if plot_cm:
        classes = sorted(np.unique(y_test))
        plot_confusion_matrix(
            cm,
            classes=[int(c) for c in classes],
            title='Wine Quality Prediction - Confusion Matrix',
            save_path=save_cm_path
        )
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }
