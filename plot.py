""" This file will contain all the plotting functions for the machine learning model.
It will contain these helpful function to plot all the necessary vizualization for the machine learning model.
These functions will be used to plot the BERT model that has been trained to perform NER on finance advise documents and other financial docs,

The functions will be used to plot the following:
1. Plot the confusion matrix
2. Plot the classification report
3. Plot the ROC curve
4. Plot the precision-recall curve
5. Plot the learning curve
6. Plot the validation curve
7. Plot the feature importance
8. Plot the correlation matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
from itertools import cycle


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix for NER predictions.
    
    Args:
        y_true: True labels (flattened for all tokens)
        y_pred: Predicted labels (flattened for all tokens)
        labels: List of label names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap for the plot
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = 'Classification Report',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot classification report as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    
    # Extract metrics for each class
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.iloc[:-3, :-1]  # Remove accuracy, macro avg, weighted avg and support
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Score'}, ax=ax)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Entity Types', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: Union[List, np.ndarray],
    y_pred_proba: np.ndarray,
    n_classes: int,
    labels: Optional[List[str]] = None,
    title: str = 'ROC Curve - Multi-class',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multi-class NER classification.
    
    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
        n_classes: Number of classes
        labels: List of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = cycle(plt.cm.tab10.colors)
    for i, color in zip(range(n_classes), colors):
        label_name = labels[i] if labels else f'Class {i}'
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label_name} (AUC = {roc_auc[i]:.2f})')
    
    # Plot micro-average
    ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: Union[List, np.ndarray],
    y_pred_proba: np.ndarray,
    n_classes: int,
    labels: Optional[List[str]] = None,
    title: str = 'Precision-Recall Curve - Multi-class',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class NER classification.
    
    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
        n_classes: Number of classes
        labels: List of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision[i] = auc(recall[i], precision[i])
    
    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_proba.ravel()
    )
    avg_precision["micro"] = auc(recall["micro"], precision["micro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = cycle(plt.cm.tab10.colors)
    for i, color in zip(range(n_classes), colors):
        label_name = labels[i] if labels else f'Class {i}'
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{label_name} (AP = {avg_precision[i]:.2f})')
    
    # Plot micro-average
    ax.plot(recall["micro"], precision["micro"], color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AP = {avg_precision["micro"]:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curve(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = 'F1 Score',
    title: str = 'Learning Curves',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curves showing loss and metrics over epochs.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch (e.g., F1 score)
        val_metrics: Validation metrics per epoch
        metric_name: Name of the metric being plotted
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    epochs = range(1, len(train_losses) + 1)
    
    if train_metrics is not None and val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Plot metrics
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}', linewidth=2, marker='o')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(f'{metric_name} Curves', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Plot only losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_validation_curve(
    param_range: Union[List, np.ndarray],
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    param_name: str = 'Parameter',
    score_name: str = 'Score',
    title: str = 'Validation Curve',
    log_scale: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot validation curve showing performance vs hyperparameter values.
    
    Args:
        param_range: Range of parameter values
        train_scores: Training scores for each parameter value (mean or array)
        val_scores: Validation scores for each parameter value (mean or array)
        param_name: Name of the parameter
        score_name: Name of the score metric
        title: Plot title
        log_scale: Whether to use log scale for x-axis
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std if multiple runs provided
    if train_scores.ndim > 1:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')
    else:
        train_mean = train_scores
        val_mean = val_scores
    
    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(score_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: Union[List, np.ndarray],
    title: str = 'Feature Importance',
    top_k: Optional[int] = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance scores (can be used for token attention weights in BERT).
    
    Args:
        feature_names: Names of features/tokens
        importance_scores: Importance scores for each feature
        title: Plot title
        top_k: Number of top features to display (None for all)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1]
    
    if top_k is not None:
        indices = indices[:top_k]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_scores = [importance_scores[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_scores, color=colors)
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = 'Correlation Matrix',
    cmap: str = 'coolwarm',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix (useful for entity co-occurrence analysis).
    
    Args:
        data: DataFrame or 2D array containing the data
        labels: Labels for the matrix (if data is numpy array)
        title: Plot title
        cmap: Colormap for the plot
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    if isinstance(data, np.ndarray):
        corr_matrix = np.corrcoef(data.T)
        if labels is None:
            labels = [f'Feature {i}' for i in range(corr_matrix.shape[0])]
    else:
        corr_matrix = data.corr()
        labels = data.columns.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_entity_distribution(
    entity_counts: Dict[str, int],
    title: str = 'Entity Distribution',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of entities in the dataset (bonus function for NER).
    
    Args:
        entity_counts: Dictionary mapping entity types to their counts
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    entities = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(entities)))
    bars = ax.bar(entities, counts, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Set default style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
