import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def get_metrics(y_true, y_pred):
    """
    Calculate confusion matrix, precision, and recall for a multi-class classification problem.

    Args:
    y_true (list or array): True labels.
    y_pred (list or array): Predicted labels.

    Returns:
    dict: Dictionary containing confusion matrix, precision, and recall for each class.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision and recall for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    # Get unique classes
    classes = sorted(set(y_true))
    
    # Create a dictionary to store precision and recall for each class
    metrics_dict = {
        "confusion_matrix": cm,
        "precision": {cls: precision[idx] for idx, cls in enumerate(classes)},
        "recall": {cls: recall[idx] for idx, cls in enumerate(classes)}
    }
    
    return metrics_dict

