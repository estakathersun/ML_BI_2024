import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp, tn, fp, fn = 0, 0, 0, 0
    for y_pred, y_true in zip(y_pred, y_true):
        if y_pred == y_true and y_true == 1:
            tp += 1
        elif y_pred == y_true and y_true == 0:
            tn += 1
        elif y_pred != y_true and y_true == 1:
            fn += 1
        else:
            fp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)

    print(f'TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\n'
          f'Accuracy: {accuracy}\n'
          f'Precision: {precision}\n'
          f'Recall: {recall}\n'
          f'F1: {f1}')
    return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_true, y_pred):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct_preds = np.sum(y_pred == y_true)
    accuracy = correct_preds / y_true.shape[0]

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    mean_y = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
