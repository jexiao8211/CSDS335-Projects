import numpy as np

"""This module contains utility functions for 10 fold cross validation and
testing evaluation."""

def cross_val_10fold(model, X, y, verbose = True):
    """
    Performs 10-fold cross validation.

    Parameters:
        model (obj): One of the models defined for this project
        X (np.ndarray): Data including features of the train set
        y (np.ndarray): Y labels of the train set
        verbose (boolean): If True, prints out statistical report
    """

    fold_size = len(X) // 10 + 1
    acc = []
    prec = []
    rec = []
    f1 = []

    for i in range(10):
        # Get start and end index of the validation set
        val_start_idx = i*fold_size
        val_end_idx = val_start_idx + fold_size

        # Slice out validation set
        X_val = X[val_start_idx:val_end_idx]
        y_val = y[val_start_idx:val_end_idx]

        # Slice out training set
        X_train = np.concatenate([X[:val_start_idx], X[val_end_idx:]], axis=0)
        y_train = np.concatenate([y[:val_start_idx], y[val_end_idx:]], axis=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc.append(calculate_accuracy(y_val, y_pred))
        prec.append(precision(y_val, y_pred))
        rec.append(recall(y_val, y_pred))
        f1.append(f1_score(y_val, y_pred))

    if verbose:
        print('----- Over 10 folds -----')
        print(f'Accuracy: {np.mean(acc):.4}')
        print(f'Precision: {np.mean(prec):.4}')
        print(f'Recall: {np.mean(rec):.4}')
        print(f'F1-Score: {np.mean(f1):.4}')

    return acc, prec, rec, f1


def calculate_accuracy(y_actual, y_pred):
    """Calculate accuracy: (TP+TN)/(TP+TN+FP+FN)"""

    return np.mean(y_actual == y_pred)


def precision(y_actual, y_pred):
    """Calculate precision: (TP)/(TP+FP)"""

    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_positive = np.sum((y_actual == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive + 1e-10)


def recall(y_actual, y_pred):
    """Calculate recall: (TP)/(TP+FN)"""

    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_negative = np.sum((y_actual == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative + 1e-10)


def f1_score(y_actual, y_pred):
    """Calculate F1 score: (2*precision*recall)/(precision+recall)"""

    prec = precision(y_actual, y_pred)
    rec = recall(y_actual, y_pred)
    return 2 * (prec * rec) / (prec + rec)


def eval_predictions(y_actual, y_pred):
    """Calculate all statistics (accuracy, precision, recall, f1)."""
    acc = calculate_accuracy(y_actual, y_pred)
    prec = precision(y_actual, y_pred)
    rec = recall(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)

    return acc, prec, rec, f1