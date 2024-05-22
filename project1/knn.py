import numpy as np
from math import sqrt

class KNNModel():
    @staticmethod
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    @staticmethod
    def get_neighbors(train_data, test_row, num_neighbors):
        distances = []
        for train_row in train_data:
            train_test_dist = KNNModel.euclidean_distance(train_row[:-1], test_row[:-1])
            distances.append((train_row, train_test_dist))
        distances.sort(key=lambda x: x[1])
        neighbors = [x[0] for x in distances[:num_neighbors]]
        return neighbors

    @staticmethod
    def predict(train_data, test_row, num_neighbors):
        neighbors = KNNModel.get_neighbors(train_data, test_row, num_neighbors)
        class_values = [row[-1] for row in neighbors]  # Extract the class label from each neighbor
        prediction = max(set(class_values), key=class_values.count)
        return prediction

def cross_val_10fold1(model, X, y, num_neighbors_values, verbose=True):
    n = len(X)
    fold_size = n // 10
    results = {'num_neighbors': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for num_neighbors in num_neighbors_values:
        acc = []
        prec = []
        rec = []
        f1 = []

        for i in range(10):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 9 else n

            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]

            X_train = np.concatenate([X[:start_idx], X[end_idx:]], axis=0)
            y_train = np.concatenate([y[:start_idx], y[end_idx:]], axis=0)

            y_pred = [model.predict(np.column_stack((X_train, y_train)), test_row, num_neighbors) for test_row in X_val]

            acc.append(calculate_accuracy(y_val, y_pred))
            prec.append(precision(y_val, y_pred))
            rec.append(recall(y_val, y_pred))
            f1.append(f1_score(y_val, y_pred))

        results['num_neighbors'].append(num_neighbors)
        results['accuracy'].append(np.mean(acc))
        results['precision'].append(np.mean(prec))
        results['recall'].append(np.mean(rec))
        results['f1_score'].append(np.mean(f1))

    return results


def calculate_accuracy(y_actual, y_pred):
    return np.mean(y_actual == y_pred)

def precision(y_actual, y_pred):
    y_pred = np.array(y_pred)  # Convert y_pred to numpy array
    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_positive = np.sum((y_actual == 0) & (y_pred == 1))
    if true_positive + false_positive == 0:
        return 0  # Handle division by zero
    else:
        return true_positive / (true_positive + false_positive)


def recall(y_actual, y_pred):
    y_pred = np.array(y_pred)  # Convert y_pred to numpy array
    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_negative = np.sum((y_actual == 1) & (y_pred == 0))
    if true_positive + false_negative == 0:
        return 0  # Handle division by zero
    else:
        return true_positive / (true_positive + false_negative)

def f1_score(y_actual, y_pred):
    prec = precision(y_actual, y_pred)
    rec = recall(y_actual, y_pred)
    if prec + rec == 0:
        return 0  # Handle division by zero
    else:
        return 2 * (prec * rec) / (prec + rec)
