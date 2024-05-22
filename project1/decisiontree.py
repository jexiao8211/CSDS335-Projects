import math
import numpy as np
from collections import Counter

class TreeNode():
  def __init__(self, dataset, feature_index, threshold, prediction_probs, info_gain) -> None:
    self.data = dataset
    self.feature_index = feature_index
    self.threshold = threshold
    self.prediction_probs = prediction_probs
    self.info_gain = info_gain
    self.left = None
    self.right = None

class DecisionTree():

  def __init__(self, max_depth=6, min_samples_split=1, min_info_gain=0.0, num_features_split=None, adaboost_weight=None) -> None:
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_info_gain = min_info_gain
    self.num_features_split = num_features_split
    self.adaboost_weight = adaboost_weight
    self.tree = None


  #using entropy class formula
  def entropy(self, class_probs: list) -> float:
    return sum([-p * np.log2(p) for p in class_probs if p > 0])

  def class_probs(self, targets: list) -> list:
    total = len(targets)
    return [target_count/total for target_count in Counter(targets).values()]

  def data_entropy(self, targets: list) -> float:
    return self.entropy(self.class_probs(targets))

  def partition_entropy(self, subsets: list) -> float:
    total = sum([len(subset) for subset in subsets])
    return sum([self.data_entropy(subset) * (len(subset) / total) for subset in subsets])

  def split(self, dataset: np.array, feature_index: int, threshold: float) -> tuple:
    #all rows that are less than threshold
    below_threshold_group = dataset[:, feature_index] < threshold
    group1 = dataset[below_threshold_group]
    group2 = dataset[~below_threshold_group]
    return group1, group2

  def target_probs(self, dataset: np.array) -> np.array:
    target_values = dataset[:, -1]
    total_target_values = len(target_values)
    target_probs = np.zeros(len(self.target_values), dtype=float)

    for i, target_val in enumerate(self.target_values):
      target_index = np.where(target_values == i)[0]
      if len(target_index) > 0:
        target_probs[i] = len(target_index) / total_target_values

    return target_probs

  def best_split(self, dataset: np.array) -> tuple:
    min_entropy = math.inf
    min_entropy_feature_index = None
    min_entropy_threshold = None

    for i in range(dataset.shape[1]-1):
      threshold = np.median(dataset[:, i])
      subtree1, subtree2 = self.split(dataset, i, threshold)
      split_entropy = self.partition_entropy([subtree1[:, -1], subtree2[:, -1]])
      #finding split with lowest entropy
      if split_entropy < min_entropy:
        min_entropy = split_entropy
        min_entropy_feature_index = i
        min_entropy_threshold = threshold
        subtree1_min, subtree2_min = subtree1, subtree2

    return subtree1_min, subtree2_min, min_entropy_feature_index, min_entropy_threshold, min_entropy

  def build_tree(self, dataset: np.array, curr_depth: int) -> TreeNode:
    if curr_depth >= self.max_depth:
      return None

    subtree1, subtree2, split_feature_index, split_threshold, split_entropy = self.best_split(dataset)

    target_probs = self.target_probs(dataset)

    node_entropy = self.entropy(target_probs)
    info_gain = node_entropy - split_entropy

    node = TreeNode(dataset, split_feature_index, split_threshold, target_probs, info_gain)

    if(self.min_samples_split > subtree1.shape[0] or self.min_samples_split > subtree2.shape[0]):
      return node

    elif info_gain < self.min_info_gain:
      return node

    curr_depth = curr_depth + 1
    #continue recursively until one of return conditions is met
    node.left = self.build_tree(subtree1, curr_depth)
    node.right = self.build_tree(subtree2, curr_depth)

    return node
  def predict_one_sample(self, X: np.array) -> np.array:
    node = self.tree

    while node:
        if node.left is None and node.right is None:  # Check if the node is a leaf
            return node.prediction_probs
        else:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

    # Handle the case where no leaf node is reached
    return None  # Or raise an exception if needed

  def train(self, X_train: np.array, Y_train: np.array) -> None:
    self.target_values = np.unique(Y_train)
    train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

    self.tree = self.build_tree(dataset=train_data, curr_depth=0)


  def predict_probs(self, X_set: np.array) -> np.array:
    pred_prob = np.apply_along_axis(self.predict_one_sample, 1, X_set)

    return pred_prob

  def predict(self, X_set: np.array) -> np.array:
    predictions = []
    for sample in X_set:
        prediction = self.predict_one_sample(sample)
        predictions.append(prediction)
    return np.array(predictions)

def cross_val_10fold2(model, X, y, verbose = True):
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

        #model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc.append(calculate_accuracy(y_val, y_pred))
        prec.append(precision(y_val, y_pred))
        rec.append(recall(y_val, y_pred))
        f1.append(f1_score(y_val, y_pred))


    return acc, prec, rec, f1


def calculate_accuracy(y_actual, y_pred):
    return np.mean(y_actual == y_pred)


def precision(y_actual, y_pred):
    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_positive = np.sum((y_actual == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive + 1e-10)


def recall(y_actual, y_pred):
    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    false_negative = np.sum((y_actual == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative + 1e-10)


def f1_score(y_actual, y_pred):
    prec = precision(y_actual, y_pred)
    rec = recall(y_actual, y_pred)
    return 2 * (prec * rec) / (prec + rec)
