"""This module contains utility functions for data loading and preprocessing."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset_1(rs = 1):
    """
    Loads and preprocesses project dataset 1. Proprocessing: shuffle samples 
    and standardize all features.

    Parameters:
        rs (float): random state for data shuffling
    """

    data = pd.read_csv('project1_dataset1.txt', delimiter='\t', header=None)
    # Shuffle data
    shuffled_data = data.sample(frac=1, random_state=rs)

    # Seperate X and y
    X_raw = shuffled_data.iloc[:, :-1].copy()
    y = shuffled_data.iloc[:, -1].copy()

    # Standardize all features in X
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    X = pd.DataFrame(X, columns = X_raw.columns)
    return X.values, y.values


def load_dataset_2(rs = 1):
    """
    Loads and preprocesses project dataset 1. Proprocessing: shuffle samples 
    and standardize all features.

    Parameters:
        rs (float): random state for data shuffling
    """

    data = pd.read_csv('project1_dataset2.txt', delimiter='\t', header=None)
    # Shuffle data
    shuffled_data = data.sample(frac=1, random_state=rs)

    # Seperate X and y
    X_raw = shuffled_data.iloc[:, :-1].copy()
    y = shuffled_data.iloc[:, -1].copy()

    # Convert Col 4 into binary
    X_raw[4] = X_raw[4].replace({'Present': 1, 'Absent': 0})

    # Standardize all features in X
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    X = pd.DataFrame(X, columns = X_raw.columns)
    return X.values, y.values

