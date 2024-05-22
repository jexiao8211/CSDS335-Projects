import numpy as np
import pandas as pd
from cv_and_eval import *

class NaiveBayes:
    """
    Naive Bayes Classifier. Assumes that all features are numerical with
    normal distribution.

    Attributes:
    prob_class (dict): Each key is a class, and the associated value is the
                       probabilty of the class.
    cond_distr_mean (dict): Each key is a class, and the associated value is a
                            list of the means of each feature, conditional on
                            Y=class.
    cond_distr_var (dict): Each key is a class, and the associated value is a
                           list of the variances of each feature, conditional on
                           Y=class.
    """

    def __init__(self):
        self.prob_class = {}
        self.cond_distr_mean = {}
        self.cond_distr_var = {}

    def fit(self, X, y):
        # calculate conditional probabilty of each x with each y
        for c in np.unique(y):
            X_c = X[y == c]
            self.prob_class[c] = len(X_c)/len(X)

            self.cond_distr_mean[c] = np.mean(X_c, axis=0)
            self.cond_distr_var[c] = np.var(X_c, axis=0)

    def predict(self, X):
        prob_x_in_c = {}
        for c in list(self.prob_class.keys()):
            prob_c = self.prob_class[c]
            vars = self.cond_distr_var[c]
            means = self.cond_distr_mean[c]

            # Normal distribution
            prob_x_given_c = (1 / (np.sqrt(2 * np.pi * vars))) \
                              * np.exp(-((X - means)**2) / ( 2 * vars))

            # Product of all conditional probabilities
            prob_x_given_c = prob_x_given_c.prod(axis=1)

            # Calculate the prob that each x is in class c
            prob_x_in_c[c] = prob_c * prob_x_given_c

        prob_x_in_c = pd.DataFrame(prob_x_in_c)

        # Get a list of all predictions
        predictions = prob_x_in_c.idxmax(axis=1)

        return predictions

def naive_bayes_test_params(X, y, split_vals):
    """ Grid search parameters, testing with 10-fold cross validation
    returns: best performing split_threshold from the grid search
    """
    
    # Keep track of best params and associated statistics
    best_threshold = -1
    best_acc = -1
    best_prec = -1
    best_rec = -1
    best_f1 = -1

    for t in split_vals:
        # Split data
        # Train test split
        split_ind = int(t*len(X))
        X_train = X[:split_ind]
        y_train = y[:split_ind]

        X_test = X[split_ind:]
        y_test = y[split_ind:]

        model = NaiveBayes()
        acc_list, prec_list, rec_list, f1_list = \
                    cross_val_10fold(model, X_train, y_train, False)

        acc = np.mean(acc_list)
        prec = np.mean(prec_list)
        rec = np.mean(rec_list)
        f1 = np.mean(f1_list)

        if acc > best_acc:
            best_threshold = t
            best_acc = acc
            best_prec = prec
            best_rec = rec
            best_f1 = f1

    print(f'best split threshold: {best_threshold}')
    print(f'best acc: {best_acc}')
    print(f'best prec: {best_prec}')
    print(f'best rec: {best_rec}')
    print(f'best f1: {best_f1}')

    return best_threshold
