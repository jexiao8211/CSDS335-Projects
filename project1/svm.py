"""
Following this implementation of quadratic problem solve for SVM: 
https://xavierbourretsicotte.github.io/SVM_implementation.html

Also includes parameter grid search for SVM
"""

import numpy as np
from cv_and_eval import *
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class rbf_SVM:
    """
    SVM classifier using Kernel as Radial Basis Function. Uses quadratic
    problem solve to satisfy constraints.

    Attributes:
    C (float): Lambda parameter, determines slack allowance.
               Smaller = more slack, larger = less slack
    gamma (float): Gamma parameter, determines the influence of a single
                   training example. Controls the flexibility of the decision
                   boundary. Smaller = smoother boundary, larger = more complex
                   boundary. Utilized in RBF kernel.
    weights (np.ndarray): Stores the weights of the trained model
    bias (float): Stores the bias of the trained model
    """

    def __init__(self, C=10.0, gamma=0.1):
        # Gamma: Determines the influence of a single training example.
        # C(lambda): smaller = more slack, larger = less slack
        self.C = C
        self.gamma = gamma
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Change y from (0 or 1) to (-1 or 1)
        y[y == 0] = -1

        # Initilize some values to format the problem into cvxopt format
        m, n = X.shape
        y = y.reshape(-1,1) * 1.
        X_dash = y * X

        # Compute kernel matrix (different from tutorial: utilizes RBF)
        n_samples = X.shape[0]
        H = self._compute_kernel_matrix(X_dash) * 1.

        # Convert to cvxopt format
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        # Suppress CVXOPT output
        cvxopt_solvers.options['show_progress'] = False

        # Solve the quadratic programming problem
        solution = cvxopt_solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers (support vector coefficients)
        alphas = np.array(solution['x'])

        # Compute weights and bias
        w = ((y * alphas).T @ X).reshape(-1,1)
        S = (alphas > 1e-4).flatten() # get support vectors
        b = y[S] - np.dot(X[S], w)

        self.weights = w.flatten()
        self.bias = b[0]

    def predict(self, X):
        # Calculate z
        z = np.dot(X, self.weights) + self.bias

        # Decision threshold 0
        pred = np.sign(z)

        # Convert predictions from -1/1 to 0/1
        pred[pred == -1] = 0

        return pred

    def _compute_kernel_matrix(self, X):
        # Compute Radial Basis Function (RBF) kernel matrix
        n_samples = len(X)
        kernel_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i,j] = np.exp(-self.gamma *
                                            np.linalg.norm(X[i] - X[j])**2)

        return kernel_matrix


def SVM_test_params(X, y):
    """ Grid search parameters, testing with 10-fold cross validation
    returns: best performing C(lambda) and gamma from the grid search
    """

    # Test different lamba and gamma values
    gamma_range = np.logspace(-2, 2, 5)
    lambda_range = np.logspace(-2, 2, 5)

    # Keep track of best params and associated statistics
    best_gamma = -1
    best_lambda = -1
    best_acc = -1
    best_prec = -1
    best_rec = -1
    best_f1 = -1

    for g in gamma_range:
        for l in lambda_range:

            model = rbf_SVM(C = l, gamma = g)
            acc_list, prec_list, rec_list, f1_list = \
                        cross_val_10fold(model, X, y, False)

            acc = np.mean(acc_list)
            prec = np.mean(prec_list)
            rec = np.mean(rec_list)
            f1 = np.mean(f1_list)

            if acc > best_acc:
                best_gamma = g
                best_lambda = l
                best_acc = acc
                best_prec = prec
                best_rec = rec
                best_f1 = f1

    print(f'best gamma: {best_gamma}')
    print(f'best lambda: {best_lambda}')
    print(f'best acc: {best_acc}')
    print(f'best prec: {best_prec}')
    print(f'best rec: {best_rec}')
    print(f'best f1: {best_f1}')

    return best_lambda, best_gamma