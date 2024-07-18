from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, labels = None):
    if labels is None:
        labels = np.unique(y_true)
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    num_ratings = len(labels)

    # Compute observed agreement
    observed_agreement = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    # Compute expected agreement
    hist_true = np.sum(conf_mat, axis = 1)
    hist_pred = np.sum(conf_mat, axis = 0)
    expected_agreement = np.dot(hist_true, hist_pred) / np.sum(conf_mat) ** 2

    # Compute quadratic weights
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = (i - j) ** 2

    # Compute kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    quadratic_weighted_kappa = 1 - (
                np.sum(weights * conf_mat) / np.sum(weights * hist_true[:, None] * hist_pred[None, :]))

    return quadratic_weighted_kappa


# Example usage:
y_true = [4, 3, 2, 2, 4, 1]
y_pred = [4, 3, 2, 1, 4, 2]

kappa = quadratic_weighted_kappa(y_true, y_pred)
print("Quadratic Weighted Kappa:", kappa)
