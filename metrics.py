import numpy as np
from sklearn.metrics import f1_score

def score_regression(y_true, y_pred):
    scores = [f1_score(y_true>th, y_pred>th) for th in [500, 1400, 5000, 10000]]
    return np.mean(scores)

def regression_scorer(estimator, X, y_true, **kwargs):
    y_hat = estimator.predict(X)
    score = score_regression(y_true, y_hat)
    return score
