import os
from os.path import join

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import GridSearchCV
#import dask_ml.model_selection as dcv
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from preprocess import get_dataset, preprocess_all, dataset_to_X_y, SUBSET_FEATURES
from metrics import score_regression, regression_scorer
from utils import output_dir

dataset = get_dataset(seed=42)
dataset = preprocess_all(dataset=dataset, subset=SUBSET_FEATURES)
X_train, y_train, X_val, y_val = dataset_to_X_y(dataset, keys="all", datatype="df")

reg = Ridge()
hparams = dict(
    alpha=np.logspace(.001, .1, 20),
    fit_intercept=[True, False],
    normalize=[False, True]
)

grid = GridSearchCV(reg, hparams, scoring=regression_scorer)
grid.fit(X_train, y_train)

print(f"Best parameters found : ",grid.best_params_)

y_val_hat = grid.predict(X_val)
val_score = score_regression(y_val, y_val_hat)

print(f"Val score : {val_score:.3f}")
