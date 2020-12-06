import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import f1_score, r2_score, mean_squared_error

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
import tensorflow.keras.backend as K

from metrics import score_regression
from preprocess import get_dataset, preprocess_all, dataset_to_X_y, SUBSET_FEATURES



BATCH_SIZE = 512
EPOCHS = 200

dataset = get_dataset(shuffle=True, seed=42)
dataset = preprocess_all(dataset, scale=True)
X_train, y_train, X_validation, y_validation = dataset_to_X_y(dataset, "all")

def to_tf_dataset(X, y, batch_size):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    data = tf.data.Dataset.from_tensor_slices((X,y)).batch(batch_size).prefetch(-1)
    return data

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, lr: lr * .995
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
    #tfp.layers.DistributionLambda(lambda t:tfd.Normal(loc=t, scale=1)),
])


train_dataset = to_tf_dataset(X_train, y_train, BATCH_SIZE)
validation_dataset = to_tf_dataset(X_validation, y_validation, BATCH_SIZE)

# Do inference.
negloglik = lambda y, p_y: -p_y.log_prob(y)
model.compile(
    loss='mae',
    optimizer=tf.keras.optimizers.Adam(.1),
    #metrics=["accuracy"]
)


model.fit(
    train_dataset, 
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[lr_scheduler]
)


y_val_hat = model.predict(validation_dataset)
y_train_hat = model.predict(train_dataset)

val_score = score_regression(y_validation, y_val_hat)
train_score = score_regression(y_train, y_train_hat)


print(train_score)
print(val_score)