import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import f1_score, r2_score, mean_squared_error

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

from metrics import score_regression
from preprocess import get_dataset, preprocess_all, dataset_to_X_y, SUBSET_FEATURES

dataset = get_dataset(shuffle=True, seed=42)
dataset = preprocess_all(dataset, scale=True)
X_train, y_train, X_validation, y_validation = dataset_to_X_y(dataset, "all")

def to_tf_dataset(X, y, batch_size):
    X = X.astype(np.float32)
    #y = y.astype(np.float32)
    data = tf.data.Dataset.from_tensor_slices((X,y)).batch(batch_size).prefetch(-1)
    return data

THRESHOLDS = [500, 1400, 5000, 10000]
_THRESHOLDS = [tf.float32.min, 0.] + THRESHOLDS + [tf.float32.max] 
_INSIDE = [-50, 250, 1000, 2500, 7500, 12500]
N_CLASSES = len(_THRESHOLDS) -1
def to_thresh_class(y):
    prev = _THRESHOLDS[0]
    ones = tf.ones_like(y)
    y_out = ones
    for idx, th in enumerate(_THRESHOLDS[1:]):
        y_out = tf.where(tf.logical_and(prev < y, y <= th), idx*ones, y_out)
        prev = th
    return y_out


def get_loss(y_true, y_pred):
    zeros = tf.zeros_like(y_pred)
    y_true_class = to_thresh_class(y_true)
    y_pred_class = to_thresh_class(y_pred)

    loss = tf.keras.losses.mse(y_true, y_pred)

    return tf.where(y_true_class != y_pred_class, loss, .1*loss)


class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return get_loss(y_true, y_pred)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, lr: lr ** (1.1 / (epoch+1))
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(N_CLASSES)
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

y_train_classes = to_thresh_class(y_train)
y_validation_classes = to_thresh_class(y_validation)

BATCH_SIZE = 512
train_dataset = to_tf_dataset(X_train, y_train_classes, BATCH_SIZE)
validation_dataset = to_tf_dataset(X_validation, y_validation_classes, BATCH_SIZE)

model.fit(
    train_dataset, 
    validation_data=validation_dataset,
    epochs=100,
    #callbacks=[lr_scheduler]
)


def get_class_thresh(y):
    a = np.array(_INSIDE)
    return a[y.astype(np.uint8)]

y_val_hat_classes = model.predict(validation_dataset).argmax(axis=-1)
y_val_hat = get_class_thresh(y_val_hat_classes)

y_train_hat_classes = model.predict(train_dataset).argmax(axis=-1)
y_train_hat = get_class_thresh(y_train_hat_classes)

val_score = score_regression(y_validation, y_val_hat)
train_score = score_regression(y_train, y_train_hat)


print(train_score)
print(val_score)