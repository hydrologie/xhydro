# -*- coding: utf-8 -*-
"""

"""

import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .create_datasets import create_LSTM_dataset_catchment_vary
from tensorflow.keras.models import load_model

class training_generator(tf.keras.utils.Sequence):
    """
    Create a training generator to empty the GPU memory during training:
    """
    def __init__(self, x_set, x_set_static, x_set_q_stds, y_set, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.x_q_stds = x_set_q_stds
        self.y = y_set

        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return (np.ceil(len(self.y) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_x_static = self.x_static[inds]
        batch_x_q_stds = self.x_q_stds[inds]
        batch_y = self.y[inds]

        return [np.array(batch_x), np.array(batch_x_static)], \
            np.vstack((np.array(batch_y), np.array(batch_x_q_stds))).T

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class training_generator_tmp(tf.keras.utils.Sequence):
    """
    Create a training generator to empty the GPU memory during training:
    """
    def __init__(self, x_set, x_set_static, y_set, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.y = y_set

        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return (np.ceil(len(self.y) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_x_static = self.x_static[inds]
        batch_y = self.y[inds]

        return [np.array(batch_x), np.array(batch_x_static)], \
            np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class testing_generator(tf.keras.utils.Sequence):
    """
    Create a testing generator to empty the GPU memory during testing:
    """

    def __init__(self, x_set, x_set_static, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(self.x.shape[0] / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_static = self.x_static[
            idx * self.batch_size: (idx + 1) * self.batch_size
            ]
        return [np.array(batch_x), np.array(batch_x_static)], np.array(batch_x_static)


def keras_kge(data, y_pred):
    """
    This function computes the Kling-Gupta Efficiency (KGE) criterion
    Parameters:
        - y_true: Observed streamflow
        - y_pred: Simulated streamflow

    Return
        - kge: KGE criterion value
    """

    y_true = data[:, 0]
    y_pred = y_pred[:, 0]

    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')

    # Compute the dimensionless correlation coefficient
    r = (
            K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred))) /
            (K.sqrt(K.sum((y_true - K.mean(y_true)) ** 2)) *
             K.sqrt(K.sum((y_pred - K.mean(y_pred)) ** 2))
             )
    )

    # Compute the dimensionless bias ratio b (beta)
    b = (K.mean(y_pred) / K.mean(y_true))

    # Compute the dimensionless variability ratio g (gamma)
    g = (K.std(y_pred) / K.mean(y_pred)) / (K.std(y_true) / K.mean(y_true))

    # Compute the Kling-Gupta Efficiency (KGE) modified criterion
    kge = 1 - (1 - K.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2));

    return kge


def nse_loss(data, y_pred):
    """
    Parameters
    ----------
    data :
        Tensor of the target variable and the.
    y_pred :
        Tensor of the predicted variable.

    Returns
    -------
    scaled_loss : TYPE
        DESCRIPTION.

    """

    y_true = data[:, 0]
    q_stds = data[:, 1]
    y_pred = y_pred[:, 0]

    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')
    # q_stds = K.print_tensor(q_stds, message='q_stds = ')

    eps = float(0.1)
    squared_error = (y_pred - y_true) ** 2
    weights = 1 / (q_stds + eps) ** 2
    scaled_loss = (weights * squared_error)

    return scaled_loss


def define_LSTM_model_simple(window_size, n_dynamic_features, n_static_features, checkpoint_path='tmp.h5'):
    """
    """
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))
    x_in_static = tf.keras.layers.Input(shape=n_static_features)

    # LSTM 365 day
    x_365 = tf.keras.layers.LSTM(64, return_sequences=False)(x_in_365)
    x_365 = tf.keras.layers.Dropout(0.2)(x_365)

    # Dense statics
    x_static = tf.keras.layers.Dense(24, activation='relu')(x_in_static)
    x_static = tf.keras.layers.Dropout(0.2)(x_static)

    # Concatenate the model
    x = tf.keras.layers.Concatenate()([x_365, x_static])
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x_out = tf.keras.layers.Dense(1, activation='relu')(x)


    model_LSTM = tf.keras.models.Model([x_in_365, x_in_static], [x_out])
    model_LSTM.compile(loss=nse_loss, optimizer=tf.keras.optimizers.AdamW())

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_freq='epoch',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
            ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=15
            ),
        tf.keras.callbacks.LearningRateScheduler(
            step_decay
            )
    ]

    return model_LSTM, callback

def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch)/epochs_drop)
                                     )
    return lrate


def run_trained_model(arr_dynamic, arr_static, q_stds, window_size, w, idx_scenario, batch_size_val, watershed_areas,
                      name_of_saved_model, cleanNans):

    # Delete and reload the model to free the memory
    K.clear_session()
    model_LSTM = load_model(
        name_of_saved_model,
        compile=False,
        custom_objects={'loss': nse_loss}
    )

    # Training Database
    X, X_static, _, y = create_LSTM_dataset_catchment_vary(
        arr_dynamic=arr_dynamic[:, :, :],
        arr_static=arr_static,
        q_stds=q_stds,
        window_size=window_size,
        watershed_list=w[np.newaxis],
        idx=idx_scenario,
        cleanNans=cleanNans,
    )

    y_pred = model_LSTM.predict(
        testing_generator(
            X,
            X_static,
            batch_size=batch_size_val)
    )

    y_pred = np.squeeze(y_pred)

    # Rescale observed and simulated streamflow from mm/d to m^3/s
    drainage_area = watershed_areas[w]
    y_pred = y_pred * drainage_area / 86.4
    y = y * drainage_area / 86.4

    # Compute the Kling-Gupta Efficiency (KGE) for the current watershed
    kge = obj_fun_kge(y, y_pred)
    flows = np.array([y, y_pred])

    return kge, flows


def obj_fun_kge(Qobs, Qsim):
    """
    # This function computes the Kling-Gupta Efficiency (KGE) criterion
    :param Qobs: Observed streamflow
    :param Qsim: Simulated streamflow
    :return: kge: KGE criterion value
    """
    # Remove all nans from both observed and simulated streamflow
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]
    Qsim = Qsim[~ind_nan]

    # Compute the dimensionless correlation coefficient
    r = np.corrcoef(Qsim, Qobs)[0, 1]

    # Compute the dimensionless bias ratio b (beta)
    b = (np.mean(Qsim) / np.mean(Qobs))

    # Compute the dimensionless variability ratio g (gamma)
    g = (np.std(Qsim) / np.mean(Qsim)) / (np.std(Qobs) / np.mean(Qobs))

    # Compute the Kling-Gupta Efficiency (KGE) modified criterion
    kge = 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2);

    # In some cases, the KGE can return nan values which will force some
    # optimization algorithm to crash. Force the worst value possible instead.
    if np.isnan(kge):
        kge = -np.inf

    return kge
