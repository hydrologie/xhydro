"""LSTM model definition and tools for LSTM model training."""

import math

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model

from xhydro.modelling.obj_funcs import get_objective_function

from .create_datasets import create_lstm_dataset, create_lstm_dataset_local

__all__ = [
    "TestingGenerator",
    "TestingGeneratorLocal",
    "TrainingGenerator",
    "TrainingGeneratorLocal",
    "get_list_of_LSTM_models",
    "run_trained_model",
    "run_trained_model_local",
]


def get_list_of_LSTM_models(model_structure):
    """Create a training generator to manage the GPU memory during training.

    Parameters
    ----------
    model_structure : str
        The name of the LSTM model to use for training. Must correspond to one of the models present in LSTM_static.py.
        The "model_structure_dict" must be updated when new models are added.

    Returns
    -------
    function :
        Handle to the LSTM model function.
    """
    # Create a list of available models:
    try:
        model_structure_dict = {
            "simple_local_lstm": _simple_local_lstm,
            "simple_regional_lstm": _simple_regional_lstm,
            "dummy_local_lstm": _dummy_local_lstm,
            "dummy_regional_lstm": _dummy_regional_lstm,
        }
    except Exception:
        raise ValueError(
            "The LSTM model structure desired is not present in the available model dictionary."
        )

    model_handle = model_structure_dict[model_structure]

    return model_handle


class TrainingGenerator(tf.keras.utils.Sequence):
    """Create a training generator to manage the GPU memory during training.

    Parameters
    ----------
    x_set : np.ndarray
        Tensor of size [batch_size x window_size x n_dynamic_variables] that contains the batch of dynamic (i.e.
        timeseries) variables that will be used during training.
    x_set_static : np.ndarray
        Tensor of size [batch_size x n_static_variables] that contains the batch of static (i.e. catchment descriptors)
        variables that will be used during training.
    x_set_q_stds : np.ndarray
        Tensor of size [batch_size] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in x_set and x_set_static. Each data point could come from any catchment and this q_std
        variable helps scale the objective function.
    y_set : np.ndarray
        Tensor of size [batch_size] containing the target variable for the same time point as in x_set, x_set_static and
        x_set_q_stds. Usually the streamflow for the day associated to each of the training points.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.

    Returns
    -------
    self : An object containing the subset of the total data that is selected for this batch.
    """

    def __init__(self, x_set, x_set_static, x_set_q_stds, y_set, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.x_q_stds = x_set_q_stds
        self.y = y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        """Get total number of batches to generate"""
        return (np.ceil(len(self.y) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Get one of the batches by taking the 'batch_size' first elements from the list of remaining indices."""
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_x_static = self.x_static[inds]
        batch_x_q_stds = self.x_q_stds[inds]
        batch_y = self.y[inds]

        return [np.array(batch_x), np.array(batch_x_static)], np.vstack(
            (np.array(batch_y), np.array(batch_x_q_stds))
        ).T

    def on_epoch_end(self):
        """Shuffle the dataset before the next batch sampling to ensure randomness, helping convergence."""
        np.random.shuffle(self.indices)


class TrainingGeneratorLocal(tf.keras.utils.Sequence):
    """Create a training generator to manage the GPU memory during training.

    Parameters
    ----------
    x_set : np.ndarray
        Tensor of size [batch_size x window_size x n_dynamic_variables] that contains the batch of dynamic (i.e.
        timeseries) variables that will be used during training.
    y_set : np.ndarray
        Tensor of size [batch_size] containing the target variable for the same time point as in x_set, x_set_static and
        x_set_q_stds. Usually the streamflow for the day associated to each of the training points.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.

    Returns
    -------
    self : An object containing the subset of the total data that is selected for this batch.
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        """Get total number of batches to generate"""
        return (np.ceil(len(self.y) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Get one of the batches by taking the 'batch_size' first elements from the list of remaining indices."""
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]

        return [np.array(batch_x)], [np.array(batch_y)]

    def on_epoch_end(self):
        """Shuffle the dataset before the next batch sampling to ensure randomness, helping convergence."""
        np.random.shuffle(self.indices)


class TestingGenerator(tf.keras.utils.Sequence):
    """Create a testing generator to manage the GPU memory during training.

    Parameters
    ----------
    x_set : np.ndarray
        Tensor of size [batch_size x window_size x n_dynamic_variables] that contains the batch of dynamic (i.e.
        timeseries) variables that will be used during training.
    x_set_static : np.ndarray
        Tensor of size [batch_size x n_static_variables] that contains the batch of static (i.e. catchment descriptors)
        variables that will be used during training.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.

    Returns
    -------
    self : An object containing the subset of the total data that is selected for this batch.
    """

    def __init__(self, x_set, x_set_static, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.batch_size = batch_size

    def __len__(self):
        """Get total number of batches to generate"""
        return (np.ceil(self.x.shape[0] / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Get one of the batches by taking the 'batch_size' first elements from the list of remaining indices."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x_static = self.x_static[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        return [np.array(batch_x), np.array(batch_x_static)], np.array(batch_x_static)


class TestingGeneratorLocal(tf.keras.utils.Sequence):
    """Create a testing generator to manage the GPU memory during training.

    Parameters
    ----------
    x_set : np.ndarray
        Tensor of size [batch_size x window_size x n_dynamic_variables] that contains the batch of dynamic (i.e.
        timeseries) variables that will be used during training.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.

    Returns
    -------
    self : An object containing the subset of the total data that is selected for this batch.
    """

    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        """Get total number of batches to generate"""
        return (np.ceil(self.x.shape[0] / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        """Get one of the batches by taking the 'batch_size' first elements from the list of remaining indices."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        return [np.array(batch_x)]


def _kge_loss(data, y_pred):
    """Compute the Kling-Gupta Efficiency (KGE) criterion under Keras for Tensorflow training.

    Needs to be separate from the regular objective function calculations because it uses Keras/tensor arithmetic for
    GPU.

    Parameters
    ----------
    data : np.ndarray
        Observed streamflow used as target for the LSTM training.
    y_pred : np.ndarray
        Simulated streamflow generated by the LSTM during training.

    Returns
    -------
    kge
        KGE metric computed using Keras tools.
    """
    y_true = data[:, 0]
    y_pred = y_pred[:, 0]

    # Compute the dimensionless correlation coefficient
    r = k.sum((y_true - k.mean(y_true)) * (y_pred - k.mean(y_pred))) / (
        k.sqrt(k.sum((y_true - k.mean(y_true)) ** 2))
        * k.sqrt(k.sum((y_pred - k.mean(y_pred)) ** 2))
    )

    # Compute the dimensionless bias ratio b (beta)
    b = k.mean(y_pred) / k.mean(y_true)

    # Compute the dimensionless variability ratio g (gamma)
    g = (k.std(y_pred) / k.mean(y_pred)) / (k.std(y_true) / k.mean(y_true))

    # Compute the Kling-Gupta Efficiency (KGE) modified criterion
    kge = 1 - (1 - k.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))

    return kge


def _nse_scaled_loss(data, y_pred):
    """Compute the modified NSE loss for regional training.

    Applies a random noise element for robustness, as well scales the flows according to their standard deviations to be
    able to compute the nse with data from multiple catchments all scaled and normalized.

    Parameters
    ----------
    data : np.ndarray
        Tensor of the target variable (column 1) and observed streamflow standard deviations (column 2).
    y_pred : np.ndarray
        Tensor of the predicted variable from the LSTM model.

    Returns
    -------
    scaled_loss
        Scaled modified NSE metric for training on regional models.
    """
    y_true = data[:, 0]
    q_stds = data[:, 1]
    y_pred = y_pred[:, 0]

    eps = float(0.1)
    squared_error = (y_pred - y_true) ** 2
    weights = 1 / (q_stds + eps) ** 2
    scaled_loss = weights * squared_error

    return scaled_loss


def _simple_regional_lstm(
    window_size: int,
    n_dynamic_features: int,
    n_static_features: int,
    training_func: str,
    checkpoint_path: str = "tmp.h5",
):
    """Define the LSTM model structure and hyperparameters to use. Must be updated by users to modify model structures.

    Parameters
    ----------
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    n_dynamic_features : int
        Number of dynamic (i.e. time-series) variables that are used for training and simulating the model.
    n_static_features : int
        Number of static (i.e. catchment descriptor) variables that are used to define the regional LSTM model and allow
        it to modulate simulations according to catchment properties.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    checkpoint_path : str
        Sting containing the path of the file where the trained model will be saved.

    Returns
    -------
    model_lstm : Tensorflow model
        The tensorflow model that will be trained, along with all of its default hyperparameters and training options.
    callback : list
        List of tensorflow objects that allow performing operations after each training epoch.
    """
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))
    x_in_static = tf.keras.layers.Input(shape=n_static_features)

    # LSTM 365 day
    x_365 = tf.keras.layers.LSTM(128, return_sequences=True)(x_in_365)
    x_365 = tf.keras.layers.LSTM(128, return_sequences=False)(x_365)
    x_365 = tf.keras.layers.Dropout(0.1)(x_365)

    # Dense statics
    x_static = tf.keras.layers.Dense(24, activation="relu")(x_in_static)
    x_static = tf.keras.layers.Dropout(0.2)(x_static)

    # Concatenate the model
    x = tf.keras.layers.Concatenate()([x_365, x_static])
    x = tf.keras.layers.Dense(12, activation="relu")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x_out = tf.keras.layers.Dense(1, activation="relu")(x)

    model_lstm = tf.keras.models.Model([x_in_365, x_in_static], [x_out])
    if training_func == "nse_scaled":
        model_lstm.compile(
            loss=_nse_scaled_loss, optimizer=tf.keras.optimizers.AdamW(clipnorm=0.1)
        )
    elif training_func == "kge":
        model_lstm.compile(
            loss=_kge_loss, optimizer=tf.keras.optimizers.AdamW(clipnorm=0.1)
        )

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_freq="epoch",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=15
        ),
        tf.keras.callbacks.LearningRateScheduler(_step_decay),
    ]

    return model_lstm, callback


def _simple_local_lstm(
    window_size: int,
    n_dynamic_features: int,
    training_func: str,
    checkpoint_path: str = "tmp.h5",
):
    """Define the local LSTM model structure and hyperparameters to use.

    Must be updated by users to modify model structures.

    Parameters
    ----------
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    n_dynamic_features : int
        Number of dynamic (i.e. time-series) variables that are used for training and simulating the model.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    checkpoint_path : str
        Sting containing the path of the file where the trained model will be saved.

    Returns
    -------
    model_lstm : Tensorflow model
        The tensorflow model that will be trained, along with all of its default hyperparameters and training options.
    callback : list
        List of tensorflow objects that allow performing operations after each training epoch.
    """
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))

    # Single LSTM layer
    x_365 = tf.keras.layers.LSTM(32, return_sequences=True)(x_in_365)
    x_365 = tf.keras.layers.LSTM(32, return_sequences=False)(x_365)
    x_365 = tf.keras.layers.Dropout(0.1)(x_365)  # Add dropout layer for robustness

    # Pass to a simple Dense layer with relu activation and LeakyReLU activation
    x = tf.keras.layers.Dense(12, activation="relu")(x_365)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # pass to a 1-unit dense layer representing output flow.
    x_out = tf.keras.layers.Dense(1, activation="relu")(x)

    # Build model
    model_lstm = tf.keras.models.Model([x_in_365], [x_out])

    if training_func == "kge":
        model_lstm.compile(
            loss=_kge_loss, optimizer=tf.keras.optimizers.AdamW(clipnorm=0.1)
        )
    else:
        raise ValueError("training_func can only be kge for the local training model.")

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_freq="epoch",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=15
        ),
        tf.keras.callbacks.LearningRateScheduler(_step_decay),
    ]

    return model_lstm, callback


def _dummy_regional_lstm(
    window_size: int,
    n_dynamic_features: int,
    n_static_features: int,
    training_func: str,
    checkpoint_path: str = "tmp.h5",
):
    """Define the LSTM model structure and hyperparameters to use. Must be updated by users to modify model structures.

    Parameters
    ----------
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    n_dynamic_features : int
        Number of dynamic (i.e. time-series) variables that are used for training and simulating the model.
    n_static_features : int
        Number of static (i.e. catchment descriptor) variables that are used to define the regional LSTM model and allow
        it to modulate simulations according to catchment properties.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    checkpoint_path : str
        Sting containing the path of the file where the trained model will be saved.

    Returns
    -------
    model_lstm : Tensorflow model
        The tensorflow model that will be trained, along with all of its default hyperparameters and training options.
    callback : list
        List of tensorflow objects that allow performing operations after each training epoch.
    """
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))
    x_in_static = tf.keras.layers.Input(shape=n_static_features)

    # LSTM 365 day
    x_365 = tf.keras.layers.LSTM(64, return_sequences=False)(x_in_365)
    x_365 = tf.keras.layers.Dropout(0.2)(x_365)

    # Dense statics
    x_static = tf.keras.layers.Dense(24, activation="relu")(x_in_static)
    x_static = tf.keras.layers.Dropout(0.2)(x_static)

    # Concatenate the model
    x = tf.keras.layers.Concatenate()([x_365, x_static])
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x_out = tf.keras.layers.Dense(1, activation="relu")(x)

    model_lstm = tf.keras.models.Model([x_in_365, x_in_static], [x_out])
    if training_func == "nse_scaled":
        model_lstm.compile(loss=_nse_scaled_loss, optimizer=tf.keras.optimizers.AdamW())
    elif training_func == "kge":
        model_lstm.compile(loss=_kge_loss, optimizer=tf.keras.optimizers.AdamW())

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_freq="epoch",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=15
        ),
        tf.keras.callbacks.LearningRateScheduler(_step_decay),
    ]

    return model_lstm, callback


def _dummy_local_lstm(
    window_size: int,
    n_dynamic_features: int,
    training_func: str,
    checkpoint_path: str = "tmp.h5",
):
    """Define the local LSTM model structure and hyperparameters to use.

    Must be updated by users to modify model structures.

    Parameters
    ----------
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    n_dynamic_features : int
        Number of dynamic (i.e. time-series) variables that are used for training and simulating the model.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    checkpoint_path : str
        Sting containing the path of the file where the trained model will be saved.

    Returns
    -------
    model_lstm : Tensorflow model
        The tensorflow model that will be trained, along with all of its default hyperparameters and training options.
    callback : list
        List of tensorflow objects that allow performing operations after each training epoch.
    """
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))

    # LSTM 365 day
    x_365 = tf.keras.layers.LSTM(64, return_sequences=False)(
        x_in_365
    )  # Single LSTM layer
    x_365 = tf.keras.layers.Dropout(0.2)(x_365)  # Add dropout layer for robustness
    x = tf.keras.layers.Dense(8, activation="relu")(
        x_365
    )  # Pass to a simple Dense layer with relu activation
    x_out = tf.keras.layers.Dense(1, activation="relu")(
        x
    )  # pass to a 1-unit dense layer representing output flow.

    model_lstm = tf.keras.models.Model([x_in_365], [x_out])

    if training_func == "kge":
        model_lstm.compile(loss=_kge_loss, optimizer=tf.keras.optimizers.AdamW())
    else:
        raise ValueError("training_func can only be kge for the local training model.")

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_freq="epoch",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=15
        ),
        tf.keras.callbacks.LearningRateScheduler(_step_decay),
    ]

    return model_lstm, callback


def _step_decay(epoch: int):
    """Callback for learning rate tuning during LSTM model training.

    Parameters
    ----------
    epoch : int
        Current epoch number during training. Used to adapt the learning rate after a certain number of epochs to aid in
        model training convergence.

    Returns
    -------
    lrate
        The updated learning rate.
    """
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


def run_trained_model(
    arr_dynamic: np.ndarray,
    arr_static: np.ndarray,
    q_stds: np.ndarray,
    window_size: int,
    w: int,
    idx_scenario: np.ndarray,
    batch_size: int,
    watershed_areas: np.ndarray,
    name_of_saved_model: str,
    remove_nans: bool,
):
    """Run the trained regional LSTM model on a single catchment from a larger set.

    Parameters
    ----------
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.ndarray
        Tensor of size [time_steps x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.ndarray
        Tensor of size [time_steps] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    w : int
        Number of the watershed from the list of catchments that will be simulated.
    idx_scenario : np.ndarray
        2-element array of indices of the beginning and end of the desired period for which the LSTM model should be
        simulated.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    watershed_areas : np.ndarray
        Area of the watershed, in square kilometers, as taken from the training dataset initial input ncfile.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    remove_nans : bool
        Remove the periods for which observed streamflow is NaN for both observed and simulated flows.

    Returns
    -------
    kge : float
        KGE value between observed and simulated flows computed for the watershed of interest and for a specified
        period.
    flows : np.ndarray
        Observed and simulated streamflows computed for the watershed of interest and for a specified period.
    """
    # Delete and reload the model to free the memory
    k.clear_session()
    model_lstm = load_model(
        name_of_saved_model, compile=False, custom_objects={"loss": _nse_scaled_loss}
    )

    # Training Database
    x, x_static, _, y = create_lstm_dataset(
        arr_dynamic=arr_dynamic[:, :, :],
        arr_static=arr_static,
        q_stds=q_stds,
        window_size=window_size,
        watershed_list=w[np.newaxis],
        idx=idx_scenario,
        remove_nans=remove_nans,
    )

    y_pred = model_lstm.predict(TestingGenerator(x, x_static, batch_size=batch_size))
    y_pred = np.squeeze(y_pred)

    # Rescale observed and simulated streamflow from mm/d to m^3/s
    drainage_area = watershed_areas[w]

    y_pred = y_pred * drainage_area / 86.4
    y = y * drainage_area / 86.4

    # Compute the Kling-Gupta Efficiency (KGE) for the current watershed
    kge = get_objective_function(qobs=y, qsim=y_pred, obj_func="kge")
    flows = np.array([y, y_pred])

    return kge, flows


def run_trained_model_local(
    arr_dynamic: np.ndarray,
    window_size: int,
    idx_scenario: np.ndarray,
    batch_size: int,
    name_of_saved_model: str,
    remove_nans: bool,
):
    """Run the trained regional LSTM model on a single catchment from a larger set.

    Parameters
    ----------
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    idx_scenario : np.ndarray
        2-element array of indices of the beginning and end of the desired period for which the LSTM model should be
        simulated.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    remove_nans : bool
        Remove the periods for which observed streamflow is NaN for both observed and simulated flows.

    Returns
    -------
    kge : float
        KGE value between observed and simulated flows computed for the watershed of interest and for a specified
        period.
    flows : np.ndarray
        Observed and simulated streamflows computed for the watershed of interest and for a specified period.
    """
    # Delete and reload the model to free the memory
    k.clear_session()
    model_lstm = load_model(
        name_of_saved_model, compile=False, custom_objects={"loss": _kge_loss}
    )

    # Training Database
    x, y = create_lstm_dataset_local(
        arr_dynamic=arr_dynamic,
        window_size=window_size,
        idx=idx_scenario,
        remove_nans=remove_nans,
    )

    y_pred = model_lstm.predict(TestingGeneratorLocal(x, batch_size=batch_size))
    y_pred = np.squeeze(y_pred)

    # Compute the Kling-Gupta Efficiency (KGE) for the watershed
    kge = get_objective_function(qobs=y, qsim=y_pred, obj_func="kge")
    flows = np.array([y, y_pred])

    return kge, flows
