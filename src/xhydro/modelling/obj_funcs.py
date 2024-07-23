# Created on Tue Dec 12 21:55:25 2023
# @author: Richard Arsenault
"""Objective function package for xhydro, for calibration and model evaluation.

This package provides a flexible suite of popular objective function metrics in
hydrological modelling and hydrological model calibration. The main function
'get_objective_function' returns the value of the desired objective function
while allowing users to customize many aspects:

    1-  Select the objective function to run;
    2-  Allow providing a mask to remove certain elements from the objective
    function calculation (e.g. for odd/even year calibration, or calibration
    on high or low flows only, or any custom setup).
    3-  Apply a transformation on the flows to modify the behaviour of the
    objective function calculation (e.g taking the log, inverse or square
    root transform of the flows before computing the objective function).

This function also contains some tools and inputs reserved for the calibration
toolbox, such as the ability to take the negative of the objective function to
maximize instead of minimize a metric according to the needs of the optimizing
algorithm.
"""
from typing import Optional

# Import packages
import numpy as np
import xarray as xr

__all__ = ["get_objective_function", "transform_flows"]


def get_objective_function(
    qobs: np.array,
    qsim: np.array,
    obj_func: str = "rmse",
    take_negative: bool = False,
    mask: Optional[np.array] = None,
    transform: Optional[str] = None,
    epsilon: Optional[float] = None,
):
    """Entrypoint function for the objective function calculation.

    More can be added by adding the function to this file and adding the option
    in this function.

    Notes
    -----
    All data corresponding to NaN values in the observation set are removed from the calculation.
    If a mask is passed, it must be the same size as the qsim and qobs vectors.
    If any NaNs are present in the qobs dataset, all corresponding data in the qobs, qsim and mask
    will be removed prior to passing to the processing function.

    Parameters
    ----------
    qobs : array_like
        Vector containing the Observed streamflow to be used in the objective
        function calculation. It is the target to attain.

    qsim : array_like
        Vector containing the Simulated streamflow as generated by the hydrological model.
        It is modified by changing parameters and resumulating the hydrological model.

    obj_func : str
        String representing the objective function to use in the calibration.
        Options must be one of the accepted objective functions:

            - "abs_bias" : Absolute value of the "bias" metric
            - "abs_pbias": Absolute value of the "pbias" metric
            - "abs_volume_error" : Absolute value of the volume_error metric
            - "agreement_index": Index of agreement
            - "bias" : Bias metric
            - "correlation_coeff": Correlation coefficient
            - "kge" : Kling Gupta Efficiency metric (2009 version)
            - "kge_mod" : Kling Gupta Efficiency metric (2012 version)
            - "mae": Mean Absolute Error metric
            - "mare": Mean Absolute Relative Error metric
            - "mse" : Mean Square Error metric
            - "nse": Nash-Sutcliffe Efficiency metric
            - "pbias" : Percent bias (relative bias)
            - "r2" : r-squared, i.e. square of correlation_coeff.
            - "rmse" : Root Mean Square Error
            - "rrmse" : Relative Root Mean Square Error (RMSE-to-mean ratio)
            - "rsr" : Ratio of RMSE to standard deviation.
            - "volume_error": Total volume error over the period.

        The default is 'rmse'.

    take_negative : bool
        Used to force the objective function to be multiplied by minus one (-1)
        such that it is possible to maximize it if the optimizer is a minimizer
        and vice-versa. Should always be set to False unless required by an
        optimization setup, which is handled internally and transparently to
        the user. The default is False.

    mask : array_like
        Array of 0 or 1 on which the objective function should be applied.
        Values of 1 indicate that the value is included in the calculation, and
        values of 0 indicate that the value is excluded and will have no impact
        on the objective function calculation. This can be useful for specific
        optimization strategies such as odd/even year calibration, seasonal
        calibration or calibration based on high/low flows. The default is None
        and all data are preserved.

    transform : str
        Indicates the type of transformation required. Can be one of the
        following values:

            - "sqrt" : Square root transformation of the flows [sqrt(Q)]
            - "log" : Logarithmic transformation of the flows [log(Q)]
            - "inv" : Inverse transformation of the flows [1/Q]

        The default value is "None", by which no transformation is performed.

    epsilon : float
        Indicates the perturbation to add to the flow time series during a
        transformation to avoid division by zero and logarithmic transformation.
        The perturbation is equal to:

            perturbation = epsilon * mean(qobs)

        The default value is 0.01.

    Returns
    -------
    float
        Value of the selected objective function (obj_fun).
    """
    # List of available objective functions
    obj_func_dict = {
        "abs_bias": _abs_bias,
        "abs_pbias": _abs_pbias,
        "abs_volume_error": _abs_volume_error,
        "agreement_index": _agreement_index,
        "bias": _bias,
        "correlation_coeff": _correlation_coeff,
        "kge": _kge,
        "kge_mod": _kge_mod,
        "mae": _mae,
        "mare": _mare,
        "mse": _mse,
        "nse": _nse,
        "pbias": _pbias,
        "r2": _r2,
        "rmse": _rmse,
        "rrmse": _rrmse,
        "rsr": _rsr,
        "volume_error": _volume_error,
    }

    # If we got a dataset, change to np.array
    # FIXME: Implement a more flexible method
    if isinstance(qsim, xr.Dataset):
        qsim = qsim["streamflow"]

    if isinstance(qobs, xr.Dataset):
        qobs = qobs["qobs"]

    # Basic error checking
    if qobs.shape[0] != qsim.shape[0]:
        raise ValueError("Observed and Simulated flows are not of the same size.")

    # Check mask size and contents
    if mask is not None:
        # Size
        if qobs.shape[0] != mask.shape[0]:
            raise ValueError("Mask is not of the same size as the streamflow data.")

        # All zero or one?
        if not np.setdiff1d(np.unique(mask), np.array([0, 1])).size == 0:
            raise ValueError("Mask contains values other than 0 or 1. Please modify.")

    # Check that the objective function is in the list of available methods
    if obj_func not in obj_func_dict:
        raise ValueError(
            "Selected objective function is currently unavailable. "
            + "Consider contributing to our project at: "
            + "github.com/hydrologie/xhydro"
        )

    # Ensure there are no NaNs in the dataset
    if mask is not None:
        mask = mask[~np.isnan(qobs)]
    qsim = qsim[~np.isnan(qobs)]
    qobs = qobs[~np.isnan(qobs)]

    # Apply mask before transform
    if mask is not None:
        qsim = qsim[mask == 1]
        qobs = qobs[mask == 1]
        mask = mask[mask == 1]

    # Transform data if needed
    if transform is not None:
        qsim, qobs = transform_flows(qsim, qobs, transform, epsilon)

    # Compute objective function by switching to the correct algorithm. Ensure
    # that the function name is the same as the obj_func tag or this will fail.
    function_call = obj_func_dict[obj_func]
    obj_fun_val = function_call(qsim, qobs)

    # Take the negative value of the Objective Function to return to the
    # optimizer.
    if take_negative:
        obj_fun_val = obj_fun_val * -1

    return obj_fun_val


def _get_objfun_minimize_or_maximize(obj_func: str):
    """Check whether the objective function needs to be maximized or minimized.

    Returns a boolean value, where True means it should be maximized, and False
    means that it should be minimized. Objective functions other than those
    programmed here will raise an error.

    Parameters
    ----------
    obj_func : str
        Label of the desired objective function.

    Returns
    -------
    bool
        Indicator of if the objective function needs to be maximized.
    """
    # Define metrics that need to be maximized:
    if obj_func in [
        "agreement_index",
        "correlation_coeff",
        "kge",
        "kge_mod",
        "nse",
        "r2",
    ]:
        maximize = True

    # Define the metrics that need to be minimized:
    elif obj_func in [
        "abs_bias",
        "abs_pbias",
        "abs_volume_error",
        "mae",
        "mare",
        "mse",
        "rmse",
        "rrmse",
        "rsr",
    ]:
        maximize = False

    # Check for the metrics that exist but cannot be used for optimization
    elif obj_func in ["bias", "pbias", "volume_error"]:
        raise ValueError(
            "The bias, pbias and volume_error metrics cannot be minimized or maximized. \
                 Please use the abs_bias, abs_pbias and abs_volume_error instead."
        )
    else:
        raise NotImplementedError("The objective function is unknown.")

    return maximize


def _get_optimizer_minimize_or_maximize(algorithm: str):
    """Find the direction in which the optimizer searches.

    Some optimizers try to maximize the objective function value, and others
    try to minimize it. Since our objective functions include some that need to
    be maximized and others minimized, it is imperative to ensure that the
    optimizer/objective-function pair work in tandem.

    Parameters
    ----------
    algorithm : str
        Name of the optimizing algorithm.

    Returns
    -------
    bool
        Indicator of the direction of the optimizer search, True to maximize.
    """
    # Define metrics that need to be maximized:
    if algorithm in [
        "DDS",
    ]:
        maximize = True

    # Define the metrics that need to be minimized:
    elif algorithm in [
        "SCEUA",
    ]:
        maximize = False

    # Any other optimizer at this date
    else:
        raise NotImplementedError("The optimization algorithm is unknown.")

    return maximize


def transform_flows(
    qsim: np.array,
    qobs: np.array,
    transform: Optional[str] = None,
    epsilon: float = 0.01,
):
    """Transform flows before computing the objective function.

    It is used to transform flows such that the objective function is computed
    on a transformed flow metric rather than on the original units of flow
    (ex: inverse, log-transformed, square-root)

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    transform : str, optional
        Indicates the type of transformation required. Can be one of the
        following values:

            - "sqrt" : Square root transformation of the flows [sqrt(Q)]
            - "log" : Logarithmic transformation of the flows [log(Q)]
            - "inv" : Inverse transformation of the flows [1/Q]

        The default value is "None", by which no transformation is performed.

    epsilon : float
        Indicates the perturbation to add to the flow time series during a
        transformation to avoid division by zero and logarithmic transformation.
        The perturbation is equal to:

            perturbation = epsilon * mean(qobs)

        The default value is 0.01.

    Returns
    -------
    qsim : array_like
        Transformed simulated flow according to user request.
    qobs : array_like
        Transformed observed flow according to user request.
    """
    # Quick check
    if transform is not None:
        if transform not in ["log", "inv", "sqrt"]:
            raise NotImplementedError("Flow transformation method not recognized.")

    # Transform the flow series if required
    if transform == "log":  # log transformation
        epsilon = epsilon * np.nanmean(qobs)
        qobs, qsim = np.log(qobs + epsilon), np.log(qsim + epsilon)

    elif transform == "inv":  # inverse transformation
        epsilon = epsilon * np.nanmean(qobs)
        qobs, qsim = 1.0 / (qobs + epsilon), 1.0 / (qsim + epsilon)

    elif transform == "sqrt":  # square root transformation
        qobs, qsim = np.sqrt(qobs), np.sqrt(qsim)

    # Return the flows after transformation (or original if no transform)
    return qsim, qobs


"""
BEGIN OBJECTIVE FUNCTIONS DEFINITIONS
"""


def _abs_bias(qsim: np.array, qobs: np.array):
    """Absolute bias metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Absolute value of the "bias" metric. This metric is useful when calibrating
        on the bias, because bias should aim to be 0 but can take large positive
        or negative values. Taking the absolute value of the bias will let the
        optimizer minimize the value to zero.

    Notes
    -----
    The abs_bias should be MINIMIZED.
    """
    return np.abs(_bias(qsim, qobs))


def _abs_pbias(qsim: np.array, qobs: np.array):
    """Absolute pbias metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The absolute value of the "pbias" metric. This metric is useful when
        calibrating on the pbias, because pbias should aim to be 0 but can take
        large positive or negative values. Taking the absolute value of the
        pbias will let the optimizer minimize the value to zero.

    Notes
    -----
    The abs_pbias should be MINIMIZED.
    """
    return np.abs(_pbias(qsim, qobs))


def _abs_volume_error(qsim: np.array, qobs: np.array):
    """Absolute value of the volume error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The absolute value of the "volume_error" metric. This metric is useful
        when calibrating on the volume_error, because volume_error should aim
        to be 0 but can take large positive or negative values. Taking the
        absolute value of the volume_error will let the optimizer minimize the
        value to zero.

    Notes
    -----
    The abs_volume_error should be MINIMIZED.
    """
    return np.abs(_volume_error(qsim, qobs))


def _agreement_index(qsim: np.array, qobs: np.array):
    """Index of agreement metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The agreement index of Willmott (1981). Varies between 0 and 1.

    Notes
    -----
    The Agreement index should be MAXIMIZED.
    """
    # Decompose into clearer chunks
    a = np.sum((qobs - qsim) ** 2)
    b = np.abs(qsim - np.mean(qobs)) + np.abs(qobs - np.mean(qobs))
    c = np.sum(b**2)

    return 1 - (a / c)


def _bias(qsim: np.array, qobs: np.array):
    """The bias metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The bias in the simulation. Can be negative or positive and gives the
        average error between the observed and simulated flows. This
        interpretation uses the definition that a positive bias value means
        that the simulation overestimates the true value (as opposed to many
        other sources on bias calculations that use the contrary interpretation).

    Notes
    -----
    BIAS SHOULD AIM TO BE ZERO AND SHOULD NOT BE USED FOR CALIBRATION. FOR
    CALIBRATION, USE "abs_bias" TO TAKE THE ABSOLUTE VALUE.
    """
    return np.mean(qsim - qobs)


def _correlation_coeff(qsim: np.array, qobs: np.array):
    """Correlation coefficient metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The correlation coefficient.

    Notes
    -----
    The correlation_coeff should be MAXIMIZED.
    """
    return np.corrcoef(qobs, qsim)[0, 1]


def _kge(qsim: np.array, qobs: np.array):
    """Kling-Gupta efficiency metric (2009 version).

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The Kling-Gupta Efficiency (KGE) metric of 2009. It can take values
        from -inf to 1 (best case).

    Notes
    -----
    The KGE should be MAXIMIZED.
    """
    # This pops up a lot, precalculate.
    qsim_mean = np.mean(qsim)
    qobs_mean = np.mean(qobs)

    # Calculate the components of KGE
    r_num = np.sum((qsim - qsim_mean) * (qobs - qobs_mean))
    r_den = np.sqrt(np.sum((qsim - qsim_mean) ** 2) * np.sum((qobs - qobs_mean) ** 2))
    r = r_num / r_den
    a = np.std(qsim) / np.std(qobs)
    b = np.sum(qsim) / np.sum(qobs)

    # Calculate the KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)

    return kge


def _kge_mod(qsim: np.array, qobs: np.array):
    """Kling-Gupta efficiency metric (2012 version).

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The modified Kling-Gupta Efficiency (KGE) metric of 2012. It can take
        values from -inf to 1 (best case).

    Notes
    -----
    The kge_mod should be MAXIMIZED.
    """
    # These pop up a lot, precalculate
    qsim_mean = np.mean(qsim)
    qobs_mean = np.mean(qobs)

    # Calc KGE components
    r_num = np.sum((qsim - qsim_mean) * (qobs - qobs_mean))
    r_den = np.sqrt(np.sum((qsim - qsim_mean) ** 2) * np.sum((qobs - qobs_mean) ** 2))
    r = r_num / r_den
    g = (np.std(qsim) / qsim_mean) / (np.std(qobs) / qobs_mean)
    b = np.mean(qsim) / np.mean(qobs)

    # Calc the modified KGE metric
    kge_mod = 1 - np.sqrt((r - 1) ** 2 + (g - 1) ** 2 + (b - 1) ** 2)

    return kge_mod


def _mae(qsim: np.array, qobs: np.array):
    """Mean absolute error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Mean Absolute Error. It can be interpreted as the average error
        (absolute) between observations and simulations for any time step.

    Notes
    -----
    The mae should be MINIMIZED.
    """
    return np.mean(np.abs(qsim - qobs))


def _mare(qsim: np.array, qobs: np.array):
    """Mean absolute relative error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Mean Absolute Relative Error. For streamflow, where qobs is always zero
        or positive, the MARE is always positive.

    Notes
    -----
    The mare should be MINIMIZED.
    """
    return np.sum(np.abs(qobs - qsim)) / np.sum(qobs)


def _mse(qsim: np.array, qobs: np.array):
    """Mean square error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Mean Square Error. It is the sum of squared errors for each day divided
        by the total number of days. Units are thus squared units, and the best
        possible value is 0.

    Notes
    -----
    The mse should be MINIMIZED.
    """
    return np.mean((qobs - qsim) ** 2)


def _nse(qsim: np.array, qobs: np.array):
    """Nash-Sutcliffe efficiency metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency (NSE) metric. It can take values from -inf to
        1, with 0 being as good as using the mean observed flow as the estimator.

    Notes
    -----
    The nse should be MAXIMIZED.
    """
    num = np.sum((qobs - qsim) ** 2)
    den = np.sum((qobs - np.mean(qobs)) ** 2)

    return 1 - (num / den)


def _pbias(qsim: np.array, qobs: np.array):
    """Percent bias metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Percent bias. Can be negative or positive and gives the average
        relative error between the observed and simulated flows. This
        interpretation uses the definition that a positive bias value means
        that the simulation overestimates the true value (as opposed to many
        other sources on bias calculations that use the contrary interpretation).

    Notes
    -----
    PBIAS SHOULD AIM TO BE ZERO AND SHOULD NOT BE USED FOR CALIBRATION. FOR
    CALIBRATION, USE "abs_pbias" TO TAKE THE ABSOLUTE VALUE.
    """
    return (np.sum(qsim - qobs) / np.sum(qobs)) * 100


def _r2(qsim: np.array, qobs: np.array):
    """The r-squred metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        The r-squared (R2) metric equal to the square of the correlation
        coefficient.

    Notes
    -----
    The r2 should be MAXIMIZED.
    """
    return _correlation_coeff(qsim, qobs) ** 2


def _rmse(qsim: np.array, qobs: np.array):
    """Root mean square error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Root Mean Square Error. Units are the same as the timeseries data
        (ex. m3/s). It can take zero or positive values.

    Notes
    -----
    The rmse should be MINIMIZED.
    """
    return np.sqrt(np.mean((qobs - qsim) ** 2))


def _rrmse(qsim: np.array, qobs: np.array):
    """Relative root mean square error (ratio of rmse to mean) metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Ratio of the RMSE to the mean of the observations. It allows scaling
        RMSE values to compare results between time series of different
        magnitudes (ex. flows from small and large watersheds). Also known as
        the CVRMSE.

    Notes
    -----
    The rrmse should be MINIMIZED.
    """
    return _rmse(qsim, qobs) / np.mean(qobs)


def _rsr(qsim: np.array, qobs: np.array):
    """Ratio of root mean square error to standard deviation metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Root Mean Square Error (RMSE) divided by the standard deviation of the
        observations. Also known as the "Ratio of the Root Mean Square Error to
        the Standard Deviation of Observations".

    Notes
    -----
    The rsr should be MINIMIZED.
    """
    return _rmse(qobs, qsim) / np.std(qobs)


def _volume_error(qsim: np.array, qobs: np.array):
    """Volume error metric.

    Parameters
    ----------
    qsim : array_like
        Simulated streamflow vector.
    qobs : array_like
        Observed streamflow vector.

    Returns
    -------
    float
        Total error in terms of volume over the entire period. Expressed in
        terms of the same units as input data, so for flow rates it is
        important to multiply by the duration of the time-step to obtain actual
        volumes.

    Notes
    -----
    The volume_error should be MINIMIZED.
    """
    return np.sum(qsim - qobs) / np.sum(qobs)


"""
ADD OBJECTIVE FUNCTIONS HERE
"""
