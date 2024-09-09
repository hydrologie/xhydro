# Created on Fri Dec 8 16:48:34 2023
# @author: Richard Arsenault
"""Calibration package for hydrological models.

This package contains the main framework for hydrological model calibration. It
uses the spotpy calibration package applied on a "model_config" object. This
object is meant to be a container that can be used as needed by any hydrologic
model. For example, it can store datasets directly, paths to datasets (nc files
or other), csv files, basically anything that can be stored in a dictionary.

It then becomes the user's responsibility to ensure that required data for a
given model be provided in the model_config object both in the data preparation
stage and in the hydrological model implementation. This can be addressed by
a set of pre-defined codes for given model structures.

For example, for GR4J, only small datasets are required and can be stored
directly in the model_config dictionary. However, for Hydrotel or Raven models,
maybe it is better to pass paths to netcdf files which can be passed to the
models. This will require pre- and post-processing, but this can easily be
handled at the stage where we create a hydrological model and prepare the data.

The calibration aspect then becomes trivial:

    1. A model_config object is passed to the calibrator.
    2. Lower and upper bounds for calibration parameters are defined and passed
    3. An objective function, optimizer and hyperparameters are also passed.
    4. The calibrator uses this information to develop parameter sets that are
       then passed as inputs to the "model_config" object.
    5. The calibrator launches the desired hydrological model with the
       model_config object (now containing the parameter set) as input.
    6. The appropriate hydrological model function then parses "model_config",
       takes the parameters and required data, launches a simulation and
       returns simulated flow (Qsim).
    7. The calibration package then compares Qobs and Qsim and computes the
       objective function value, and returns this to the sampler that will then
       repeat the process to find optimal parameter sets.
    8. The code returns the best parameter set, objective function value, and
       we also return the simulated streamflow on the calibration period for
       user convenience.

This system has the advantage of being extremely flexible, robust, and
efficient as all data can be either in-memory or only the reference to the
required datasets on disk is passed around the callstack.

Currently, the model_config object has 3 mandatory keywords for the package to
run correctly in all instances:

    - model_config["Qobs"]: Contains the observed streamflow used as the
                            calibration target.

    - model_config["model_name"]: Contains a string referring to the
                                  hydrological model to be run.

    - model_config["parameters"]: While not necessary to provide this, it is
                                  a reserved keyword used by the optimizer.

Any comments are welcome!
"""

from copy import deepcopy

# Import packages
from typing import Optional

import numpy as np
import spotpy
import xarray as xr
from spotpy import analyser
from spotpy.parameter import Uniform

from xhydro.modelling import hydrological_model
from xhydro.modelling.obj_funcs import (
    _get_objfun_minimize_or_maximize,
    _get_optimizer_minimize_or_maximize,
    get_objective_function,
)

__all__ = ["perform_calibration"]


class SpotSetup:
    """Create the spotpy calibration system that is used for hydrological model calibration.

    Parameters
    ----------
    model_config : dict
        The model configuration object that contains all info to run the model.
        The model function called to run this model should always use this object and read-in data it requires.
        It will be up to the user to provide the data that the model requires.
    bounds_high : np.array
        High bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    bounds_low : np.array
        Low bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    obj_func : str
        The objective function used for calibrating. Can be any one of these:

            - "abs_bias" : Absolute value of the "bias" metric
            - "abs_pbias": Absolute value of the "pbias" metric
            - "abs_volume_error" : Absolute value of the volume_error metric
            - "agreement_index": Index of agreement
            - "correlation_coeff": Correlation coefficient
            - "kge" : Kling Gupta Efficiency metric (2009 version)
            - "kge_mod" : Kling Gupta Efficiency metric (2012 version)
            - "mae": Mean Absolute Error metric
            - "mare": Mean Absolute Relative Error metric
            - "mse" : Mean Square Error metric
            - "nse": Nash-Sutcliffe Efficiency metric
            - "r2" : r-squared, i.e. square of correlation_coeff.
            - "rmse" : Root Mean Square Error
            - "rrmse" : Relative Root Mean Square Error (RMSE-to-mean ratio)
            - "rsr" : Ratio of RMSE to standard deviation.

    take_negative : bool
        Inidactor to take the negative of the objective function value in optimization to ensure convergence
        in the right direction.
    mask : np.array
        A vector indicating which values to preserve/remove from the objective function computation. 0=remove, 1=preserve.
    transform : str
        The method to transform streamflow prior to computing the objective function. Can be one of:
        Square root ('sqrt'), inverse ('inv'), or logarithmic ('log') transformation.
    epsilon : float
        Used to add a small delta to observations for log and inverse transforms, to eliminate errors
        caused by zero flow days (1/0 and log(0)). The added perturbation is equal to the mean observed streamflow
        times this value of epsilon.

    Returns
    -------
    SpotSetup object for calibration
    """

    def __init__(
        self,
        model_config: dict,
        bounds_high: np.ndarray,
        bounds_low: np.ndarray,
        obj_func: str | None = None,
        take_negative: bool = False,
        mask: np.ndarray | None = None,
        transform: str | None = None,
        epsilon: float = 0.01,
    ):
        """
        Initialize the SpotSetup object.

        The initialization of the SpotSetup object includes a generic
        "model_config" object containing hydrological modelling data required,
        low and high parameter bounds, as well as an objective function.
        Depending on the objective function, either spotpy or hydroeval will
        compute the value, since some functions are found only in one package.

        Parameters
        ----------
        model_config : dict
            A dictionary containing the configuration for the hydrological model.
            Must contain a key "model_name" with the name of the model to use: "Hydrotel".
            The required keys depend on the model being used. Use the function
            `xh.modelling.get_hydrological_model_inputs` to get the required keys for a given model.
        obj_func : str
            The objective function used for calibrating. Can be any one of these:

                - "abs_bias" : Absolute value of the "bias" metric
                - "abs_pbias": Absolute value of the "pbias" metric
                - "abs_volume_error" : Absolute value of the volume_error metric
                - "agreement_index": Index of agreement
                - "correlation_coeff": Correlation coefficient
                - "kge" : Kling Gupta Efficiency metric (2009 version)
                - "kge_mod" : Kling Gupta Efficiency metric (2012 version)
                - "mae": Mean Absolute Error metric
                - "mare": Mean Absolute Relative Error metric
                - "mse" : Mean Square Error metric
                - "nse": Nash-Sutcliffe Efficiency metric
                - "r2" : r-squared, i.e. square of correlation_coeff.
                - "rmse" : Root Mean Square Error
                - "rrmse" : Relative Root Mean Square Error (RMSE-to-mean ratio)
                - "rsr" : Ratio of RMSE to standard deviation.
        bounds_high : np.array
            High bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
            within these bounds. The size must be equal to the number of parameters to calibrate.
        bounds_low : np.array
            Low bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
            within these bounds. The size must be equal to the number of parameters to calibrate.
        evaluations : int
            Maximum number of model evaluations (calibration budget) to perform before stopping the calibration process.
        algorithm : str
            The optimization algorithm to use. Currently, "DDS" and "SCEUA" are available, but more can be easily added.
        take_negative : bool
            Wether to take the negative of the objective function value in optimization to ensure convergence
            in the right direction.
        mask : np.array, optional
            A vector indicating which values to preserve/remove from the objective function computation. 0=remove, 1=preserve.
        transform : str, optional
            The method to transform streamflow prior to computing the objective function. Can be one of:
            Square root ('sqrt'), inverse ('inv'), or logarithmic ('log') transformation.
        epsilon : float
            Used to add a small delta to observations for log and inverse transforms, to eliminate errors
            caused by zero flow days (1/0 and log(0)). The added perturbation is equal to the mean observed streamflow
            times this value of epsilon.

        Returns
        -------
        SpotSetup object for calibration.
        """
        # Gather the model_config dictionary and obj_func string, and other
        # optional arguments.
        self.model_config = deepcopy(model_config)
        self.obj_func = obj_func
        self.mask = mask
        self.transform = transform
        self.epsilon = epsilon
        self.take_negative = take_negative

        # Create the sampler for each parameter based on the bounds
        self.parameters = [
            Uniform("param" + str(i), bounds_low[i], bounds_high[i])
            for i in range(0, len(bounds_high))
        ]

    def simulation(self, x):
        """Simulation function for spotpy.

        This is where the optimizer generates a parameter set from within the
        given bounds and generates the simulation results. We add the parameter
        "x" that is generated by spotpy to the model_config object at the
        reserved keyword "parameters".

        Parameters
        ----------
        x : array_like
            Tested parameter set.

        Returns
        -------
        array_like
            Simulated streamflow.
        """
        self.model_config.update({"parameters": x})

        # Run the model and return qsim, with model_config containing the
        # tested parameter set.
        qsim = hydrological_model(self.model_config).run()

        # Return the array of values from qsim for the objective function eval.
        return qsim["streamflow"].values

    def evaluation(self):
        """Evaluation function for spotpy.

        Here is where we get the observed streamflow and make it available to
        compare the simulation and compute an objective function. It has to be
        in the Qobs keyword, although with some small changes
        model_config['Qobs'] could be a string to a file. Probably more
        efficient to load it into memory during preprocessing anyway to
        prevent recurring input/output and associated overhead. Currently, the
        package supposes that Qobs and Qsim have the same length, but this can
        be changed in the model_config parameterization and adding conditions
        here.

        Returns
        -------
        array_like
            Observed streamflow.
        """
        qobs = self.model_config["qobs"]
        if isinstance(qobs, xr.Dataset):
            qobs = qobs["qobs"]

        return qobs

    def objectivefunction(
        self,
        simulation,
        evaluation,
    ):
        """
        Objective function for spotpy.

        This function is where spotpy computes the objective function.

        Parameters
        ----------
        simulation : array_like
            Vector of simulated streamflow.
        evaluation : array_like
            Vector of observed streamflow to compute objective function.

        Returns
        -------
        float
            The value of the objective function.
        """
        obj_fun_val = get_objective_function(
            evaluation,
            simulation,
            obj_func=self.obj_func,
            take_negative=self.take_negative,
            mask=self.mask,
            transform=self.transform,
            epsilon=self.epsilon,
        )

        return obj_fun_val


def perform_calibration(
    model_config: dict,
    obj_func: str,
    bounds_high: np.ndarray,
    bounds_low: np.ndarray,
    evaluations: int,
    algorithm: str = "DDS",
    mask: np.ndarray | None = None,
    transform: str | None = None,
    epsilon: float = 0.01,
    sampler_kwargs: dict | None = None,
):
    """Perform calibration using SPOTPY.

    This is the entrypoint for the model calibration. After setting-up the
    model_config object and other arguments, calling "perform_calibration" will
    return the optimal parameter set, objective function value and simulated
    flows on the calibration period.

    Parameters
    ----------
    model_config : dict
        The model configuration object that contains all info to run the model.
        The model function called to run this model should always use this object and read-in data it requires.
        It will be up to the user to provide the data that the model requires.
    obj_func : str
        The objective function used for calibrating. Can be any one of these:

            - "abs_bias" : Absolute value of the "bias" metric
            - "abs_pbias": Absolute value of the "pbias" metric
            - "abs_volume_error" : Absolute value of the volume_error metric
            - "agreement_index": Index of agreement
            - "correlation_coeff": Correlation coefficient
            - "kge" : Kling Gupta Efficiency metric (2009 version)
            - "kge_mod" : Kling Gupta Efficiency metric (2012 version)
            - "mae": Mean Absolute Error metric
            - "mare": Mean Absolute Relative Error metric
            - "mse" : Mean Square Error metric
            - "nse": Nash-Sutcliffe Efficiency metric
            - "r2" : r-squared, i.e. square of correlation_coeff.
            - "rmse" : Root Mean Square Error
            - "rrmse" : Relative Root Mean Square Error (RMSE-to-mean ratio)
            - "rsr" : Ratio of RMSE to standard deviation.

    bounds_high : np.array
        High bounds for the model parameters to be calibrated. SPOTPY will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    bounds_low : np.array
        Low bounds for the model parameters to be calibrated. SPOTPY will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    evaluations : int
        Maximum number of model evaluations (calibration budget) to perform before stopping the calibration process.
    algorithm : str
        The optimization algorithm to use. Currently, "DDS" and "SCEUA" are available, but more can be easily added.
    mask : np.array, optional
        A vector indicating which values to preserve/remove from the objective function computation. 0=remove, 1=preserve.
    transform : str, optional
        The method to transform streamflow prior to computing the objective function. Can be one of:
        Square root ('sqrt'), inverse ('inv'), or logarithmic ('log') transformation.
    epsilon : scalar float
        Used to add a small delta to observations for log and inverse transforms, to eliminate errors
        caused by zero flow days (1/0 and log(0)). The added perturbation is equal to the mean observed streamflow
        times this value of epsilon.
    sampler_kwargs : dict
        Contains the keywords and hyperparameter values for the optimization algorithm.
        Keywords depend on the algorithm choice. Currently, SCEUA and DDS are supported with
        the following default values:
        - SCEUA: dict(ngs=7, kstop=3, peps=0.1, pcento=0.1)
        - DDS: dict(trials=1)

    Returns
    -------
    best_parameters : array_like
        The optimized parameter set.
    qsim : xr.Dataset
        Simulated streamflow using the optimized parameter set.
    bestobjf : float
        The best objective function value.
    """
    # Get objective function and algo optimal convergence direction. Necessary
    # to ensure that the algorithm is optimizing in the correct direction
    # (maximizing or minimizing). This code determines the required direction
    # for the objective function and the working direction of the algorithm.
    of_maximize = _get_objfun_minimize_or_maximize(obj_func)
    algo_maximize = _get_optimizer_minimize_or_maximize(algorithm)

    # They are not working in the same direction. Take the negative of the OF.
    if of_maximize != algo_maximize:
        take_negative = True
    else:
        take_negative = False

    # Set up the spotpy object to prepare the calibration
    spotpy_setup = SpotSetup(
        model_config,
        bounds_high=bounds_high,
        bounds_low=bounds_low,
        obj_func=obj_func,
        take_negative=take_negative,
        mask=mask,
        transform=transform,
        epsilon=epsilon,
    )

    # Select an optimization algorithm and parameterize it, then run the
    # optimization process.
    sampler_kwargs = deepcopy(sampler_kwargs) or None
    if sampler_kwargs is None:
        sampler_kwargs = {}

    if algorithm == "DDS":
        sampler = spotpy.algorithms.dds(
            spotpy_setup, dbname="DDS_optim", dbformat="ram", save_sim=False
        )

        # Get the sampler hyperparameters, either default or user-provided.
        defaults = {"trials": 1}
        sampler_kwargs = defaults | sampler_kwargs

        # Ensure there is only 1 hyperparameter passed by the user, if
        # applicable
        if len(sampler_kwargs) == 1:
            sampler.sample(evaluations, **sampler_kwargs)
        else:
            raise ValueError(
                "sampler_kwargs should only contain the keyword 'trials' when using DDS."
            )

    elif algorithm == "SCEUA":
        sampler = spotpy.algorithms.sceua(
            spotpy_setup, dbname="SCEUA_optim", dbformat="ram", save_sim=False
        )
        # Get the sampler hyperparameters, either default or user-provided.
        defaults = {"ngs": 7, "kstop": 3, "peps": 0.1, "pcento": 0.1}
        sampler_kwargs = defaults | sampler_kwargs

        # If the user provided a custom sampler hyperparameter set.
        if len(sampler_kwargs) == 4:
            sampler.sample(evaluations, **sampler_kwargs)
        else:
            raise ValueError(
                "sampler_kwargs should only contain the keywords [ngs, kstop, peps, pcento] when using SCEUA."
            )

    # Gather optimization results
    results = sampler.getdata()

    # Get the best parameter set
    best_parameters = analyser.get_best_parameterset(
        results, like_index=1, maximize=of_maximize
    )
    best_parameters = [best_parameters[0][i] for i in range(0, len(best_parameters[0]))]

    # Get the best objective function as well depending if maximized or
    # minimized
    if of_maximize:
        _, bestobjf = analyser.get_maxlikeindex(results)
    else:
        _, bestobjf = analyser.get_minlikeindex(results)

    # Reconvert objective function if required.
    if take_negative:
        bestobjf = bestobjf * -1

    # Update the parameter set to put the best parameters in model_config...
    model_config = deepcopy(model_config)
    model_config.update({"parameters": best_parameters})

    # ... which can be used to run the hydrological model and get the best Qsim.
    qsim = hydrological_model(model_config).run()

    # Return the best parameters, qsim and best objective function value.
    return best_parameters, qsim, bestobjf
