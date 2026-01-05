# Created on Fri Dec 8 16:48:34 2023
# @author: Laura Gatel based on calibration script from Richard Arsenault
"""
Global sensitivity analysis package for hydrological models.

This package contains the main framework for hydrological model sensitivity analysis. It
uses the spotpy global sensitivity analysis package applied on a "model_config" object. This
object is meant to be a container that can be used as needed by any hydrologic
model. For example, it can store datasets directly, paths to datasets (nc files
or other), csv files, basically anything that can be stored in a dictionary.

It then becomes the user's responsibility to ensure that required data for a
given model be provided in the model_config object both in the data preparation
stage and in the hydrological model implementation. This can be addressed by
a set of pre-defined codes for given model structures.

The sensitivity analysis aspect proceeds as follow :
### CHANGER LA SUITE

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

import os
import warnings
from copy import deepcopy

# Import packages
import numpy as np
import spotpy
import xarray as xr
from spotpy.parameter import Uniform

from xhydro.modelling import hydrological_model
from xhydro.modelling.obj_funcs import (
    get_objective_function,
)


__all__ = ["perform_global_sensitivity_analysis"]


class SpotSetup:
    """
    Create the spotpy global sensitivity analysis system that is used for hydrological model calibration.

    Parameters
    ----------
    model_config : dict
        The model configuration object that contains all info to run the model.
        The model function called to run this model should always use this object and read-in data it requires.
        It will be up to the user to provide the data that the model requires.
    bounds_high : np.ndarray
        High bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    bounds_low : np.ndarray
        Low bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
        within these bounds. The size must be equal to the number of parameters to calibrate.
    qobs : os.PathLike or np.ndarray or xr.Dataset or xr.DataArray
        Observed streamflow dataset (or path to it), used to compute the objective function.
        If using a dataset, it must contain a "streamflow" variable.
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

    mask : np.ndarray
        A vector indicating which values to preserve/remove from the objective function computation. 0=remove, 1=preserve.
    transform : str
        The method to transform streamflow prior to computing the objective function. Can be one of:
        Square root ('sqrt'), inverse ('inv'), or logarithmic ('log') transformation.
    epsilon : float
        Used to add a small delta to observations for log and inverse transforms, to eliminate errors
        caused by zero flow days (1/0 and log(0)). The added perturbation is equal to the mean observed streamflow
        times this value of epsilon.
    repetitions : int
        Simulation repetition number to perform the global sensitivity analysis. How to determine an appropriate number of repetitions :
        Check out https://spotpy.readthedocs.io/en/latest/Sensitivity_analysis_with_FAST/ .

    Returns
    -------
    SpotSetup object for calibration
    """

    def __init__(
        self,
        model_config: dict,
        bounds_high: np.ndarray | list[float | int],
        bounds_low: np.ndarray | list[float | int],
        qobs: os.PathLike | np.ndarray | xr.Dataset | xr.DataArray,
        obj_func: str | None = None,
        mask: np.ndarray | list[float | int] | None = None,
        transform: str | None = None,
        epsilon: float = 0.01,
        repetitions: int | None = 1000,
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
        bounds_high : np.ndarray
            High bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
            within these bounds. The size must be equal to the number of parameters to calibrate.
        bounds_low : np.ndarray
            Low bounds for the model parameters to be calibrated. Spotpy will sample parameter sets from
            within these bounds. The size must be equal to the number of parameters to calibrate.
        qobs : os.PathLike or np.ndarray or xr.Dataset or xr.DataArray
            Observed streamflow dataset (or path to it), used to compute the objective function.
            If using a dataset, it must contain a "streamflow" variable.
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
        evaluations : int
            Maximum number of model evaluations (calibration budget) to perform before stopping the calibration process.
        algorithm : str
            The optimization algorithm to use. Currently, "DDS" and "SCEUA" are available, but more can be easily added.
        mask : np.ndarray, optional
            A vector indicating which values to preserve/remove from the objective function computation. 0=remove, 1=preserve.
        transform : str, optional
            The method to transform streamflow prior to computing the objective function. Can be one of:
            Square root ('sqrt'), inverse ('inv'), or logarithmic ('log') transformation.
        epsilon : float
            Used to add a small delta to observations for log and inverse transforms, to eliminate errors
            caused by zero flow days (1/0 and log(0)). The added perturbation is equal to the mean observed streamflow
            times this value of epsilon.
        repetitions : int
            Simulation repetition number to perform the global sensitivity analysis. How to determine an appropriate number of repetitions :
            Check out https://spotpy.readthedocs.io/en/latest/Sensitivity_analysis_with_FAST/ .

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
        self.repetitions = repetitions

        # Create the sampler for each parameter based on the bounds
        self.parameters = [Uniform("param" + str(i), bounds_low[i], bounds_high[i]) for i in range(0, len(bounds_high))]

        # Load the observations
        if isinstance(qobs, np.ndarray):
            self.qobs = qobs
        else:
            # FIXME: This should be more robust, and should be able to handle other names
            if isinstance(qobs, xr.Dataset):
                if "streamflow" in qobs and "q" not in qobs:
                    warnings.warn(
                        "Default variable name has changed from 'streamflow' to 'q'. "
                        "Supporting 'streamflow' is deprecated and will be removed in xHydro v0.7.0.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    da = qobs.streamflow
                else:
                    da = qobs.q
            elif isinstance(qobs, xr.DataArray):
                da = qobs
            elif isinstance(qobs, os.PathLike):
                with xr.open_dataset(qobs) as ds:
                    if "streamflow" in ds and "q" not in ds:
                        warnings.warn(
                            "Default variable name has changed from 'streamflow' to 'q'. "
                            "Supporting 'streamflow' is deprecated and will be removed in xHydro v0.7.0.",
                            FutureWarning,
                            stacklevel=2,
                        )
                        da = ds.streamflow
                    else:
                        da = ds.q
            else:
                raise ValueError("qobs must be a NumPy array, xarray Dataset, xarray DataArray, or a path to a file.")
            da = da.squeeze()

            if self.model_config["model_name"] == "HELP" or self.model_config["model_name"] == "HELP":
                # changement pour ne pas utiliser slice car les time code ne se suivent pas (données mensuelles)
                da = da.where(da.time.values <= np.datetime64(self.model_config["end_date"]))
                da = da.where(da.time.values >= np.datetime64(self.model_config["start_date"]))
            else:
                # Subset the observed streamflow to the calibration period
                da = da.sel(time=slice(self.model_config["start_date"], self.model_config["end_date"]))

            self.qobs = da.values

    def simulation(self, x):
        """
        Simulation function for spotpy.

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
        """
        Evaluation function for spotpy.

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
        return self.qobs

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
            mask=self.mask,
            transform=self.transform,
            epsilon=self.epsilon,
        )

        return obj_fun_val


def perform_global_sensitivity_analysis(
    model_config: dict,
    obj_func: str,
    bounds_high: np.ndarray | list[float | int],
    bounds_low: np.ndarray | list[float | int],
    evaluations: int,
    qobs: os.PathLike | np.ndarray | xr.Dataset | xr.DataArray,
    algorithm: str = "DDS",
    mask: np.ndarray | list[float | int] | None = None,
    transform: str | None = None,
    epsilon: float = 0.01,
    repetitions: int | None = 1000,
    sampler_kwargs: dict | None = None,
):
    """
    Perform calibration using SPOTPY.

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
    qobs : os.PathLike or np.ndarray or xr.Dataset or xr.DataArray
        Observed streamflow dataset (or path to it), used to compute the objective function.
        If using a dataset, it must contain a "streamflow" variable.
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
    repetitions : int
        Simulation repetition number to perform the global sensitivity analysis. How to determine an appropriate number of repetitions :
        Check out https://spotpy.readthedocs.io/en/latest/Sensitivity_analysis_with_FAST/ .
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
    # Set up the spotpy object to prepare the GSA
    spotpy_setup = SpotSetup(
        model_config,
        qobs=qobs,
        bounds_high=bounds_high,
        bounds_low=bounds_low,
        obj_func=obj_func,
        mask=mask,
        transform=transform,
        epsilon=epsilon,
        repetitions=repetitions,
    )

    # Select a GSA algorithm then run the process.
    sampler_kwargs = deepcopy(sampler_kwargs) or None
    if sampler_kwargs is None:
        sampler_kwargs = {}

    if algorithm == "FAST":
        sampler = spotpy.algorithms.fast(spotpy_setup, dbname="FAST_hymod", dbformat="csv", db_precision=np.float32)
        sampler.sample(repetitions)

        # Get the sampler hyperparameters, either default or user-provided.
        defaults = {"trials": 1}
        sampler_kwargs = defaults | sampler_kwargs

    else:
        raise ValueError(f"Algorithm {algorithm} is not supported.")

    # Gather sensitivity results
    results = spotpy.analyser.load_csv_results("FAST_hymod")

    # Example to get the sensitivity index of each parameter
    si = spotpy.analyser.get_sensitivity_of_fast(results)

    # Return the SI of each parameters
    return si
