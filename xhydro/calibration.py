"""
Created on Fri Dec  8 16:48:34 2023

@author: Richard Arsenault
"""

"""
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

    - model_config["model_name"]: Contains a string refering to the
                                  hydrological model to be run.

    - model_config["parameters"]: While not necessary to provide this, it is
                                  a reserved keyword used by the optimizer.

Any comments are welcome!
"""

"""
Import packages
"""
import hydroeval as he
import numpy as np
import spotpy
from spotpy import analyser
from spotpy.objectivefunctions import rmse
from spotpy.parameter import Uniform

from xhydro.hydrological_modelling import hydrological_model_selector


class spot_setup:
    """
    This class is used to create the spotpy calibration system that is used for
    hydrological model calibration.
    """

    def __init__(self, model_config, bounds_high, bounds_low, obj_func=None):
        """
        The initialization of the spot_setup object includes a generic
        "model_config" object containing hydrological modelling data required,
        low and high parameter bounds, as well as an objective function.
        Depending on the objective function, either spotpy or hydroeval will
        compute the value, since some functions are found only in one package.

        accepted obj_func values:
            Computed by hydroeval: ["nse", "kge", "kgeprime", "kgenp", "rmse",
                                    "mare", "pbias"]
            Computed by spotpy: ["bias","nashsutcliffe", "lognashsutcliffe",
                                 "log_p", "correlationcoefficient", "rsquared",
                                 "mse", "mae", "rrmse", "agreementindex",
                                 "covariance", "decomposed_mse", "rsr",
                                 "volume_error"]
        """

        # Gather the model_config dictionary and obj_func string.
        self.model_config = model_config
        self.obj_func = obj_func

        # Create the sampler for each parameter based on the bounds
        self.parameters = [
            Uniform("param" + str(i), bounds_low[i], bounds_high[i])
            for i in range(0, len(bounds_high))
        ]

    def simulation(self, x):
        """
        This is where the optimizer generates a parameter set from within the
        given bounds and generates the simulation results. We add the parameter
        "x" that is generated by spotpy to the model_config object at the
        reserved keyword "parameters".
        """
        self.model_config.update({"parameters": x})

        # Run the model and return Qsim, with model_config containing the
        # tested parameter set.
        return hydrological_model_selector(self.model_config)

    def evaluation(self):
        """
        Here is where we get the observed streamflow and make it available to
        compare the simulation and compute an objective function. It has to be
        in the Qobs keyword, although with some small changes
        model_config['Qobs'] could be a string to a file. Probably more
        efficient to load it into memory during preprocessing anyways to
        prevent recurring input/output and associated overhead. Currently, the
        package supposes that Qobs and Qsim have the same length, but this can
        be changed in the model_config parameterization and adding conditions
        here.
        """
        return self.model_config["Qobs"]

    def objectivefunction(
        self, simulation, evaluation, params=None, transform=None, epsilon=None
    ):
        """
        This function is where spotpy computes the objective function. Note
        that there are other possible inputs:
            - params: if we force a parameter set, this function can be used to
              compute the objective function easily.
            - transform: Possibility to transform flows before computing the
              objective function. "inv" takes 1/Q, "log" takes log(Q) and
              "sqrt" takes sqrt(Q) before computing.
            - epsilon: a small value to adjust flows before transforming to
              prevent log(0) or 1/0 computations when flow is zero.
        """
        return get_objective_function_value(
            evaluation, simulation, params, transform, epsilon, self.obj_func
        )


def perform_calibration(
    model_config,
    evaluation_metric,
    maximize,
    bounds_high,
    bounds_low,
    evaluations,
    algorithm="DDS",
):
    """
    TODO: maximize is not automatically defined. We should remove this option
    and force the algo/obj_fun to be coordinated to return the best value.
    """

    """
    This is the entrypoint for the model calibration. After setting-up the
    model_config object and other arguments, calling "perform_calibration" will
    return the optimal parameter set, objective function value and simulated
    flows on the calibration period.

    Inputs are as follows:

    - 'model_config' is the model configuration object (dict type) that
      contains all info to run the model. The model function called to run this
      model should always use this object and read-in data it requires. It will
      be up to the user to provide the data that the model requires.

    - evaluation_metric: The objective function (string) used for calibrating.

    - maximize: a Boolean flag to indicate that the objective function should
      be maximized (ex: "nse", "kge") instead of minimized (ex: "rmse", "mae").

    - bounds_high, bounds_low: high and low bounds respectively for the model
      parameters to be calibrated. Spotpy will sample parameter sets from
      within these bounds. The size must be equal to the number of parameters
      to calibrate. These are in np.array type.

    - evaluations: Maximum number of model evaluations (calibration budget) to
      perform before stopping the calibration process (int).

    - algorithm: The optimization algorithm to use. Currently, "DDS" and
      "SCEUA" are available, but it would be trivial to add more.

    """

    # Setup the spotpy object to prepare the calibration
    spotpy_setup = spot_setup(
        model_config,
        bounds_high=bounds_high,
        bounds_low=bounds_low,
        obj_func=evaluation_metric,
    )

    # Select an optimization algorithm and parameterize it, then run the
    # optimization process.
    if algorithm == "DDS":
        sampler = spotpy.algorithms.dds(
            spotpy_setup, dbname="DDS_optim", dbformat="ram", save_sim=False
        )
        sampler.sample(evaluations, trials=1)

    elif algorithm == "SCEUA":
        sampler = spotpy.algorithms.sceua(
            spotpy_setup, dbname="SCEUA_optim", dbformat="ram", save_sim=False
        )
        sampler.sample(evaluations, ngs=7, kstop=3, peps=0.1, pcento=0.1)

    # Gather optimization results
    results = sampler.getdata()

    # Get the best parameter set
    best_parameters = analyser.get_best_parameterset(
        results, like_index=1, maximize=maximize
    )
    best_parameters = [best_parameters[0][i] for i in range(0, len(best_parameters[0]))]

    # Get the best objective function as well depending if maximized or
    # minimized
    if maximize == True:
        bestindex, bestobjf = analyser.get_maxlikeindex(results)
    else:
        bestindex, bestobjf = analyser.get_minlikeindex(results)

    # Update the parameter set to put the best parameters in model_config...
    model_config.update({"parameters": best_parameters})

    # ... which can be used to run the hydrological model and get the best Qsim.
    Qsim = hydrological_model_selector(model_config)

    # Return the best parameters, Qsim and best objective function value.
    return best_parameters, Qsim, bestobjf


def transform_flows(Qobs, Qsim, transform=None, epsilon=None):
    """
    This subset of code is taken from the hydroeval package:
    https://github.com/ThibHlln/hydroeval/blob/main/hydroeval/hydroeval.py

    It is used to transform flows such that the objective function is computed
    on a transformed flow metric rather than on the original units of flow
    (ex: inverse, log-transformed, square-root)
    """

    # Generate a subset of simulation and evaluation series where evaluation
    # data is available
    Qsim = Qsim[~np.isnan(Qobs[:, 0]), :]
    Qobs = Qobs[~np.isnan(Qobs[:, 0]), :]

    # Transform the flow series if required
    if transform == "log":  # log transformation
        if not epsilon:
            # determine an epsilon value to avoid zero divide
            # (following recommendation in Pushpalatha et al. (2012))
            epsilon = 0.01 * np.mean(Qobs)
        Qobs, Qsim = np.log(Qobs + epsilon), np.log(Qsim + epsilon)

    elif transform == "inv":  # inverse transformation
        if not epsilon:
            # determine an epsilon value to avoid zero divide
            # (following recommendation in Pushpalatha et al. (2012))
            epsilon = 0.01 * np.mean(Qobs)
        Qobs, Qsim = 1.0 / (Qobs + epsilon), 1.0 / (Qsim + epsilon)

    elif transform == "sqrt":  # square root transformation
        Qobs, Qsim = np.sqrt(Qobs), np.sqrt(Qsim)

    # Return the flows after transformation (or original if no transform)
    return Qobs, Qsim


def get_objective_function_value(
    Qobs, Qsim, params, transform=None, epsilon=None, obj_func=None
):
    """
    This code returns the objective function as determined by the user, after
    transforming (if required) using the Qobs provided by the user and the Qsim
    simulated by the model controled by spotpy during the calibration process.
    """

    # Transform flows if needed
    if transform:
        Qobs, Qsim = transform_flows(Qobs, Qsim, transform, epsilon)

    # Default objective function if none provided by the user
    if not obj_func:
        like = rmse(Qobs, Qsim)

    # Popular ones we can use the hydroeval package
    elif obj_func.lower() in [
        "nse",
        "kge",
        "kgeprime",
        "kgenp",
        "rmse",
        "mare",
        "pbias",
    ]:
        # Get the correct function from the package
        like = he.evaluator(
            getattr(he, obj_func.lower()), simulations=Qsim, evaluation=Qobs
        )

    # In neither of these cases, use the obj_func method from spotpy
    else:
        like = obj_func(Qobs, Qsim)

    # Return results
    return like
