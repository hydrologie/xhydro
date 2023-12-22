"""Hydrological modelling framework.

This collection of functions should serve as the main entry point for
hydrological modelling. The entire framework is based on the "model_config"
object. This object is meant to be a container that can be used as needed by
any hydrologic model. For example, it can store datasets directly, paths to
datasets (nc files or other), csv files, basically anything that can be stored
in a dictionary.

It then becomes the user's responsibility to ensure that required data for a
given model be provided in the model_config object both in the data preparation
stage and in the hydrological model implementation. This can be addressed by
a set of pre-defined codes for given model structures. This present package
(hydrological_modelling.py) should contain the logic to:

    1. From the model_config["model_name"] key, select the appropriate function
       (hydrological model) to run.
    2. Pass the model_config object to the correct hydrological modelling
       function.
    3. Parse the model_config object to extract required data for the given
       model, such as: parameters, meteorological data, paths to input files, and catchment characteristics as required
    4. Run the hydrological model with the given data
    5. Return the streamflow (Qsim).

This will make it very easy to add hydrological models, no matter their
complexity. It will also make it easy for newer Python users to implement
models as the logic is clearly exposed and does not make use of complex classes
or other dynamic objects. Models can be added here, and a specific routine
should also be defined to produce the required model_config for each model.

Once this is accomplished, running the model from a model_config object becomes
trivial and allows for easy calibration, regionalisation, analysis and any
other type of interaction.
"""

import numpy as np


def hydrological_model_selector(model_config):
    """Hydrological model selector.

    This is the main hydrological model selector. This is the code that looks
    at the "model_config["model_name"]" keyword and calls the appropriate
    hydrological model function.
    """
    if model_config["model_name"] == "Dummy":
        Qsim = dummy_model(model_config)

    elif model_config["model_name"] == "ADD_OTHER_HERE":
        # ADD OTHER MODELS HERE
        Qsim = 0
    else:
        raise NotImplementedError()

    return Qsim


def dummy_model(model_config):
    """Dummy model.

    Dummy model to show the implementation we should be aiming for. Each model
    will have its own required data that users can pass.
    """
    # Parse the model_config object to extract required information
    precip = model_config["precip"]
    temperature = model_config["temperature"]
    area = model_config["drainage_area"]
    X = model_config["parameters"]

    # Run the dummy model using these data. Keeping it simple to calculate by
    # hand to ensure the calibration algorithm is working correctly and data
    # are handled correctly
    Qsim = np.empty(len(precip))
    for t in range(0, len(precip)):
        Qsim[t] = (precip[t] * X[0] + abs(temperature[t]) * X[1]) * X[2] * area

    return Qsim
