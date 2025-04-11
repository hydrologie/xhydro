============
Installation
============

We strongly recommend installing `xHydro` in an Anaconda Python environment. Furthermore, due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment. If `mamba` is not already your default solver, consider running the following commands in order to speed up the process:

    .. code-block:: console

        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Furthermore, some libraries used by `xHydro` or its dependencies may require the activation of the environment to work properly. It is therefore recommended to always activate the environment before running any Python code to avoid issues. You can do this by running the following command:

    .. code-block:: console

        conda activate name_of_your_environment

This is also true for some GUI applications, such as PyCharm, which fail to properly activate the project environment by themselves. If you are using such an application, make sure to activate the environment before launching it.
For Windows users, this can be done by running the command above in the Anaconda Prompt before launching the application through that same prompt, or by launching the application from the Anaconda Navigator (with the correct environment activated in it).


Stable release (Anaconda)
-------------------------
Some of the dependencies of `xHydro` can be difficult to install using `pip`. For this reason, we highly recommend installing `xHydro` using Anaconda. This will ensure that all dependencies are met and that the package will work as expected.

    .. code-block:: console

        conda install -c conda-forge xhydro

This will install the latest stable release of `xHydro` with all its dependencies, with two exception:

If you wish to use the `extreme_value_analysis` module, which relies on the `Extremes.jl`_ Julia package, you will need to install the `julia` extra:

    .. code-block:: console

        pip install xhydro[julia]

.. _Extremes.jl: https://github.com/jojal5/Extremes.jl

If you wish to use the `frequency_analysis.regional` module, you will need to install the `lmoments3` library yourself. This library is available on both `PyPI` or `conda-forge`, but has a restricted license. Note that if you installed `xHydro` using `conda`, you will need to install `lmoments3` using `conda` as well. If you installed `xHydro` using the `environment-dev.yml` file, `lmoments3` will have been added automatically.

    .. code-block:: console

        conda install -c conda-forge lmoments3

Stable release (PyPI)
---------------------
You can also install `xHydro` from `PyPI` using `pip`:

    .. code-block:: console

       pip install xhydro

The points above about the `extreme_value_analysis` and `frequency_analysis.regional` modules also apply here. Also note that some of the dependencies of `xHydro` might not be fully functional when installed using `pip`, such as `xESMF`.
**IMPORTANT:** There currently seems to be an issue with the `juliacall` library when installing `xHydro` from `PyPI`, which will cause segmentation faults when trying to import it. Until this issue is resolved or a workaround is found, we recommend installing `xHydro` from `conda` instead if you want to use the `extreme_value_analysis` module.

Additionally, `ravenpy` and `raven-hydro` can be quite difficult to install in `pip` environments. Thus, installing `xHydro` from `PyPI` will not include those two dependencies and the associated modules will be deactivated. If you want to use Raven-based hydrological modelling, some commands similar to the following should be run:

    .. code-block:: console

       apt-get upgrade && apt-get upgrade -y && apt-get install -y git gdal-bin python3-gdal libgdal-dev gcc libnetcdf-dev
       pip install xhydro[raven]

More information on how to install `ravenpy` can be found in the `ravenpy`_ and `raven-hydro`_ documentation. Windows users may not be able to install all of the dependencies of `ravenpy` using `pip`, and may need to install `xHydro` using `conda` instead if they want to use `RavenPy`.

.. _ravenpy: https://ravenpy.readthedocs.io/en/latest/installation.html#python-installation-pip
.. _raven-hydro: https://github.com/Ouranosinc/raven-hydro?tab=readme-ov-file#installation

From sources
------------
`xHydro` is still under active development and the latest features might not yet be available on `PyPI` or `conda-forge`. If you want to use the latest features, or if you want to contribute to the development of `xHydro`, you can install it from the sources.

The sources for xHydro can be downloaded from the `Github repo`_.

#. Download the source code from the `Github repo`_ using one of the following methods:

    * Clone the public repository:

        .. code-block:: console

            git clone git@github.com:hydrologie/xhydro

    * Download the `tarball <https://github.com/hydrologie/xhydro/tarball/main>`_:

        .. code-block:: console

            curl -OJL https://github.com/hydrologie/xhydro/tarball/main

#. Once you have a copy of the source, you can install it with:

    .. code-block:: console

         conda env create -f environment-dev.yml
         conda activate xhydro-dev
         make dev

    If you are on Windows, replace the ``make dev`` command with the following:

    .. code-block:: console

        python -m pip install -e .[all]

    Even if you do not intend to contribute to `xHydro`, we favor using `environment-dev.yml` over `environment.yml` because it includes additional packages that are used to run all the examples provided in the documentation. If for some reason you wish to install the `PyPI` version of `xHydro` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

#. When new changes are made to the `Github repo`_, you can update your local copy using the following commands from the root of the repository:

    .. code-block:: console

         git fetch
         git checkout main
         git pull origin main
         conda env update -n xhydro-dev -f environment-dev.yml
         conda activate xhydro-dev
         make dev

    These commands should work most of the time, but if big changes are made to the repository, you might need to remove the environment and create it again.

.. _Github repo: https://github.com/hydrologie/xhydro
