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

.. note::

   Some libraries used by `xHydro` or its dependencies may not function correctly unless the appropriate environment is activated. To avoid potential issues, it is **strongly recommended** to always activate your environment before running any Python code. You can do this with the following command:

   .. code-block:: console

      conda activate name_of_your_environment

   This recommendation also applies to certain GUI applications, such as PyCharm, which may not automatically activate the project environment. In such cases, be sure to activate the environment before launching the application.

   On Windows, this can be done by running the command above in the *Anaconda Prompt*, and then launching the application from that same prompt. Alternatively, you can launch the application via *Anaconda Navigator*, ensuring the correct environment is selected beforehand.

Stable release (Anaconda)
-------------------------
Some of the dependencies of `xHydro` can be difficult to install using `pip`. For this reason, we strongly recommend installing `xHydro` using Anaconda. This will ensure that all dependencies are met and that the package will work as expected.

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
Although not recommended, you can also install `xHydro` from `PyPI` using `pip`:

.. code-block:: console

   pip install xhydro

Some dependencies of `xHydro` may not be fully functional when installed via `pip`. For example, packages like `xESMF` may require additional system-level configuration to work correctly.

The considerations mentioned above regarding the `extreme_value_analysis` and `frequency_analysis.regional` modules also apply to the `PyPI` installation, but with a few additional notes:

* The `julia` extra requires a working C++ compiler. Some users have reported segmentation faults when using the `juliacall` library with `xHydro` installed from `PyPI`, which we believe is due to the Julia environment not being launched with the correct C++ compiler. If you encounter this issue, we recommend using the `conda` installation method instead.

* The `raven` extra relies on the `ravenpy` and `raven-hydro` packages, which can be challenging to install in standard `pip` environments due to complex system-level dependencies. See the note below for more details. Unless you are familiar with the installation of these packages, we recommend using the `conda` installation method instead.

.. note::

    We have not currently done extensive testing for installing the `raven` extra using `pip`, but the following commands are known to work on some Linux systems:

    .. code-block:: console

       apt-get update && apt-get upgrade -y
       apt-get install -y git gdal-bin python3-gdal libgdal-dev libnetcdf-dev build-essential
       pip install xhydro[raven]

    This will install `ravenpy` and `raven-hydro` from `PyPI`. Alternatively, you can replace the last line and install `ravenpy` only, provided that you have your own Raven executable. For further guidance on installing these packages, refer to the official documentation of `ravenpy`_ and `raven-hydro`_.

.. warning::

   On **Windows**, installing all the dependencies for `ravenpy` may require Administrator privileges, as some packages must be added to the system `PATH`. If you encounter issues, consider using the `Anaconda` installation method instead.

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
