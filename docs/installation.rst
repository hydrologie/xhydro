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

Stable release
--------------
Some of the dependencies of `xHydro` can be difficult to install using `pip`. For this reason, we highly recommend installing `xHydro` using Anaconda. This will ensure that all dependencies are met and that the package will work as expected.

    .. code-block:: console

     conda install -c conda-forge xhydro

Alternatively, you can install `xHydro` using `pip`. All features in `xHydro` itself will work, but some libraries (`xscen` and `xESMF` in particular) might not be fully functional. If you are on Windows, installing through `pip` might not currently be possible because of `raven-hydro`.

    .. code-block:: console

     pip install xhydro

Either of those will install the latest stable release of `xHydro` with all its dependencies, with two exception:

If you wish to use the `extreme_value_analysis` module, which relies on the `Extremes.jl`_ Julia package, you will need to install the `julia` extra:

    .. code-block:: console

     pip install xhydro[julia]

.. _Extremes.jl: https://github.com/jojal5/Extremes.jl

If you wish to use the `frequency_analysis.regional` module, you will need to install the `lmoments3` library yourself. This library is available on both `PyPI` or `conda-forge`, but has a restricted license.

    .. code-block:: console

     conda install -c conda-forge lmoments3

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
