============
Installation
============

We strongly recommend installing `xhydro` in an Anaconda Python environment. Futhermore, due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment. If `mamba` is not already your default solver, consider running the following commands in order to speed up the process:

    .. code-block:: console

        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Stable release
--------------
Due to the complexity of the install process of some dependencies, `xhydro` should not be installed directly from `PyPI` unless you are sure that all requirements are met.

Until the library is available on `Conda-Forge` for a more streamlined installation, we recommend following the instructions below, but replacing the first step with files from the latest release on `PyPI`_.

.. _PyPI: https://pypi.org/project/xhydro/

To create a working environment and install xHydro, copy the `environment-dev.yml` file from the root of the repository and run the following commands:

    .. code-block:: console

     conda env create -f environment-dev.yml
     conda activate xhydro
     python -m pip install xhydro --no-deps

This is the preferred method to install xHydro, as it will always install the most recent stable release.

If for some reason you wish to install the `PyPI` version of `xhydro` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

From sources
------------
`xHydro` is still under active development and the latest features might not yet be available on `PyPI`.

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

        python -m pip install -e .[dev]

        Even if you do not intend to contribute to `xHydro`, we favor using `environment-dev.yml` over `environment.yml` because it includes additional packages that are used to run all the examples provided in the documentation.
        If for some reason you wish to install the `PyPI` version of `xHydro` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

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
