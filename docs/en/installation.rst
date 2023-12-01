============
Installation
============

Stable release
--------------
Due to the complexity of the install process of some dependencies, we strongly recommend installing `xhydro` in an Anaconda Python environment.
To create a working environment and install xHydro, copy the `environment.yml` file from the root of the repository and run the following commands:

.. code-block:: console

    $ mamba env create -f environment.yml
    $ mamba activate xhydro
    $ python -m pip install xhydro --no-deps

This is the preferred method to install `xHydro`, as it will always install the most recent stable release.

If for some reason you wish to install the `PyPI` version of `xhydro` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

From sources
------------
`xHydro` is still under active development and the latest features might not yet be available on `PyPI`.
To install the latest development version, you can install `xHydro` directly from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:hydrologie/xhydro

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/hydrologie/xhydro/tarball/main

Once you have a copy of the source, you can create a working environment and install `xHydro` in it:

.. code-block:: console

    $ mamba env create -f environment.yml
    $ mamba activate xhydro
    $ python -m pip install . --no-deps

When new changes are made to the `Github repo`_, you can update your local copy using:

.. code-block:: console

    $ git pull origin main
    $ mamba env update -n xhydro -f environment.yml
    $ mamba activate xhydro
    $ python -m pip install . --no-deps

.. _Github repo: https://github.com/hydrologie/xhydro
.. _tarball: https://github.com/hydrologie/xhydro/tarball/main
