.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/hydrologie/xhydro/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xHydro could always use more documentation, whether as part of the official xHydro docs, in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/hydrologie/xhydro/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome. :)

Get Started!
------------

.. note::

    If you are new to using GitHub and `git`, please read `this guide <https://guides.github.com/activities/hello-world/>`_ first.

.. warning::

    Anaconda Python users: Due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment. Consider running the following commands in order to speed up the process::

        $ conda install -n base conda-libmamba-solver
        $ conda config --set solver libmamba

    For more information, please see the following link: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

    Alternatively, you can use the `mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ package manager, which is a drop-in replacement for ``conda``. If you are already using `mamba`, replace the following commands with ``mamba`` instead of ``conda``.

Ready to contribute? Here's how to set up ``xhydro`` for local development.

#. If you are not already an `xhydro` collaborator, fork the ``xhydro`` repo on GitHub.
#. Clone your fork locally::

    $ git clone git@github.com:your_name_here/xhydro.git

#. Install your local copy into a development environment. You can create a new Anaconda development environment with::

    $ conda env create -f environment-dev.yml
    $ conda activate xhydro-dev
    $ flit install --symlink

   This installs ``xhydro`` in an "editable" state, meaning that changes to the code are immediately seen by the environment.

#. To ensure a consistent coding style, install the ``pre-commit`` hooks to your local clone::

    $ pre-commit install

   On commit, ``pre-commit`` will check that ``black``, ``blackdoc``, ``isort``, ``flake8``, and ``ruff`` checks are passing, perform automatic fixes if possible, and warn of violations that require intervention. If your commit fails the checks initially, simply fix the errors, re-add the files, and re-commit.

   You can also run the hooks manually with::

    $ pre-commit run -a

   If you want to skip the ``pre-commit`` hooks temporarily, you can pass the ``--no-verify`` flag to `$ git commit`.

#. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, we **strongly** suggest running the tests in your environment or with the help of ``tox``::

    $ python -m pytest
    # Or, to run multiple build tests
    $ tox

   .. note::

       Running `pytest` or `tox` will automatically fetch and cache the testing data for the package to your local cache (using the `platformdirs` library). On Linux, this is located at ``XDG_CACHE_HOME`` (usually ``~/.cache``). On Windows, this is located at ``%LOCALAPPDATA%`` (usually ``C:\Users\username\AppData\Local``). On MacOS, this is located at ``~/Library/Caches``.

       If for some reason you wish to cache this data elsewhere, you can set the ``XHYDRO_DATA_DIR`` environment variable to a different location before running the tests. For example, to cache the data in the current working directory, run::

            $ export XHYDRO_DATA_DIR=$(pwd)/.cache

#. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

   If ``pre-commit`` hooks fail, try re-committing your changes (or, if need be, you can skip them with `$ git commit --no-verify`).

#. Submit a `Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_ through the GitHub website.

#. When pushing your changes to your branch on GitHub, the documentation will automatically be tested to reflect the changes in your Pull Request. This build process can take several minutes at times. If you are actively making changes that affect the documentation and wish to save time, you can compile and test your changes beforehand locally with::

    # To generate the html and open it in your browser
    $ make docs
    # To only generate the html
    $ make autodoc
    $ make -C docs html
    # To simply test that the docs pass build checks
    $ tox -e docs

#. Once your Pull Request has been accepted and merged to the ``main`` branch, several automated workflows will be triggered:

    - The ``bump-version.yml`` workflow will automatically bump the patch version when pull requests are pushed to the ``main`` branch on GitHub. **It is not recommended to manually bump the version in your branch when merging (non-release) pull requests (this will cause the version to be bumped twice).**
    - `ReadTheDocs` will automatically build the documentation and publish it to the `latest` branch of `xhydro` documentation website.
    - If your branch is not a fork (ie: you are a maintainer), your branch will be automatically deleted.

You will have contributed your first changes to ``xhydro``!

.. warning::

    If your Pull Request relies on modifications to the testing data of `xhydro`, you will need to update the testing data repository as well. As a preliminary testing measure, the branch of the testing data can be modified at testing time (from `main`) by setting the ``XHYDRO_TESTDATA_BRANCH`` environment variable to the branch name of the ``xhydro-testdata`` repository.

    Be sure to consult the ReadMe found at https://github.com/hydrologie/xhydro-testdata as well.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

#. The pull request should include tests and should aim to provide `code coverage <https://en.wikipedia.org/wiki/Code_coverage>`_ for all new lines of code. You can use the ``--cov-report html --cov xhydro`` flags during the call to ``pytest`` to generate an HTML report and analyse the current test coverage.

#. If the pull request adds functionality, the docs should also be updated. Put your new functionality into a function with a docstring, and add the feature to the list in ``README.rst``.

#. The pull request should work for Python 3.9, 3.10, 3.11, and 3.12. Check that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

    $ pytest tests.test_xhydro

To run specific code style checks::

    $ black --check xhydro tests
    $ isort --check xhydro tests
    $ blackdoc --check xhydro docs
    $ ruff xhydro tests
    $ flake8 xhydro tests

To get ``black``, ``isort``, ``blackdoc``, ``ruff``, and ``flake8`` (with plugins ``flake8-alphabetize`` and ``flake8-rst-docstrings``) simply install them with ``pip`` (or ``conda``) into your environment.

Versioning/Tagging
------------------

A reminder for the **maintainers** on how to deploy. This section is only relevant when producing a new point release for the package.

.. warning::

    It is important to be aware that any changes to files found within the ``xhydro`` folder (with the exception of ``xhydro/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

#. Create a new branch from `main` (e.g. `release-0.2.0`).
#. Update the `CHANGES.rst` file to change the `Unreleased` section to the current date.
#. Bump the version in your branch to the next version (e.g. `v0.1.0 -> v0.2.0`)::

    $ bump-my-version bump minor # In most cases, we will be releasing a minor version
    $ git push

#. Create a pull request from your branch to `main`.
#. Once the pull request is merged, create a new release on GitHub. On the main branch, run::

    $ git tag v0.2.0
    $ git push --tags

   This will trigger a GitHub workflow to build the package and upload it to TestPyPI. At the same time, the GitHub workflow will create a draft release on GitHub. Assuming that the workflow passes, the final release can then be published on GitHub by finalizing the draft release.

#. Once the release is published, the `publish-pypi.yml` workflow will go into an `awaiting approval` mode on Github Actions. Only authorized users may approve this workflow (notifications will be sent) to trigger the upload to PyPI.

#. To generate the release notes, run:

    .. code-block:: python

        import xhydro.testing.utils as xhu

        print(xhu.publish_release_notes())

   This will print the release notes (taken from the `HISTORY.rst` file) to your python console. Copy and paste them into the GitHub release description, keeping only the changes for the current version.

#. Once the release is published, it will go into a `staging` mode on Github Actions. Once the tests pass, admins can approve the release (an e-mail will be sent) and it will be published on PyPI.

.. warning::

    Uploads to PyPI can **never** be overwritten. If you make a mistake, you will need to bump the version and re-release the package. If the package uploaded to PyPI is broken, you should modify the GitHub release to mark the package as broken, as well as yank the package (mark the version  "broken") on PyPI.

Packaging
---------

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the pip-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

The simple approach
~~~~~~~~~~~~~~~~~~~

The simplest approach to packaging for general support (pip wheels) requires that ``flit`` be installed::

    $ python -m pip install flit

From the command line on your Linux distribution, simply run the following from the clone's main dev branch::

    # To build the packages (sources and wheel)
    $ python -m flit build

    # To upload to PyPI
    $ python -m flit publish dist/*

The new version based off of the version checked out will now be available via `pip` (`$ pip install xhydro`).

Releasing on conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~

Initial Release
^^^^^^^^^^^^^^^

Before preparing an initial release on conda-forge, we *strongly* suggest consulting the following links:
 * https://conda-forge.org/docs/maintainer/adding_pkgs.html
 * https://github.com/conda-forge/staged-recipes

In order to create a new conda build recipe, to be used when proposing packages to the conda-forge repository, we strongly suggest using the ``grayskull`` tool::

    $ python -m pip install grayskull
    $ grayskull pypi xhydro

For more information on ``grayskull``, please see the following link: https://github.com/conda/grayskull

Before updating the main conda-forge recipe, we echo the conda-forge documentation and *strongly* suggest performing the following checks:
 * Ensure that dependencies and dependency versions correspond with those of the tagged version, with open or pinned versions for the `host` requirements.
 * If possible, configure tests within the conda-forge build CI (e.g. `imports: xhydro`, `commands: pytest xhydro`).

Subsequent releases
^^^^^^^^^^^^^^^^^^^

If the conda-forge feedstock recipe is built from PyPI, then when a new release is published on PyPI, `regro-cf-autotick-bot` will open Pull Requests automatically on the conda-forge feedstock. It is up to the conda-forge feedstock maintainers to verify that the package is building properly before merging the Pull Request to the main branch.

Building sources for wide support with `manylinux` image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    This section is for building source files that link to or provide links to C/C++ dependencies.
    It is not necessary to perform the following when building pure Python packages.

In order to do ensure best compatibility across architectures, we suggest building wheels using the `PyPA`'s `manylinux`
docker images (at time of writing, we endorse using `manylinux_2_24_x86_64`).

With `docker` installed and running, begin by pulling the image::

    $ sudo docker pull quay.io/pypa/manylinux_2_24_x86_64

From the xhydro source folder we can enter into the docker container, providing access to the `xhydro` source files by linking them to the running image::

    $ sudo docker run --rm -ti -v $(pwd):/xhydro -w /xhydro quay.io/pypa/manylinux_2_24_x86_64 bash

Finally, to build the wheel, we run it against the provided Python3.9 binary::

    $ /opt/python/cp39-cp39m/bin/python -m build --sdist --wheel

This will then place two files in `xhydro/dist/` ("xhydro-1.2.3-py3-none-any.whl" and "xhydro-1.2.3.tar.gz").
We can now leave our docker container (`$ exit`) and continue with uploading the files to PyPI::

    $ twine upload dist/*
