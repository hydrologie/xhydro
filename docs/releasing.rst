=========
Releasing
=========

Deployment
----------

A reminder for the **maintainers** on how to deploy. This section is only relevant when producing a new point release for the package.

.. warning::

    It is important to be aware that any changes to files found within the ``src/xhydro`` folder (with the exception of ``src/xhydro/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

#. Create a new branch from `main` (e.g. `release-0.2.0`).
#. Update the `CHANGELOG.rst` file to change the `Unreleased` section to the current date.
#. Bump the version in your branch to the next version (e.g. `v0.1.0 -> v0.2.0`):

    .. code-block:: console

        bump-my-version bump minor # In most cases, we will be releasing a minor version
        bump-my-version bump release # This will update the version strings to drop the `dev` suffix
        git push

#. Create a pull request from your branch to `main`.
#. Once the pull request is merged, create a new release on GitHub. On the `main` branch, run:

    .. code-block:: console

        git tag v0.2.0
        git push --tags

   This will trigger a GitHub workflow to build the package and upload it to TestPyPI. At the same time, the GitHub workflow will create a draft release on GitHub. Assuming that the workflow passes, the final release can then be published on GitHub by finalizing the draft release.

#. To generate the release notes, run:

    .. code-block:: python

        import xhydro.testing.utils as xhu

        print(xhu.publish_release_notes())

   This will print the release notes (taken from the `HISTORY.rst` file) to your python console. Copy and paste them into the GitHub release description, keeping only the changes for the current version.

#. Once the release is published, the `publish-pypi.yml` workflow will go into an `awaiting approval` mode on Github Actions. Only authorized users may approve this workflow (notifications will be sent) to trigger the upload to PyPI.

.. warning::

    Uploads to PyPI can **never** be overwritten. If you make a mistake, you will need to bump the version and re-release the package. If the package uploaded to PyPI is broken, you should modify the GitHub release to mark the package as broken, as well as yank the package (mark the version "broken") on PyPI.

Packaging
---------

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the pip-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

The simple approach
~~~~~~~~~~~~~~~~~~~

The simplest approach to packaging for general support (pip wheels) requires that `flit` be installed:

    .. code-block:: console

        python -m pip install flit

From the command line on your Linux distribution, simply run the following from the clone's main dev branch:

    .. code-block:: console

        # To build the packages (sources and wheel)
        make dist

        # To upload to PyPI
        make release

The new version based off of the version checked out will now be available via `pip` (`pip install xhydro`).

Releasing on conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~

Initial Release
^^^^^^^^^^^^^^^

Before preparing an initial release on conda-forge, we *strongly* suggest consulting the following links:
 * https://conda-forge.org/docs/maintainer/adding_pkgs.html
 * https://github.com/conda-forge/staged-recipes

In order to create a new conda build recipe, to be used when proposing packages to the conda-forge repository, we strongly suggest using the `grayskull` tool::

    .. code-block:: console

        python -m pip install grayskull
        grayskull pypi xhydro

For more information on `grayskull`, please see the following link: https://github.com/conda/grayskull

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

In order to do ensure best compatibility across architectures, we suggest building wheels using the `PyPA`'s `manylinux` docker images (at time of writing, we endorse using `manylinux_2_24_x86_64`).

With `docker` installed and running, begin by pulling the image:

    .. code-block:: console

        sudo docker pull quay.io/pypa/manylinux_2_24_x86_64

From the xHydro source folder we can enter into the docker container, providing access to the `src/xhydro` source files by linking them to the running image:

    .. code-block:: console

        sudo docker run --rm -ti -v $(pwd):/src/xhydro -w /src/xhydro quay.io/pypa/manylinux_2_24_x86_64 bash

Finally, to build the wheel, we run it against the provided Python3.9 binary:

    .. code-block:: console

        /opt/python/cp39-cp39m/bin/python -m build --sdist --wheel

This will then place two files in `xhydro/dist/` ("xhydro-1.2.3-py3-none-any.whl" and "xHydro-1.2.3.tar.gz").
We can now leave our docker container (`exit`) and continue with uploading the files to PyPI:

    .. code-block:: console

        python -m twine upload dist/*
