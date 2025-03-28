=========
Releasing
=========

Deployment
----------

This page is a reminder for the **maintainers** on how to deploy. This section is only relevant when producing a new point release for the package.

.. warning::

    It is important to be aware that any changes to files found within the ``src/xhydro`` folder (with the exception of ``src/xhydro/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

#. Create a new branch from `main` (e.g. `release-0.2.0`).
#. Update the `CHANGELOG.rst` file to change the `Unreleased` section to the current date.
#. Bump the version in your branch to the next version (e.g. `v0.1.0 -> v0.2.0`):

    .. code-block:: console

        bump-my-version bump minor # In most cases, we will be releasing a minor version
        bump-my-version bump release # This will update the version strings to drop the `dev` suffix
        git push origin release-0.2.0

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

   This will print the release notes (taken from the `CHANGELOG.rst` file) to your python console. Copy and paste them into the GitHub release description, keeping only the changes for the current version.

#. Once the release is published, the `publish-pypi.yml` workflow will go into an `awaiting approval` mode on Github Actions. Only authorized users may approve this workflow (notifications will be sent) to trigger the upload to PyPI.

    .. warning::

        Uploads to PyPI can **never** be overwritten. If you make a mistake, you will need to bump the version and re-release the package. If the package uploaded to PyPI is broken, you should modify the GitHub release to mark the package as broken, as well as yank the package (mark the version "broken") on PyPI.

#. A new version of `xHydro` on PyPI will trigger the `regro-cf-autotick-bot` to open a pull request on the conda-forge feedstock. This will automatically update the feedstock with the new version of the package. The feedstock maintainers will need to review and merge the pull request.
