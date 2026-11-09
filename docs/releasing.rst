=========
Releasing
=========

Deployment
----------

This page is a reminder for the **maintainers** on how to deploy. This section is only relevant when producing a new point release for the package.

.. warning::

    It is important to be aware that any changes to files found within the ``src/xhydro`` folder (with the exception of ``src/xhydro/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

.. note::

    Before beginning, it is a good idea to practice `commit signing`_ at the very least for _tags_ and _releases_. This is a method of cryptographically signing your work to ensure that packages can verifiably be traced to a known and trusted developer.
    Once a GPG key has been `created`_ and `added`_ to your GitHub profile, you can enable GPG signing for all commits by launching `$ git config --local/--global commit.gpgsign true` within a shell running in your local clone (or anywhere if `--global`).

Before beginning, maintainers must have already configured `Trusted Publisher`_ on GitHub_ with TestPyPI_ and PyPI_ and have also added the necessary tokens to their CI secrets (this requires active accounts on both indexes as well as GitHub).

For creating tags and releases, we **strongly recommend** enabling the "release immutability" option in the repository settings. This prevents _tags_ and _releases_ from being modified of GitHub after they have been published.

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the pip-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

From a new branch (e.g. `prepare-v123`), open a Pull Request and make sure all your changes to support a new version are committed (**update the entry for newest version in CHANGELOG.rst**), then run:

   .. code-block:: console

       $ bump-my-version bump <option>  # Possible options: major / minor / patch
       $ bump-my-version bump release # This will update the version strings to drop the `-dev` suffix
       $ git push

#. Create a pull request from your branch to `main`.
#. Once the pull request is merged, create a new release on GitHub. On the `main` branch, run:

   .. code-block:: console

       $ git tag -s/--sign "v0.2.0" -m "Release v0.2.0"
       $ git push --tags

   This will trigger a GitHub workflow to build the package and upload it to TestPyPI. At the same time, the GitHub workflow will create a draft release on GitHub. Assuming that the workflow passes, the final release can then be published on GitHub by finalizing the draft release.

#. To generate the release notes, run:

    .. code-block:: python

        import xhydro.testing.utils as xhu

        print(xhu.publish_release_notes())

   This will print the release notes (taken from the `CHANGELOG.rst` file) to your python console. Copy and paste them into the GitHub release description, keeping only the changes for the current version.

#. Once the release is **published**, the `publish-pypi.yml` workflow will go into an `awaiting approval` mode on Github Actions. Only authorized users may approve this workflow (notifications will be sent) to trigger the upload to PyPI.

   .. warning::

       Uploads to PyPI (and releases on GitHub if using release immutability) can **never** be overwritten. If you make a mistake, you will need to bump the version and re-release the package. If the package uploaded to GitHub and PyPI is broken, you should modify the GitHub release to mark the package as broken, as well as yank the package (mark the version "broken") on PyPI.

#. A new version of `xHydro` on PyPI will trigger the `regro-cf-autotick-bot` to open a pull request on the conda-forge feedstock. This will automatically update the feedstock with the new version of the package. The feedstock maintainers will need to review and merge the pull request.

.. _`commit signing`: https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work
.. _GitHub: https://github.com/
.. _`Trusted Publisher`: https://docs.pypi.org/trusted-publishers/
.. _added: https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account
.. _created: https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key
.. _PyPI: https://pypi.org/
.. _TestPyPI: https://test.pypi.org/
