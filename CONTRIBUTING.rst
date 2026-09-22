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
* Remember that this is a volunteer-driven project, and that contributions are welcome. :)

Get Started!
------------

.. note::

    If you are new to using `GitHub <https://github.com/>`_ and ``git``, please read `this guide <https://guides.github.com/activities/hello-world/>`_ first.

.. warning::

    Anaconda Python users: Due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment. Consider running the following commands in order to speed up the process:

    .. code-block:: console

        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba

    For more information, please see the following link: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

    Alternatively, you can use the `mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ package manager, which is a drop-in replacement for ``conda``. If you are already using `mamba`, replace the following commands with ``mamba`` instead of ``conda``.

Ready to contribute? Here's how to set up ``xHydro`` for local development.

#. First, clone the ``xHydro`` repo locally.

   * If you are not a ``xHydro`` collaborator, first fork the ``xHydro`` repo on GitHub, then clone your fork locally.

   .. code-block:: console

       $ git clone git@github.com:your_name_here/xHydro.git

   * If you are a ``xHydro`` collaborator, clone the ``xHydro`` repo directly.

   .. code-block:: console

       $ git clone git@github.com:hydrologie/xHydro.git

#. Install your local copy into a development environment. You can create a new Anaconda development environment with:

   .. code-block:: console

       $ conda env create -f environment-dev.yml
       $ conda activate xhydro-dev
       $ make dev

   If you are on Windows, replace the ``make dev`` command with the following:

   .. code-block:: console

       $ python -m pip install --group dev
       $ python -m pip install --editable .
       $ prek install

   This installs ``xhydro`` in an "editable" state, meaning that changes to the code are immediately seen by the environment. To ensure a consistent coding style, `make dev` also installs the ``pre-commit`` hooks to your local clone. It also installs all the libraries necessary to run the ``Extremes.jl`` hooks of the ``extreme_value_analysis`` module.

   On commit, ``prek`` will run ``pre-commit`` checks that ensure code quality checks are passing, perform automatic fixes if possible, and warn of violations that require intervention. If your commit fails the checks initially, simply fix the errors, re-add the files, and re-commit.

   You can also run the hooks manually with:

   .. code-block:: console

       $ prek run -a

   If you want to skip the ``pre-commit`` hooks temporarily, you can pass the `--no-verify` flag to `git commit`.

#. Create a branch for local development:

   .. code-block:: console

       $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, we **strongly** suggest running the tests in your environment or with the help of ``tox``:

   .. code-block:: console

       $ make lint
       $ python -m pytest
       # Or, to run multiple build tests
       $ python -m tox

   .. note::

      Running `pytest` or `tox` will automatically fetch and cache the testing data for the package to your local cache (using the `platformdirs` library). On Linux, this is located at ``XDG_CACHE_HOME`` (usually ``~/.cache``). On Windows, this is located at ``%LOCALAPPDATA%`` (usually ``C:\Users\username\AppData\Local``). On MacOS, this is located at ``~/Library/Caches``.

      If for some reason you wish to cache this data elsewhere, you can set the ``XHYDRO_DATA_DIR`` environment variable to a different location before running the tests. For example, to cache the data in the current working directory, run:

           export XHYDRO_DATA_DIR=$(pwd)/.cache

#. Commit your changes and push your branch to GitHub:

   .. code-block:: console

       $ git add .
       $ git commit -s -m "Your detailed description of your changes."
       $ git push origin name-of-your-bugfix-or-feature

   On commit, ``prek`` will run ``pre-commit`` checks that ensure code quality checks are passing, perform automatic fixes if possible, and warn of violations that require intervention. If your commit fails the checks initially, simply fix the errors, re-add the files, and re-commit.

   You can always run the hooks manually with:

   .. code-block:: console

       $ pre-commit run -a

   If you want to skip the ``pre-commit`` hooks temporarily, you can pass the `--no-verify` flag to `git commit`.

   If ``pre-commit`` hooks fail, try fixing the issues, re-staging the files to be committed, and re-committing your changes (or, if need be, you can skip them with `--no-verify` flag).


#. Submit a `Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_ through the GitHub website.

#. When pushing your changes to your branch on GitHub, the documentation will automatically be tested to reflect the changes in your Pull Request. This build process can take several minutes at times. If you are actively making changes that affect the documentation and wish to save time, you can compile and test your changes beforehand locally with:

   .. code-block:: console

       # To generate the html and open it in your browser
       $ make docs
       # To only generate the html
       $ make autodoc
       $ make -C docs html
       # To simply test that the docs pass build checks
       $ python -m tox -e docs

#. If changes to your branch are made on GitHub, you can update your local branch with:

   .. code-block:: console

       $ git checkout name-of-your-bugfix-or-feature
       $ git fetch
       $ git pull origin name-of-your-bugfix-or-feature

   If you have merge conflicts, you might need to replace `git pull` with `git merge` and resolve the conflicts manually.
   Resolving conflicts from the command line can be tricky. If you are not comfortable with this, you can ignore the last command and instead use a GUI like PyCharm or Visual Studio Code to merge the remote changes and resolve the conflicts.

#. Before merging, your Pull Request will need to be based on the `main` branch of the ``xHydro`` repository. If your branch is not up-to-date with the `main` branch, you can perform similar steps as above to update your branch:

   .. code-block:: console

       $ git checkout name-of-your-bugfix-or-feature
       $ git fetch
       $ git pull origin main

   See the previous step for more information on resolving conflicts.

#. To prevent unnecessary testing of branches that are not ready for review, the `xHydro` repository is set up to run tests only when a Pull Request has been "approved" by a maintainer. Similarly, the notebooks within documentation will only be rebuilt when the Pull Request is "approved", or if the Pull Request makes explicit changes to them. As such, additional changes to the Pull Request might be required after the Pull Request is approved to ensure that the tests pass and the documentation can be built.

#. Once your Pull Request has been accepted and merged to the `main` branch, several automated workflows will be triggered:

   - The ``bump-version.yml`` workflow will automatically bump the patch version when pull requests are pushed to the `main` branch on GitHub. **It is not recommended to manually bump the version in your branch when merging (non-release) pull requests (this will cause the version to be bumped twice).**
   - `ReadTheDocs` will automatically build the documentation and publish it to the `latest` branch of `xhydro` documentation website.
   - If your branch is not a fork (i.e. you are a maintainer), your branch will be automatically deleted.

You will have contributed to ``xHydro``!

.. warning::

    If your Pull Request relies on modifications to the testing data of `xHydro`, you will need to update the testing data repository as well. As a preliminary testing measure, the branch of the testing data can be modified at testing time (from `main`) by setting the ``XHYDRO_TESTDATA_BRANCH`` environment variable to the branch name of the ``xhydro-testdata`` repository.

    Be sure to consult the ReadMe found at https://github.com/hydrologie/xhydro-testdata as well.

Pull Request Guidelines
-----------------------

Before you submit a Pull Request, check that it meets these guidelines:

#. The Pull Request should include tests and should aim to provide `code coverage <https://en.wikipedia.org/wiki/Code_coverage>`_ for all new lines of code. You can use the `--cov-report html --cov xhydro` flags during the call to ``pytest`` to generate an HTML report and analyse the current test coverage.

#. All functions should be documented with `docstrings` following the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format.

#. If the Pull Request adds a functionality, either update the documentation or create a new notebook that demonstrates the feature. Library-defining features should also be listed in ``README.rst``.

#. The ChangeLog should be updated with a brief description of the changes made in the Pull Request. If this is your first contribution to the project, please add your name and information to the `AUTHORS.rst` and `.zenodo.json` files.

#. The pull request should work for all currently supported Python versions. Check the `pyproject.toml` or `tox.toml` files for the list of supported versions.

.. #. If you haven't already, ensure that you have read and agreed to the `Developer Certificate of Origin (DCO) <https://developercertificate.org/>`_, and that you have signed off on your commits using:

..    .. code-block:: console

..        $ git commit -s/--signoff

.. This will add a `Signed-off-by:` line to your commit message, which indicates that you agree to the DCO.

AI Assistance Notice
--------------------

.. note::

    The following information has been adapted from AI assistance usage policies developed by `Avocado <https://avocado-framework.readthedocs.io/en/latest/guides/contributor/chapters/ai_policy.html>`_ and `ghostty <https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md>`_
    with suggestions from software developers and maintainers at Ouranos Inc.

What is AI Assistance ?
~~~~~~~~~~~~~~~~~~~~~~~

Within the context of xHydro development, AI assistance constitutess any code, code snippets, configurations, documentation, or other programmatic assets produced, suggested, or significantly modified by an artificial intelligence model, tool, or service.

AI assistance can also include code that is heavily adapted from or influenced by outputs from prompts made to generative artificial intelligence models, tools, or services.

Usage Disclosure
~~~~~~~~~~~~~~~~

+If you are using any kind of AI assistance while contributing to xHydro, **this must be disclosed in the pull request**, along with the extent to which AI assistance was used (e.g. documentation, code generation, review comments).
+Using AI to help draft responses is allowed only if the contributor meaningfully reviews, edits, and takes full responsibility for the content.
+Submitting AI-generated replies without human verification or understanding is not permitted.
+Misrepresenting AI-generated content as purely human-authored is considered a violation of the contribution guidelines.

Some example disclosures:

.. admonition:: Pull Request Description

    ### Other information:

    * This PR was primarily written using a combination of OpenClaw (Kimi K2.5) and ChatGPT (GPT-o4-mini). After using these tools, I reviewed the content.

    <!-- or -->

    * This PR was primarily written by me with small code snippets copy-pasted from chats with Claude (Sonnet 4.6).

    <!-- or -->

    * I asked GitHub Copilot to clean up and improve code I had written. I then reviewed the final product.

As a small exception, trivial tab-completion doesn't need to be disclosed, so long as it is limited to single keywords or short phrases.

For commits originating exclusively from AI, we also ask that their commits are signed with "Assisted-by: <tool name>" disclosure (this demand is repeated in `AGENTS.md`):

.. admonition:: Commit Message

    Added new method that implements the ReallyCoolNewFeature algorithm.

    Assisted-by: Claude Opus 4.5

Failing to disclose the use of artificial intelligence models, tools, or services is disrespectful to the developers and maintainers of open source software projects.
We ask that contributors are forthcoming with this information so that the effort and scrutiny applied to Pull Request reviews can be adjusted accordingly.

Code Review
~~~~~~~~~~~

All AI-assisted code contributions should be reviewed by the contributor before opening a Pull Request, except for minor or marginal changes, which may not require that step.
All contributions should still be reviewed by at least one human reviewer before approval and merging into the `main` branch.

Low Effort Contributions
~~~~~~~~~~~~~~~~~~~~~~~~

AI-generated low-effort contributions are unacceptable; Human contributors must be capable of explaining their changes, responding to review comments, and precisely adapting their contributions as requested.

Disclaimer
~~~~~~~~~~

The xHydro maintainers reserve the right to reject Pull Requests and contributions for many different reasons (functionality out-of-scope, code quality issues, non-adherence to code of conduct, etc.)
and this extends to code that has not been adequately disclosed as being assisted by artificial intelligence models, tools, or services.

Tips
----

To run a subset of tests:

.. code-block:: console

    $ python -m pytest tests/test_xhydro.py

You can also directly call a specific test class or test function using:

.. code-block:: console

    $ python -m pytest tests/test_xhydro.py::TestClassName::test_function_name

For more information on running tests, see the `pytest documentation <https://docs.pytest.org/en/latest/usage.html>`_.

Translations
------------

If you would like to contribute to the French translation of the documentation, you can do so by running the following command:

.. code-block:: console

    $ make initialize-translations

This will create or update the French translation files in the `docs/locales/fr/LC_MESSAGES` directory. You can then edit the `.po` files in this directory to provide translations for the documentation.

For convenience, you can simply run the following command:

.. code-block:: console

    $ make translate

Alternatively, you can use the `translator.py` script located in the `CI` directory to automatically translate the English documentation to French, which uses Google Translate by default. Note that this script requires the `deep-translator` package to be installed in your environment.

.. code-block:: console

    $ pip install deep-translator
    $ python scripts/translator.p

We aim to automate this process eventually but until then, we want to keep the French translation up-to-date with the English documentation at least when a new release is made.

Code of Conduct
---------------

Please note that this project is released with a `Contributor Code of Conduct <https://github.com/hydrologie/xhydro/blob/main/CODE_OF_CONDUCT.md>`_.
By participating in this project you agree to abide by its terms.
