=========
Changelog
=========

v0.3.4 (2024-02-29)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Thomas-Charles Fortier Filion (:user:`TC-FF`), Gabriel Rondeau-Genesse (:user:`RondeauG`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added French language support to the documentation. (:issue:`53`, :pull:`55`).
* Added a new set of functions to support creating and updating `pooch` registries, caching testing datasets from `hydrologie/xhydro-testdata`, and ensuring that testing datasets can be loaded into temporary directories. (:pull:`62`).
* `xhydro` is now configured to use `pooch` to download and cache testing datasets from `hydrologie/xhydro-testdata`. (:pull:`62`).
* `xhydro` is now `Semantic Versioning v2.0.0 <https://semver.org/spec/v2.0.0.html>`_ compliant. (:pull:`70`).
* Added new functions to `xhydro.frequency_analysis.local` to calculate plotting positions and to prepare plots. (:pull:`87`).
* `xscen` now supports Python3.12. (:pull:`99`).
* `xscen` now supports `pandas` >= 2.2.0, `xarray` >= 2023.11.0, and `xclim` >= 0.47.0. (:pull:`99`).
* Added `xh.cc.sampled_indicators` to compute future indicators using a perturbation approach and random sampling. (:pull:`54`).

Breaking changes
^^^^^^^^^^^^^^^^
* Added `pooch` as an installation dependency. (:pull:`62`).
* `xhydro` now requires `xarray`>=2023.11.0, `xclim`>=0.48.2, `xscen`>=0.8.3, and, indirectly, `pandas`>=2.2.0. The main breaking change is in how yearly frequencies are called ('YS-' instead of 'AS-'). (:pull:`54`).
* Functions that output a dict with keys as xrfreq (namely, ``xh.indicators.compute_indicators``) will now return the new nomenclature (e.g. "YS-JAN" instead of "AS-JAN"). (:pull:`54`).

Internal changes
^^^^^^^^^^^^^^^^
* Added a new module for testing purposes: `xhydro.testing.helpers` with some new functions. (:pull:`62`):
    * `generate_registry`: Parses data found in package (`xhydro.testing.data`), and adds it to the `registry.txt`
    * `load_registry`: Loads installed (or custom) registry and returns dictionary
    * `populate_testing_data`: Fetches the registry and optionally caches files at a different location (helpful for `pytest-xdist`).
* Added a `pre-commit` hook (`numpydoc`) to ensure that `numpy` docstrings are formatted correctly. (:pull:`62`).
* The cookiecutter has been updated to the latest commit (:pull:`70`, :pull:`106`):
    * Added some workflows (Change file labelling, Cache cleaning, Dependency scans, `OpenSSF Scorecard <https://securityscorecards.dev/>`_).
    * The README has been updated to organize badges in a table, including a badge for the OpenSSF Scorecard.
    * Updated pre-commit hook versions to the latest available.
    * Formatting tools are now pinned to their pre-commit equivalents.
    * `actions-version-updater.yml` has been replaced by `dependabot <https://docs.github.com/en/code-security/dependabot/working-with-dependabot>`_.
    * Addressed a handful of misconfigurations in the workflows.
    * Updated ruff to v0.2.0 and black to v24.2.0.
* Added a few functions missing from the API to their respective modules via ``__all__``. (:pull:`99`).

v0.3.0 (2023-12-01)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Trevor James Smith (:user:`Zeitsperre`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* The `xhydro` planification was added to the documentation. (:issue:`39`, :pull:`49`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xhydro` now adheres to PEPs 517/518/621 using the `flit` backend for building and packaging. (:pull:`50`).

Bug fixes
^^^^^^^^^
* The `return_level` dimension in `xh.frequency_analysis.local.parametric_quantiles()` is now the actual return level, not the quantile. (:issue:`41`, :pull:`43`).

Internal changes
^^^^^^^^^^^^^^^^
* Added `xhydro.testing.utils.publish_release_notes()` to help with the release process. (:pull:`37`).
* `xh.frequency_analysis.local.parametric_quantiles()` and `xh.frequency_analysis.local.criteria()` are now lazier. (:issue:`41`, :pull:`43`).
* The `cookiecutter` template has been updated to the latest commit via `cruft`. (:pull:`50`):
    * `Manifest.in` and `setup.py` have been removed.
    * `pyproject.toml` has been added, with most package configurations migrated into it.
    * `HISTORY.rst` has been renamed to `CHANGES.rst`.
    * `actions-version-updater.yml` has been added to automate the versioning of the package.
    * `bump-version.yml` has been added to automate patch versioning of the package.
    * `pre-commit` hooks have been updated to the latest versions; `check-toml` and `toml-sort` have been added to cleanup the `pyproject.toml` file.
    * `ruff` has been added to the linting tools to replace most `flake8` and `pydocstyle` verifications.

v0.2.0 (2023-10-10)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Thomas-Charles Fortier Filion (:user:`TC-FF`), Sébastien Langlois (:user:`sebastienlanglois`)

Announcements
^^^^^^^^^^^^^
* Support for Python3.8 and lower has been dropped. (:pull:`11`).
* `xHydro` now hosts its documentation on `Read the Docs <https://xhydro.readthedocs.io/en/latest/>`_. (:issue:`22`, :pull:`26`).
* Local frequency analysis functions have been added under a new module `xhydro.frequency_analysis`. (:pull:`20`, :pull:`27`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* GitHub Workflows for automated testing using `tox` have been added. (:pull:`11`).
* Support for various `xscen` functions has been added to compute indicators and various climate change metrics. (:pull:`21`).
* New function `xh.indicators.compute_volume` to convert streamflow data to volumes. (:pull:`20`, :pull:`27`).
* New function `xh.indicators.get_yearly_op` to compute block operation (e.g. block maxima, minima, etc.). (:pull:`20`, :pull:`27`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xHydro` repository has renamed its primary development branch from `master` to `main`. (:pull:`13`).
* `xHydro` now requires a conda environment to be installed. (:pull:`21`).

Bug fixes
^^^^^^^^^
* N/A

Internal changes
^^^^^^^^^^^^^^^^
* Added a Pull Request template. (:pull:`14`).
* Various updates to the autogenerated boilerplate (Ouranosinc/cookiecutter-pypackage) via `cruft`. (:pull:`11`, :pull:`12`, :pull:`13`):
    * General updates to pre-commit hooks, development dependencies, documentation.
    * Added configurations for Pull Request and Issues templates, Zenodo.
    * Documentation now makes use of sphinx directives for usernames, issues, and pull request hyperlinks (via sphinx.ext.extlinks). (:issue:`15`).
    * GitHub Workflows have been added for automated testing, and publishing.
    * Some sphinx extensions have been added/enabled (sphinx-codeautolink, sphinx-copybutton).
    * Automated testing with tox now updated to use v4.0+ conventions.
    * Removed all references to travis.ci.
* Deployments to TestPyPI and PyPI are now run using GitHub Workflow Environments as a safeguarding mechanism. (:pull:`28`).
* Various cleanups of the environment files. (:issue:`23`, :pull:`30`).
* `xhydro` now uses the trusted publishing mechanism for PyPI and TestPyPI deployment. (:pull:`32`).
* Added tests. (:pull:`27`).

0.1.2 (2023-05-10)
------------------

* First release on PyPI.
