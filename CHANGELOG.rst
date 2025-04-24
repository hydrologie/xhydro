=========
Changelog
=========

v0.5.0 (2025-04-24)
-------------------
Contributors to this version: Thomas-Charles Fortier Filion (:user:`TC-FF`) Gabriel Rondeau-Genesse (:user:`RondeauG`), Trevor James Smith (:user:`Zeitsperre`), Julián Ospina (:user:`ospinajulian`), Essi Parent (:user:`essicolo`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added a new module `xhydro.extreme_value_analysis` that wraps the `Extremes.jl` package for Julia. (:pull:`175`).
  * The module provides functions to fit extreme value distributions to data, compute return levels, and get uncertainty estimates.
  * The module is designed to be optional and requires the user to have Julia installed with the `Extremes.jl` package, along with the `PyJuliaCall` package for Python.
  * You can use `pip install xhydro[julia]` to install the required dependencies.
* Multiple improvements to the documentation. (:pull:`274`, :pull:`279`, :pull:`293`).

Bug fixes
^^^^^^^^^
* Patched the outputs of `xh.optimal_interpolation.execute` to remove a superfluous `station` dimension and to ensure that the `time` dimension has coordinates. (:pull:`274`).
  * Note that this change does not fix the underlying issue with the code, which will be addressed in a future release.
* Added attributes to variables instead of global attributes in `xh.extreme_value_analysis`. Modified dimension names and introduced a new dimension, `return_period`, to the results of `xh.extreme_value_analysis.return_level()`. (:pull:`283`).

Breaking changes
^^^^^^^^^^^^^^^^
* The ``xh.cc.sampled_indicators`` function has been separated into two functions: ``xh.cc.weighted_random_sampling`` and ``xh.cc.sampled_indicators``. (:pull:`240`).
  * Many of the arguments and outputs have been renamed or reorganized. Since no one was using this function yet AFAIK, no backward compatibility has been maintained.
* The default `use_defaults` argument of Hydrotel has been changed from 'True' to 'False'. (:pull:`233`).
* Multiple functions in the `xhydro.frequency_analysis` and `xhydro.indicators.pmp` modules have been updated to require literal arguments instead of positional arguments. (:pull:`274`).

Internal changes
^^^^^^^^^^^^^^^^
* `"land_use_classification"` default collection has been changed to `"io-lulc-annual-v02"`, as the previous one will be deprecated in December 2024. (:pull:`227`).
* Added some collection, year, resolution and history attributes to `xarray` output of `"land_use_classification"`. (:pull:`227`).
* Added a "User-Agent" to fix an issue related to `pooch` calls in the notebooks for recent ReadTheDocs builds. (:pull:`231`).
* Patched the ``xhydro.testing.helpers.devereaux()`` function to add a "User-Agent" by default. (:pull:`234`).
* Fixed the URL joining logic of the ``load_registry()`` and ``devereaux()`` functions in the `xhydro.testing.helpers` module. (:pull:`234`).
* Updated ``uncertainties.calc_q_iter()`` to allow for ungaged catchments. (:pull:`235`).
* Updated the cookiecutter template. (:pull:`252`):
    * GitHub Actions versions and Python dependencies updated.
    * Added pre-commit hooks for `vulture` (find dead code), `codespell` (spelling mistakes), `gitleaks` (token commit prevention), and `zizmor` (workflow security).
    * Added an advanced CodeQL workflow for security scanning.
* Adjusted a handful of dependency pins to handle the latest version of `sphinx`. (:pull:`258`).
* The internal ``xh.cc._weighted_sampling`` function has been almost entirely rewritten to hopefully be more efficient. Results should be the same as before. (:pull:`240`).
* Reduced the number of tests performed on Notebooks. (:pull:`267`).
* Removed ``_fix_dates`` from `_hydrotel` module since it's not relevant and likely to generate errors. (:pull:`233`).
* Updated and fixed many dependencies. (:pull:`295`).
* Added a workflow to test `pip` installations of `xhydro`. (:pull:`295`).
* Removed steps involving ESMF installation from the `Makefile` and the `tox` configuration. (:pull:`295`).

v0.4.1 (2024-11-07)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Trevor James Smith (:user:`Zeitsperre`), Louise Arnal (:user:`lou-a`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* A `periods` parameter was added to ``frequency_analysis.local.fit`` to compute multiple separate periods with a single call. (:pull:`219`).
* In ``xhydro.cc.sampled_indicators``, the `delta_type` argument can now be a dictionary or None, in which case the attribute `delta_kind` is used. (:pull:`220`).
* In ``xhydro.cc.sampled_indicators``, weights along a `time` or `horizon` dimension will no longer reduce that dimension. (:pull:`220`).

Bug fixes
^^^^^^^^^
* Fixed a bug in `xhydro.modelling.format_input` where the function would fail if the input data was a `dask` array. (:pull:`214`).
* The `executable` parameter in the Hydrotel model class is now always required. (:pull:`214`).

Breaking changes
^^^^^^^^^^^^^^^^
* The `xhydro` testing utilities have been rewritten to use `pooch` for downloading and caching testing datasets from `hydrologie/xhydro-testdata`. (:pull:`212`).
* The `xhydro` testing utilities now require `pytest-xdist` as a development dependency. (:pull:`212`).
* Many core dependencies have been updated to more modern versions. (:pull:`218`, :pull:`228`).
* The `delta_type` argument in ``xhydro.cc.sampled_indicators`` has been renamed to `delta_kind` and is no longer positional. (:pull:`220`).
* The ``xhydro.pmp`` module has been moved to ``xhydro.indicators.pmp``. (:pull:`226`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated the notebooks to reduce the load on ReadTheDocs. (:pull:`211`).
* Tests can now be run using the `pytest-xdist` plugin for distributed testing. See the `pytest-xdist documentation <https://pytest-xdist.readthedocs.io/en/stable/>`_ for more information. (:pull:`212`).
* Several tests reliant on online servers and services have been marked as `online` to prevent them from running in contexts where internet access is limited. (:pull:`212`).
* Many function docstrings and type hints have been updated for accuracy and precision. (:pull:`212`).
* The `xHydro` project now has a set of logos for use in documentation, the readme, and other materials. (:pull:`217`).
* ``xhydro.indicators`` is now a package with submodules for different types of indicators. Previous functions have been moved to a `generic` submodule and are imported at the top level for backwards compatibility. (:pull:`226`).
* ``xh.testing.utils.publish_release_notes`` now has a `latest` parameter to print only the latest release notes. (:pull:`228`).

v0.4.0 (2024-10-04)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Thomas-Charles Fortier Filion (:user:`TC-FF`), Julián Ospina (:user:`ospinajulian`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `xhydro` now supports `RavenPy` v0.15.0 (`RavenHydroFramework` v3.8.1). (:pull:`161`).
* Regional frequency analysis functions as well as Resampling function for uncertainties have been added to the ``xhydro.frequency_analysis`` module. (:pull:`186`).
* New function ``xhydro.modelling.format_input`` to format CF-compliant input data for hydrological models (currently only supports Hydrotel). (:pull:`185`).
* `xhydro` now has a `pmp` module to compute the Probable Maximum Precipitation from climate change scenarios. (:pull:`176`).

Internal changes
^^^^^^^^^^^^^^^^
* `numpy` has been pinned below v2.0.0 until `xclim` and other dependencies are updated to support it. (:pull:`161`).
* A helper script has been added in the `CI` directory to facilitate the translation of the `xhydro` documentation. (:issue:`63`, :pull:`163`).
* The `conda` environment now relies on the newly created `xdatasets` package. (:pull:`164`).
* The cookiecutter has been updated to the latest commit. Changes include workflow fixes, stricter coding standards, and many small adjustments to the documentation. (:pull:`164`).
* A previously uncaught YAML formatting issue has been addressed. Stricter style conventions are now enforced. (:pull:`174`).
* Chunking was adjusted in a few functions to work with the new requirements of ``apply_ufunc``. (:pull:`180`).
* Updated the cookiecutter template to the latest commit. (:pull:`177`):
    * Actions have been updated and synchronized.
    * Warnings in Pull Requests from forks are now less buggy.
    * A new pre-commit hook and linting step for validating numpy docstrings has been added (`numpydoc`).
    * All `pip`-based dependencies used to run in CI are now managed by a ``CI/requirements_ci.txt`` that uses hashes of packages for security.
* Added two new Batch (`.bat`) files to help facilitate the translation of and the generation of the `xhydro` documentation in Windows environments. (:pull:`196`).
* The bumpversion workflow now uses the Hydrologie Helper Bot to make signed commits. (:pull:`199`).
* Updated the cookiecutter template to the latest commit. (:pull:`199`):
    * Updated development dependencies to the latest versions.
    * Staged support for Python3.13.
    * Added environment caching to existing workflows.

Breaking changes
^^^^^^^^^^^^^^^^
* `xhydro` now requires `python` >= 3.10. (:pull:`195`).

v0.3.6 (2024-06-10)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Richard Arsenault (:user:`richardarsenault`), Sébastien Langlois (:user:`sebastienlanglois`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added support for the Hydrotel hydrological model. (:pull:`18`).
* Added support for various hydrological models emulated through the Raven hydrological framework. (:pull:`128`).
* Added optimal interpolation functions for time-series and streamflow indicators. (:pull:`88`, :pull:`129`).
* Added optimal interpolation notebooks. (:pull:`123`).
* Added surface properties (elevation, slope, aspect ratio) to the `gis` module. (:pull:`151`).

Breaking changes
^^^^^^^^^^^^^^^^
* Hydrological models are now classes instead of functions and dictionaries. (:issue:`93`, :pull:`18`).
* `xhydro` now uses a `'src' layout <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout>`_ for the package. (:pull:`147`).

Internal changes
^^^^^^^^^^^^^^^^
* Tests using the `gamma` distribution were changed to the `gumbel_r` to avoid changes in `xclim v0.49.0`. (:pull:`145`).
* The cookiecutter template has been updated to the latest commit. Changes include the addition of a `CODE_OF_CONDUCT.rst` file, the renaming of `CHANGES.rst` to `CHANGELOG.rst`, and many small adjustments to the documentation. (:pull:`147`).
* Added a CODE_OF_CONDUCT.rst file with Contributor Covenant guidelines. (:pull:`147`).

v0.3.5 (2024-03-20)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Thomas-Charles Fortier Filion (:user:`TC-FF`), Sébastien Langlois (:user:`sebastienlanglois`), Gabriel Rondeau-Genesse (:user:`RondeauG`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `xhydro` has implemented a `gis` module that facilitates geospatial tasks needed for gathering hydrological inputs. (:issue:`60`, :pull:`61`).

Internal changes
^^^^^^^^^^^^^^^^
* Added a workflow based on `actions/labeler` to automatically label Pull Requests based on files changed. (:pull:`68`).
* Added a conditional trigger to the `test-notebooks` job to run in advance of pull request approval in the event that the notebooks found within `docs/notebooks` have been modified (labeled `"notebooks"`). (:pull:`68`).
* Significant changes to the Continuous Integration (CI) setup. (:pull:`65`):
    * Added a workflow configuration using ``label_on_approval.yml`` and modifications of ``main.yml`` so that fewer tests are run on Pull Requests before they are fully approved.
    * Added some `pre-commit` configurations to both clean up the code within notebooks (`NbQA`) and strip their outputs (`nbstripout`).
    * `tox` is now fully v4.0-compliant.
    * Added a `Makefile` recipe to facilitate installation of `esmpy` when `esmf` is installed and visible on the `$PATH`.
    * Added a `Makefile` recipe for running tests over Jupyter notebooks.
    * Synchronized dependencies between `pyproject.toml` and `conda` configuration files.
* Moved the notebooks under a Usage section in the documentation. (:issue:`114`, :pull:`118`).

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
