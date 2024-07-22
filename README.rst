======
xHydro
======

+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |versions|                                   |
+----------------------------+-----------------------------------------------------+
| Documentation and Support  | |docs|                                              |
+----------------------------+-----------------------------------------------------+
| Open Source                | |license| |ossf|                                    |
+----------------------------+-----------------------------------------------------+
| Coding Standards           | |black| |ruff| |pre-commit|                         |
+----------------------------+-----------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                        |
+----------------------------+-----------------------------------------------------+

Hydrological analysis library built with xarray

* Free software: Apache-2.0
* Documentation: https://xhydro.readthedocs.io.

Features
--------

* Easily find and extract geospacial data from the Planetary Computer API and watershed boundaries from the HydroSHEDS API over any area of interest.
* Calibrate and execute Hydrotel and Raven-emulated hydrological models.
* Perform optimal interpolation on hydrological data (daily streamflow and indices).
* Compute hydrological indicators (e.g. n-day peak flow, annual maximum series, low flow, average flow, etc.) over custom date ranges.
* Perform frequency analysis on hydrological indicators using a variety of methods (e.g. Gumbel, GEV, etc.).
* Perform climate change impact analysis of hydrological data.

Credits
-------

This package was created with Cookiecutter_ and the `Ouranosinc/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
.. _`Ouranosinc/cookiecutter-pypackage`: https://github.com/Ouranosinc/cookiecutter-pypackage

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |build| image:: https://github.com/hydrologie/xhydro/actions/workflows/main.yml/badge.svg
        :target: https://github.com/hydrologie/xhydro/actions
        :alt: Build Status

.. |coveralls| image:: https://coveralls.io/repos/github/hydrologie/xhydro/badge.svg
        :target: https://coveralls.io/github/hydrologie/xhydro
        :alt: Coveralls

..
    .. |docs| image:: https://img.shields.io/badge/Docs-GitHub.io-blue
            :target: https://hydrologie.github.io/xhydro/
            :alt: Documentation Status

.. |docs| image:: https://readthedocs.org/projects/xhydro/badge/?version=latest
        :target: https://xhydro.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. |license| image:: https://img.shields.io/pypi/l/xhydro
        :target: https://github.com/hydrologie/xhydro/blob/main/LICENSE
        :alt: License

.. |ossf| image:: https://api.securityscorecards.dev/projects/github.com/hydrologie/xhydro/badge
        :target: https://securityscorecards.dev/viewer/?uri=github.com/hydrologie/xhydro
        :alt: OpenSSF Scorecard

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/hydrologie/xhydro/main.svg
        :target: https://results.pre-commit.ci/latest/github/hydrologie/xhydro/main
        :alt: pre-commit.ci Status

.. |pypi| image:: https://img.shields.io/pypi/v/xhydro.svg
        :target: https://pypi.python.org/pypi/xhydro
        :alt: PyPI

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/xhydro.svg
        :target: https://pypi.python.org/pypi/xhydro
        :alt: Supported Python Versions
