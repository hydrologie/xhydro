#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "bottleneck>=1.3.1",
    "cartopy",
    "cf-xarray>=0.6.1",
    "cftime>=1.4.1",
    "dask[array]>=2.6",
    "geopandas",
    "h5netcdf",
    "intake-xarray>=0.6.1",
    "jsonpickle",
    "numba",
    "numpy>=1.16.0",
    "pandas>=0.23",
    "pint>=0.10",
    "pyarrow",
    "pyyaml",
    "s3fs>=2022.7.0",
    "scipy>=1.2",
    "xarray>=0.17.0",
    "xclim>=0.43.0",
    "zarr>=2.11.1",
]

test_requirements = ["pytest>=3"]

docs_requirements = [
    dependency for dependency in open("requirements_docs.txt").readlines()
]

dev_requirements = [
    dependency for dependency in open("requirements_dev.txt").readlines()
]

setup(
    author="Thomas-Charles Fortier Filion",
    author_email="tcff_hydro@outlook.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Hydrological analysis library built with xarray",
    entry_points={
        "console_scripts": [
            "xhydro=xhydro.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache-2.0",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="xhydro",
    name="xhydro",
    packages=find_packages(include=["xhydro", "xhydro.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        "docs": docs_requirements,
        "dev": dev_requirements,
    },
    url="https://github.com/TC-FF/xhydro",
    version="0.1.3",
    zip_safe=False,
)
