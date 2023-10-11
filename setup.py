#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "statsmodels", "xarray", "xclim>=0.45.0", "xscen"]

dev_requirements = ["pytest", "pytest-cov"]

setup(
    author="Thomas-Charles Fortier Filion",
    author_email="tcff_hydro@outlook.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    description="Hydrological analysis library built with xarray",
    entry_points={
        "console_scripts": [
            "xhydro=xhydro.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="xhydro",
    name="xhydro",
    packages=find_packages(include=["xhydro", "xhydro.*"]),
    test_suite="tests",
    extras_require={
        "dev": dev_requirements,
    },
    url="https://github.com/hydrologie/xhydro",
    version="0.2.0",
    zip_safe=False,
)
