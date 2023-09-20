#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["xarray"]

dev_requirements = ["pytest", "pytest-cov"]

setup(
    author="Thomas-Charles Fortier Filion",
    author_email="tcff_hydro@outlook.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
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
    url="https://github.com/TC-FF/xhydro",
    version="0.1.8",
    zip_safe=False,
)
