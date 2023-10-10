#!/usr/bin/env python

"""Tests for `xhydro` package."""

import pathlib
import pkgutil

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: https://doc.pytest.org/en/latest/explanation/fixtures.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
    pass


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    pass


def test_package_metadata():
    """Test the package metadata."""
    project = pkgutil.get_loader("xhydro").get_filename()

    metadata = pathlib.Path(project).resolve().parent.joinpath("__init__.py")

    with open(metadata) as f:
        contents = f.read()
        assert """Thomas-Charles Fortier Filion""" in contents
        assert '__email__ = "tcff_hydro@outlook.com"' in contents
        assert '__version__ = "0.2.0"' in contents
