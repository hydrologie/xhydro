#!/usr/bin/env python
"""Tests for `xhydro` package."""

import pathlib
from importlib.util import find_spec

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
    project = find_spec("xhydro").submodule_search_locations[0]

    metadata = pathlib.Path(project).resolve().joinpath("__init__.py")

    with metadata.open() as f:
        contents = f.read()
        assert """Thomas-Charles Fortier Filion""" in contents
        assert '__email__ = "tcff_hydro@outlook.com"' in contents
        assert '__version__ = "0.4.0"' in contents
