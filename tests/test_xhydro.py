#!/usr/bin/env python

"""Tests for `xhydro` package."""

from pathlib import Path

import pytest

from xhydro import xhydro


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


def test_imports():
    metadata = Path(xhydro.__file__).resolve().parent.joinpath("__init__.py")

    with open(metadata) as f:
        contents = f.read()
        assert '__author__ = """Thomas-Charles Fortier Filion"""' in contents
        assert '__email__ = "tcff_hydro@outlook.com"' in contents
        assert '__version__ = "0.1.3"' in contents
