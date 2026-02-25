.PHONY: clean clean-build clean-pyc clean-test coverage development dist docs help install lint release test
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
LOCALES := docs/locales

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-docs clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove docs artifacts
	rm -fr docs/notebooks/_data/
	rm -fr docs/notebooks/.ipynb_checkpoints/
	rm -f docs/apidoc/xhydro*.rst
	rm -f docs/apidoc/modules.rst
	rm -f docs/locales/fr/LC_MESSAGES/*.mo
	$(MAKE) -C docs clean

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr .pytest_cache
	rm -fr .tox/
	rm -fr htmlcov/

install-lint: ## install dependencies needed for linting
	python -m pip install --quiet --group lint
 
install-docs: ## install dependencies needed for building the docs
	python -m pip install --quiet --group docs

install-test: ## install dependencies needed for standard testing
	python -m pip install --quiet --group test

install-tox: ## install base dependencies needed for running tox
	python -m pip install --quiet --group tox


lint: install-lint ## check style
	python -m ruff check src/xhydro tests
	python -m flake8 --config=.flake8 src/xhydro tests
	python -m numpydoc lint src/xhydro/**.py
	python -m vulture src/xhydro tests
	codespell src/xhydro tests docs
	python -m deptry src
	python -m yamllint --config-file=.yamllint.yaml src/xhydro

test: install-test ## run tests quickly with the default Python
	python -m pytest

test-distributed: ## run tests quickly with the default Python and distributed workers
	python -m pytest --num-processes=logical

test-notebooks: ## run tests on notebooks and compare outputs
	pytest --no-cov --nbval --rootdir=tests/ docs/notebooks

test-notebooks-lax: ## run tests on notebooks but don't be so strict about outputs
	pytest --no-cov --nbval-lax --rootdir=tests/ docs/notebooks

test-notebooks-lax-onlyextremes: ## run tests exclusively on the Julia notebook
	pytest --no-cov --nbval-lax --rootdir=tests/ docs/notebooks/extreme_value_analysis.ipynb

test-notebooks-lax-noextremes: ## run tests on notebooks but don't be so strict about outputs
	pytest --no-cov --nbval-lax --rootdir=tests/ docs/notebooks --ignore='docs/notebooks/extreme_value_analysis.ipynb'

test-all: install-tox ## run tests on every Python version with tox
	python -m tox

coverage: install-test ## check code coverage quickly with the default Python
	python -m coverage run --source src/xhydro -m pytest
	python -m coverage report -m
	python -m coverage html
	$(BROWSER) htmlcov/index.html

autodoc: install-docs clean-docs ## create sphinx-apidoc files:
	sphinx-apidoc -o docs/apidoc --private --module-first src/xhydro

initialize-translations: clean-docs autodoc ## initialize translations, including autodoc-generated files (but not the API docs)
	env SKIP_NOTEBOOKS=1 ${MAKE} -C docs gettext
	sphinx-intl update -p docs/_build/gettext -d docs/locales -l fr
	rm -fr docs/locales/fr/LC_MESSAGES/apidoc

linkcheck: autodoc ## run checks over all external links found throughout the documentation
	$(MAKE) -C docs linkcheck

build-docs: autodoc ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html BUILDDIR="_build/html/en"
ifneq ("$(wildcard $(LOCALES))","")
	${MAKE} -C docs gettext
	$(MAKE) -C docs html BUILDDIR="_build/html/fr" SPHINXOPTS="-D language='fr'"
endif

docs: build-docs  ## open the built documentation in a web browser
ifndef READTHEDOCS
 	$(BROWSER) docs/_build/html/en/html/index.html
endif
servedocs: autodoc ## compile the docs while watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m flit build
	ls -l dist

release: dist ## package and upload a release
	python -m flit publish dist/*

install: clean ## install the package to the active Python's site-packages
	python -m pip install --no-user .

development: clean ## install the package to the active Python's site-packages
	python -m pip install --group dev
	python -m pip install --no-user --editable .[extras]
	prek install
