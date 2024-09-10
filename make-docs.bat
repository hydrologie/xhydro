@echo off
rem Batch script to clean docs artifacts and generate Sphinx HTML documentation

rem Remove docs/notebooks/_data/
if exist docs\notebooks\_data\ (
    rd /s /q docs\notebooks\_data\
)

rem Remove docs/notebooks/.ipynb_checkpoints/
if exist docs\notebooks\.ipynb_checkpoints\ (
    rd /s /q docs\notebooks\.ipynb_checkpoints\
)

rem Remove specific rst files from docs/apidoc
del /q docs\apidoc\xhydro*.rst
del /q docs\apidoc\modules.rst

rem Remove French locale .mo files
del /q docs\locales\fr\LC_MESSAGES\*.mo

rem Generate API documentation
sphinx-apidoc -o docs\apidoc --private --module-first src\xhydro

rem Generate gettext files
sphinx-build -b gettext docs docs\_build\gettext

rem Run sphinx-intl update
sphinx-intl update -p docs\_build\gettext -d docs\locales -l fr

rem Generate Sphinx HTML documentation in English
nmake -C docs html BUILDDIR=_build\html\en

rem Generate Sphinx HTML documentation in French
nmake -C docs html BUILDDIR=_build\html\fr SPHINXOPTS=-D language=fr

if %ERRORLEVEL% neq 0 (
    echo Error occurred during documentation update.
    exit /b %ERRORLEVEL%
)

echo Documentation generation completed successfully.
