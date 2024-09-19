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

rem Generate API documentation
sphinx-apidoc -o docs\apidoc --private --module-first src\xhydro

rem Generate Sphinx HTML documentation in English
call docs\make.bat html "-D language=en"

rem Generate Sphinx HTML documentation in French
call docs\make.bat html "-D language=fr"

if %ERRORLEVEL% neq 0 (
    echo Error occurred during documentation generation.
    exit /b %ERRORLEVEL%
)

echo Documentation generation completed successfully.
