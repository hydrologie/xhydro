@echo off
rem Batch script to handle translation work

rem Remove French locale .mo files
del /q docs\locales\fr\LC_MESSAGES\*.mo

rem Generate gettext files
sphinx-build -b gettext docs docs\_build\gettext

rem Run sphinx-intl update
sphinx-intl update -p docs\_build\gettext -d docs\locales -l fr

rem Remove API translation files
rmdir /s /q docs\fr\LC_MESSAGES\apidoc

if %ERRORLEVEL% neq 0 (
    echo Error occurred during translation update.
    exit /b %ERRORLEVEL%
)

echo Translation update completed successfully.
