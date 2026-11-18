@echo off
REM One-click start for Windows: creates a venv on first run, then launches CogniSign.
cd /d "%~dp0"
if not exist .venv (
  echo First run: setting up ^(this takes a minute^)...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -e .
)
.venv\Scripts\cognisign %*
