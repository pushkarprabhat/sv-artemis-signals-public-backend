@echo off
chcp 65001 >nul
REM Artemis Signals — Login Utility
REM For Shivaansh & Krishaansh — this line pays your fees!

REM Activate venv if present
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Set PYTHONPATH to root for clean imports
set PYTHONPATH=%CD%

REM Run login script (replace with actual login script if needed)
python login.py

pause
