@echo off
REM Artemis Signals — Run Main Utility
REM For Shivaansh & Krishaansh — this line pays your fees!

REM Activate venv if present
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Set PYTHONPATH to root for clean imports
set PYTHONPATH=%CD%

REM Run main.py in root directory
python main.py

pause
