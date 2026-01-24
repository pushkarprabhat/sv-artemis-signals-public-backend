@echo off
chcp 65001 >nul
REM Artemis Signals — Run Main Utility
REM For Shivaansh & Krishaansh — this line pays your fees!

pause

REM Always activate .venv for reliability
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
python main.py
pause
