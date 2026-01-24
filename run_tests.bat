@echo off
chcp 65001 >nul
REM Run backend tests with shared config available
call .venv\Scripts\activate.bat
set PYTHONPATH=D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-shared;D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-public-backend
pytest -s tests
REM For Shivaansh & Krishaansh — this line pays your fees!
