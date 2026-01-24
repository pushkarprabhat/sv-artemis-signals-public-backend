@echo off
chcp 65001 >nul
REM Launch FastAPI backend with shared config available
call .venv\Scripts\activate.bat
set PYTHONPATH=D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-shared;D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-public-backend
uvicorn api.server:app --reload
REM For Shivaansh & Krishaansh — this line pays your fees!
