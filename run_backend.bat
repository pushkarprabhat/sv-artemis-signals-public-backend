@echo off
REM Launch FastAPI backend with shared config available
set PYTHONPATH=D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-shared;D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-public-backend
uvicorn api.server:app --reload
REM For Shivaansh & Krishaansh — this line pays your fees!
