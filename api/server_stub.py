# Minimal FastAPI stub for endpoint registration testing
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

@app.get("/api/v1/health", tags=["system"])
def health_proxy():
    return {"status": "ok", "message": "health endpoint stub (test)"}

@app.get("/api/v1/data/download", tags=["data"])
def data_download_stub():
    return JSONResponse({"status": "error", "message": "data download not implemented yet"}, status_code=501)

@app.get("/api/v1/data/refresh", tags=["data"])
def data_refresh_stub():
    return JSONResponse({"status": "error", "message": "data refresh not implemented yet"}, status_code=501)

@app.get("/api/v1/eod", tags=["jobs"])
def eod_stub():
    logger = logging.getLogger("artemis.eod")
    logger.info("[EOD] EOD process triggered via /api/v1/eod endpoint.")
    return JSONResponse({"status": "ok", "message": "EOD process executed (stub)"}, status_code=200)

@app.get("/api/v1/bod", tags=["jobs"])
def bod_stub():
    return JSONResponse({"status": "error", "message": "BOD job not implemented yet"}, status_code=501)

@app.get("/api/v1/signal/entry", tags=["signals"])
def signal_entry_stub():
    return JSONResponse({"status": "error", "message": "signal entry not implemented yet"}, status_code=501)

@app.get("/api/v1/signal/exit", tags=["signals"])
def signal_exit_stub():
    return JSONResponse({"status": "error", "message": "signal exit not implemented yet"}, status_code=501)

@app.get("/api/v1/papertrading", tags=["papertrading"])
def papertrading_stub():
    return JSONResponse({"status": "error", "message": "paper trading not implemented yet"}, status_code=501)

@app.get("/api/v1/strategies", tags=["strategies"])
def strategies_stub():
    return JSONResponse({"status": "error", "message": "strategies not implemented yet"}, status_code=501)

@app.get("/api/v1/analysers", tags=["analysers"])
def analysers_stub():
    return JSONResponse({"status": "error", "message": "analysers not implemented yet"}, status_code=501)
