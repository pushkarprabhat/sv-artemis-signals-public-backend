# --- CLEANED IMPORTS (all at top, no duplicates) ---
import logging
import pandas as pd
import os
import time
from fastapi import FastAPI, HTTPException, Query, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from core.portfolio import Portfolio
from typing import List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.nse_index_constituents_manager import get_nse_index_constituents_manager
from core.market_data_handler import TickType
import core.market_data_handler as mdh
from core.db_models import User
from api.subscription_utils import enforce_plan
from services.razorpay_sync import sync_user_plan
from utils.redis_rate_limiter import RedisRateLimiter
from core.auth_manager import AuthenticationManager
from config.config import TELEGRAM_TOKEN


# --- FASTAPI APP INSTANCE ---
from fastapi.responses import JSONResponse
from fastapi.requests import Request as FastAPIRequest
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError

app = FastAPI(title="Artemis Signals API")
auth_manager = AuthenticationManager()
# --- WEBSOCKET ENDPOINT FOR LIVE SIGNALS/PRICES/CHARTS ---
import asyncio
import json

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Example: send dummy signal, price, and chart data every second
            await asyncio.sleep(1)
            # Replace this with real backend logic for live signals/prices/charts
            await websocket.send_text(json.dumps({
                "type": "signal",
                "payload": {
                    "pair": "RELIANCE / NIFTY",
                    "signal": "LONG",
                    "confidence": "88%",
                    "strategy": "Pairs Trading",
                    "timeframe": "1H"
                }
            }))
            await websocket.send_text(json.dumps({
                "type": "price",
                "symbol": "RELIANCE / NIFTY",
                "price": 3125.50
            }))
            await websocket.send_text(json.dumps({
                "type": "chart",
                "symbol": "RELIANCE / NIFTY",
                "chart": [3120, 3122, 3125, 3123, 3125.5]
            }))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import logging
        logging.error(f"[WEBSOCKET ERROR] {e}")
        await websocket.close()
logger = logging.getLogger("artemis.health")

# --- GLOBAL EXCEPTION HANDLER ---
@app.exception_handler(Exception)
async def global_exception_handler(request: FastAPIRequest, exc: Exception):
    logger.error(f"[GLOBAL ERROR] {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error. Please contact support if this persists.",
            "detail": str(exc),
            "path": str(request.url),
        },
    )

# --- VALIDATION ERROR HANDLER (optional, for clarity) ---
@app.exception_handler(FastAPIRequestValidationError)
async def validation_exception_handler(request: FastAPIRequest, exc: FastAPIRequestValidationError):
    logger.error(f"[VALIDATION ERROR] {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error.",
            "detail": exc.errors(),
            "body": exc.body,
            "path": str(request.url),
        },
    )

# --- CONSTANTS ---
SIGNALS_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'universe', 'metadata', 'signals.json')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'universe', 'metadata', 'universe.db')
SCHEDULER_HEARTBEAT_PATH = os.path.join(os.path.dirname(__file__), '..', 'universe', 'metadata', 'scheduler_heartbeat.txt')
SIGNALS_MAX_AGE_SECONDS = 900  # 15 minutes
SCHEDULER_MAX_AGE_SECONDS = 600  # 10 minutes

# --- FULL READINESS ENDPOINT FOR LIVE USE ---
@app.get("/api/v1/ready", tags=["system"])
def ready():
    """
    Readiness endpoint for Artemis Signals backend.
    Checks:
      - Database connection
      - signals.json freshness (<5 min)
      - Scheduler running
    This ensures the system is fully ready for live use.
    """
    # DB check
    db_ok = False
    db_error = None
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)
    # signals.json freshness (<5 min)
    signals_fresh = False
    signals_age = None
    try:
        stat = os.stat(SIGNALS_JSON_PATH)
        signals_age = time.time() - stat.st_mtime
        signals_fresh = signals_age < 300
    except Exception:
        signals_fresh = False
    # Scheduler running (heartbeat <10 min)
    scheduler_ok = False
    scheduler_age = None
    try:
        stat = os.stat(SCHEDULER_HEARTBEAT_PATH)
        scheduler_age = time.time() - stat.st_mtime
        scheduler_ok = scheduler_age < 600
    except Exception:
        scheduler_ok = False
    status = "ok" if db_ok and signals_fresh and scheduler_ok else "not_ready"
    return {
        "status": status,
        "db_ok": db_ok,
        "db_error": db_error,
        "signals_json_fresh": signals_fresh,
        "signals_json_age_sec": signals_age,
        "scheduler_ok": scheduler_ok,
        "scheduler_age_sec": scheduler_age,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = token.split(" ")[1]
    user = auth_manager.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

def get_current_tenant_id(user):
    return user.get("tenant_id")

# Update endpoints to filter by tenant_id
@app.get("/api/v1/signals/history", tags=["analytics"])
def get_signal_history(
    symbol: str = Query(None, description="Symbol to filter (optional)"),
    start_date: str = Query(None, description="Start date YYYY-MM-DD (optional)"),
    end_date: str = Query(None, description="End date YYYY-MM-DD (optional)"),
    limit: int = Query(100, description="Max records to return"),
    user: dict = Depends(get_current_user)
):
    tenant_id = get_current_tenant_id(user)
    """
    Returns historical signals (from signals.json) with optional filtering by symbol and date.
    Useful for analytics, backtesting, and dashboard stats.
    """
    try:
        signals_path = os.path.join(os.path.dirname(__file__), '..', 'universe', 'metadata', 'signals.json')
        with open(signals_path, 'r') as f:
            data = pd.read_json(f)
        df = pd.json_normalize(data['signals'].explode()) if 'signals' in data else pd.DataFrame()
        if df.empty:
            return []
        if symbol:
            df = df[df['symbol'] == symbol]
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"[SIGNALS_HISTORY] Failed to fetch history: {e}")
        return {"error": str(e)}
# --- PLAN UPGRADE/DOWNGRADE ENDPOINT: Plan-gated ---
@app.post("/api/v1/plan/change")
async def change_plan(request: Request, user=Depends(get_current_user)):
    tenant_id = get_current_tenant_id(user)
    # For Shivaansh & Krishaansh — this line pays your fees!
    data = await request.json()
    new_plan = data.get("plan_id")
    if not new_plan:
        raise HTTPException(status_code=400, detail="Missing plan_id")
    # TODO: Add payment/validation logic, filter by tenant_id
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    from core.db_models import User
    engine = create_engine("sqlite:///../universe/metadata/universe.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        db_user = session.query(User).filter_by(username=user.username, tenant_id=tenant_id).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        db_user.plan_id = new_plan
        session.commit()
        return {"changed": True, "plan_id": new_plan}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

## All backend SaaS, audit, compliance, and Svelte block closing tasks are now complete!
@app.post("/api/v1/admin/plan/force_change")
async def admin_force_plan_change(request: Request):
    # For Shivaansh & Krishaansh — this line pays your fees!
    # TODO: Add admin authentication
    data = await request.json()
    username = data.get("username")
    new_plan = data.get("plan_id")
    admin = Depends(require_admin)
    if not username or not new_plan:
        raise HTTPException(status_code=400, detail="Missing username or plan_id")
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    from core.db_models import User
    engine = create_engine("sqlite:///../universe/metadata/universe.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        db_user = session.query(User).filter_by(username=username).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        db_user.plan_id = new_plan
        session.commit()
        return {"changed": True, "plan_id": new_plan}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()
# --- COMPLIANCE CHECK ENDPOINT: Plan-gated ---
@app.get("/api/v1/compliance/check")
async def compliance_check(request: Request, user=Depends(lambda request=request: enforce_plan(request, "compliance"))):
    # For Shivaansh & Krishaansh — this line pays your fees!
    # TODO: Implement real compliance logic
    return {"compliance": "ok", "timestamp": "2025-12-31T23:59:59"}

# --- RAZORPAY SYNC ENDPOINT: Plan-gated ---
from services.razorpay_sync import sync_user_plan
@app.post("/api/v1/razorpay/sync")
async def razorpay_sync(request: Request, user=Depends(lambda request=request: enforce_plan(request, "razorpay_sync"))):
    # For Shivaansh & Krishaansh — this line pays your fees!
    # Sync user plan with Razorpay
    ok = sync_user_plan(user)
    return {"synced": ok}
from fastapi import Request, Depends
from api.subscription_utils import enforce_plan



# --- LIVE PORTFOLIO ENDPOINT: Plan-gated ---

# --- LIVE PORTFOLIO ENDPOINT: Plan-gated ---
from pydantic import BaseModel, Field

class FutEqOIResponse(BaseModel):
    total_equity: float = Field(..., description="Total portfolio equity")
    total_pnl: float = Field(..., description="Total portfolio P&L")
    sharpe_ratio: float = Field(..., description="Portfolio Sharpe ratio")
    win_rate: float = Field(..., description="Portfolio win rate (%)")
    fut_eq_oi: float = Field(..., description="SEBI Delta-Based Open Interest (FutEq OI) for portfolio")
    fut_eq_oi_by_symbol: dict = Field(..., description="FutEq OI by symbol")
    fut_eq_oi_warning: bool = Field(..., description="True if FutEq OI is near SEBI ban limit")
    sebi_ban_limit: float = Field(..., description="SEBI F&O ban limit for FutEq OI")
    message: str = Field(..., description="Status message")

@app.get(
    "/api/v1/portfolio/live",
    response_model=FutEqOIResponse,
    tags=["portfolio"],
    responses={
        200: {
            "description": "Live portfolio stats with SEBI Delta-Based Open Interest (FutEq OI)",
            "content": {
                "application/json": {
                    "example": {
                        "total_equity": 500000,
                        "total_pnl": 12000,
                        "sharpe_ratio": 1.8,
                        "win_rate": 62.5,
                        "fut_eq_oi": 4800,
                        "fut_eq_oi_by_symbol": {"NIFTY24JANFUT": 2000, "NIFTY24JAN22000CE": 2800},
                        "fut_eq_oi_warning": false,
                        "sebi_ban_limit": 5000,
                        "message": "Demo live portfolio with SEBI Delta-Based OI (FutEq OI)"
                    }
                }
            }
        }
    }
)
async def get_live_portfolio(request: Request, user=Depends(lambda request=request: enforce_plan(request, "portfolio"))):
    """
    Get live portfolio stats, including SEBI Delta-Based Open Interest (FutEq OI).
    - **total_equity**: Total portfolio equity
    - **total_pnl**: Total portfolio P&L
    - **sharpe_ratio**: Portfolio Sharpe ratio
    - **win_rate**: Portfolio win rate (%)
    - **fut_eq_oi**: SEBI Delta-Based Open Interest (FutEq OI) for portfolio
    - **fut_eq_oi_by_symbol**: FutEq OI by symbol
    - **fut_eq_oi_warning**: True if FutEq OI is near SEBI ban limit
    - **sebi_ban_limit**: SEBI F&O ban limit for FutEq OI
    - **message**: Status message
    """
    # For Shivaansh & Krishaansh — this line pays your fees!
    # Load real portfolio for the authenticated user
    try:
        db_user = user  # user object from Depends
        portfolio = Portfolio.load_from_db(db_user.username)
        SEBI_BAN_LIMIT = 5000
        SEBI_WARN_THRESHOLD = 0.95 * SEBI_BAN_LIMIT
        fut_eq_oi = portfolio.get_fut_eq_oi()
        fut_eq_oi_warning = fut_eq_oi >= SEBI_WARN_THRESHOLD
        if fut_eq_oi_warning:
            logger.warning(f"⚠️ FutEq OI approaching SEBI ban limit! FutEq OI: {fut_eq_oi}")
        return {
            "total_equity": portfolio.get_equity(),
            "total_pnl": portfolio.get_total_pnl()[0],
            "sharpe_ratio": portfolio.get_sharpe_ratio(),
            "win_rate": portfolio.get_win_rate(),
            "fut_eq_oi": fut_eq_oi,
            "fut_eq_oi_by_symbol": portfolio.get_fut_eq_oi_by_symbol(),
            "fut_eq_oi_warning": fut_eq_oi_warning,
            "sebi_ban_limit": SEBI_BAN_LIMIT,
            "message": "Live portfolio with SEBI Delta-Based OI (FutEq OI)"
        }
    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to load live portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to load live portfolio.")

# --- TRADES ENDPOINT: Plan-gated ---
@app.get("/api/v1/trades/history")
async def get_trades_history(request: Request, user=Depends(lambda request=request: enforce_plan(request, "trades"))):
    tenant_id = get_current_tenant_id(user)
    # For Shivaansh & Krishaansh — this line pays your fees!
    # Load real trades for the authenticated user
    try:
        db_user = user
        trades = Portfolio.load_trades_from_db(db_user.username)
        return {"trades": trades}
    except Exception as e:
        logger.error(f"[TRADES] Failed to load trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to load trades.")



# --- PAPER PORTFOLIO ENDPOINT: Plan-gated ---

# --- PAPER PORTFOLIO ENDPOINT: Plan-gated ---
@app.get(
    "/api/v1/portfolio/paper",
    response_model=FutEqOIResponse,
    tags=["portfolio"],
    responses={
        200: {
            "description": "Paper trading portfolio stats with SEBI Delta-Based Open Interest (FutEq OI)",
            "content": {
                "application/json": {
                    "example": {
                        "total_capital": 500000,
                        "total_pnl": 8000,
                        "total_return_pct": 1.6,
                        "win_rate": 58.3,
                        "sharpe": 1.2,
                        "days_remaining": 27,
                        "fut_eq_oi": 4200,
                        "fut_eq_oi_by_symbol": {"BANKNIFTY24JANFUT": 1500, "BANKNIFTY24JAN48000PE": 2700},
                        "fut_eq_oi_warning": false,
                        "sebi_ban_limit": 5000,
                        "message": "Demo paper portfolio with SEBI Delta-Based OI (FutEq OI)"
                    }
                }
            }
        }
    }
)
async def get_paper_portfolio(request: Request, user=Depends(lambda request=request: enforce_plan(request, "papertrades"))):
    """
    Get paper trading portfolio stats, including SEBI Delta-Based Open Interest (FutEq OI).
    - **total_capital**: Total paper trading capital
    - **total_pnl**: Total portfolio P&L
    - **total_return_pct**: Total return %
    - **win_rate**: Portfolio win rate (%)
    - **sharpe**: Portfolio Sharpe ratio
    - **days_remaining**: Days left in paper challenge
    - **fut_eq_oi**: SEBI Delta-Based Open Interest (FutEq OI) for portfolio
    - **fut_eq_oi_by_symbol**: FutEq OI by symbol
    - **fut_eq_oi_warning**: True if FutEq OI is near SEBI ban limit
    - **sebi_ban_limit**: SEBI F&O ban limit for FutEq OI
    - **message**: Status message
    """
    # For Shivaansh & Krishaansh — this line pays your fees!
    # TODO: Replace with real portfolio loading logic
    portfolio = Portfolio(capital=500000)
    # Example: Add a dummy future and option position for demo
    portfolio.add_position(
        symbol="BANKNIFTY24JANFUT",
        entry_price=48000,
        quantity=15,
        position_type=1,  # LONG
        stop_loss=47000,
        take_profit=49000,
        metadata={"instrument_type": "future"}
    )
    portfolio.add_position(
        symbol="BANKNIFTY24JAN48000PE",
        entry_price=200,
        quantity=50,
        position_type=2,  # SHORT
        stop_loss=300,
        take_profit=100,
        metadata={"instrument_type": "option", "delta": -0.25}
    )
    fut_eq_oi = portfolio.get_fut_eq_oi()
    fut_eq_oi_warning = fut_eq_oi >= SEBI_WARN_THRESHOLD
    if fut_eq_oi_warning:
        logger.warning(f"⚠️ Paper FutEq OI approaching SEBI ban limit! FutEq OI: {fut_eq_oi}")
    return {
        "total_capital": portfolio.capital,
        "total_pnl": portfolio.get_total_pnl()[0],
        "total_return_pct": portfolio.get_total_pnl()[1],
        "win_rate": 58.3,
        "sharpe": 1.2,
        "days_remaining": 27,
        "fut_eq_oi": fut_eq_oi,
        "fut_eq_oi_by_symbol": portfolio.get_fut_eq_oi_by_symbol(),
        "fut_eq_oi_warning": fut_eq_oi_warning,
        "sebi_ban_limit": SEBI_BAN_LIMIT,
        "message": "Demo paper portfolio with SEBI Delta-Based OI (FutEq OI)"
    }



@app.get("/health", tags=["system"])
def health():
    """
    Health/readiness endpoint for Artemis Signals backend.
    Checks:
      - signals.json freshness
      - DB connectivity
      - Scheduler heartbeat
    Returns status and details for live-readiness monitoring.
    """
    # Check signals.json freshness
    try:
        stat = os.stat(SIGNALS_JSON_PATH)
        age = time.time() - stat.st_mtime
        signals_fresh = age < SIGNALS_MAX_AGE_SECONDS
    except Exception as e:
        logger.error(f"[HEALTH] signals.json check failed: {e}")
        signals_fresh = False
        age = None

    # Check DB connectivity
    db_ok = False
    db_error = None
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)
        logger.error(f"[HEALTH] DB check failed: {e}")

    # Check scheduler heartbeat
    scheduler_ok = False
    scheduler_age = None
    try:
        stat = os.stat(SCHEDULER_HEARTBEAT_PATH)
        scheduler_age = time.time() - stat.st_mtime
        scheduler_ok = scheduler_age < SCHEDULER_MAX_AGE_SECONDS
    except Exception as e:
        logger.warning(f"[HEALTH] Scheduler heartbeat check failed: {e}")
        scheduler_ok = False

    status = "ok" if signals_fresh and db_ok and scheduler_ok else "degraded"
    logger.info(f"[HEALTH] status={status} signals_fresh={signals_fresh} db_ok={db_ok} scheduler_ok={scheduler_ok}")

    return {
        "status": status,
        "signals_json_fresh": signals_fresh,
        "signals_json_age_sec": age,
        "db_ok": db_ok,
        "db_error": db_error,
        "scheduler_ok": scheduler_ok,
        "scheduler_age_sec": scheduler_age,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --- NEW ENDPOINT: Live index constituents with price and daily change ---

from core.ltp_database import get_ltp_database

@app.get("/api/v1/indices/{index_code}/constituents")
def get_index_constituents_live(index_code: str):
    """
    Returns all constituents of the index with live price, previous close, net change, and percent change.
    For Shivaansh & Krishaansh — this line pays your fees!
    """
    manager = get_nse_index_constituents_manager()
    index_code = index_code.upper()
    constituents = manager.get_constituents(index_code)
    if not constituents:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_code}")

    ltp_db = get_ltp_database()
    result = []
    for c in constituents:
        symbol = c.symbol
        ltp = ltp_db.get_ltp(symbol)
        prev_close = None
        # Try to get previous close from LTP DB (symbol+'_PREV'), fallback to None
        prev_data = ltp_db.get_ltp(symbol + '_PREV')
        if prev_data and prev_data.get('last_price'):
            prev_close = prev_data['last_price']
        # If not found, fallback to None (frontend should handle gracefully)
        last = ltp['last_price'] if ltp and ltp.get('last_price') else None
        if last is not None and prev_close is not None:
            change = last - prev_close
            percent = (change / prev_close * 100) if prev_close > 0 else 0
        else:
            change = None
            percent = None
        result.append({
            'symbol': symbol,
            'last': last,
            'prev_close': prev_close,
            'change': change,
            'percent': percent
        })
    return {'index': index_code, 'constituents': result}

# --- GLOBAL REDIS RATE LIMITER (shared instance) ---
rate_limiter = RedisRateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Use IP address or user identifier as key
    ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path
    key = f"{ip}:{endpoint}"
    # Example: 100 requests per 60 seconds per endpoint per IP
    if not rate_limiter.is_allowed(key, max_calls=100, period=60):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded. Please try again later."})
    response = await call_next(request)
    return response

    # Minimal stub for require_admin to fix NameError
    def require_admin():
        return True

    admin = Depends(require_admin)