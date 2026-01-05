# razorpay_sync.py
# For Shivaansh & Krishaansh — this sync pays your fees!
# Syncs user plans with Razorpay subscription status
def get_subscription_status(subscription_id):
    url = f"{RAZORPAY_BASE_URL}/subscriptions/{subscription_id}"
    resp = requests.get(url, auth=(RAZORPAY_API_KEY, RAZORPAY_API_SECRET))
    if resp.status_code == 200:
        return resp.json()
    logger.warning(f"Razorpay sync failed: {resp.status_code} {resp.text}")
    return None
def sync_user_plan(user):
    # user must have razorpay_subscription_id
    sub_id = getattr(user, 'razorpay_subscription_id', None)
    if not sub_id:
        logger.warning(f"User {getattr(user, 'username', '?')} missing Razorpay subscription id.")
        return False
    status = get_subscription_status(sub_id)
    if not status:
        return False
    # Update user plan/expiry in DB as per Razorpay
    # TODO: Implement DB update logic
    logger.info(f"User {user.username} plan synced: {status.get('status')}")
    return True

import requests
import os
from utils.logger import logger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from core.db_models import User

RAZORPAY_API_KEY = os.getenv("RAZORPAY_API_KEY", "demo")
RAZORPAY_API_SECRET = os.getenv("RAZORPAY_API_SECRET", "demo")
RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "../universe/metadata/universe.db")

def get_db_session():
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    Session = sessionmaker(bind=engine)
    return Session()

def sync_user_plan(user):
    sub_id = getattr(user, 'razorpay_subscription_id', None)
    if not sub_id:
        logger.warning(f"User {getattr(user, 'username', '?')} missing Razorpay subscription id.")
        return False
    status = get_subscription_status(sub_id)
    if not status:
        return False
    # Extract plan info from Razorpay response
    plan_id = status.get('plan_id')
    plan_expiry = status.get('current_end')  # Unix timestamp
    is_active = status.get('status') == 'active'
    # Update user in DB
    session = get_db_session()
    try:
        db_user = session.query(User).filter_by(username=user.username).first()
        if not db_user:
            logger.warning(f"User {user.username} not found in DB for Razorpay sync.")
            return False
        db_user.plan_id = plan_id or db_user.plan_id
        if plan_expiry:
            from datetime import datetime
            db_user.plan_expiry = datetime.utcfromtimestamp(plan_expiry)
        db_user.is_active = is_active
        session.commit()
        logger.info(f"User {user.username} plan synced: {status.get('status')}")
        return True
    except Exception as e:
        logger.error(f"DB update failed for user {user.username}: {e}")
        session.rollback()
        return False
    finally:
        session.close()
