"""
core/database.py — Database connection and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# Create declarative base for ORM models
Base = declarative_base()

# Get database URL from .env or use SQLite for testing
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'sqlite:///./sv_pair_trading.db'  # Default: SQLite for local dev
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

def get_db():
    """Dependency injection for DB sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
