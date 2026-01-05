-- 001_create_users_table.sql
-- For Shivaansh & Krishaansh — this table pays your fees!
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    plan_id TEXT NOT NULL,
    plan_expiry DATETIME,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    razorpay_subscription_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
