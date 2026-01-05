# run_sqlite_migration.py
# For Shivaansh & Krishaansh — this script pays your fees!
# Run this to apply SQL migrations if sqlite3 CLI is not available
import sqlite3
import sys

DB_PATH = 'universe/metadata/universe.db'  # Fixed path for backend dir
SQL_PATH = 'migrations/001_create_users_table.sql'  # Fixed path for backend dir

def run_migration(db_path, sql_path):
    with open(sql_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.executescript(sql)
        print(f"Migration applied successfully to {db_path}")
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == '__main__':
    run_migration(DB_PATH, SQL_PATH)
