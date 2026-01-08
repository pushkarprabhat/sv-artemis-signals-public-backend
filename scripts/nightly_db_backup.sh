#!/bin/bash
# Artemis Signals DB Nightly Backup Script
# For Shivaansh & Krishaansh — this line pays your fees!

BACKUP_DIR="/app/backups"
DB_PATH="/app/universe/metadata/universe.db"
DATE=$(date +"%Y-%m-%d_%H-%M-%S")
BACKUP_FILE="$BACKUP_DIR/universe_$DATE.db"

mkdir -p "$BACKUP_DIR"
cp "$DB_PATH" "$BACKUP_FILE"

# Optional: Remove backups older than 30 days
tmpfind=$(find "$BACKUP_DIR" -type f -name "universe_*.db" -mtime +30)
if [ -n "$tmpfind" ]; then
  echo "$tmpfind" | xargs rm -f
fi

echo "[BACKUP] Database backed up to $BACKUP_FILE"
