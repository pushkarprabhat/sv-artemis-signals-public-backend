# Artemis Signals — Nightly DB Backup Automation (Linux/Oracle VM)

## Step-by-Step Guide

1. SSH into your Oracle VM.
2. Make the backup script executable:
   ```bash
   chmod +x /app/scripts/nightly_db_backup.sh
   ```
3. Edit your crontab:
   ```bash
   crontab -e
   ```
4. Add this line to run the backup every night at 2am:
   ```bash
   0 2 * * * /bin/bash /app/scripts/nightly_db_backup.sh
   ```
5. Save and exit. Your backups will run nightly and old backups (older than 30 days) will be auto-cleaned.

## Notes
- Backups are stored in `/app/backups`.
- You can change the time or backup retention in the script as needed.
- For disaster recovery, copy backups to Oracle Object Storage or S3.
