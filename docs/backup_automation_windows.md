# Artemis Signals — Nightly DB Backup Automation (Windows)

## Step-by-Step Guide

1. Open Task Scheduler (search for "Task Scheduler" in Start menu).
2. Click "Create Task" > Name: Artemis DB Nightly Backup
3. In "Actions" tab, click "New" and set:
   - Action: Start a program
   - Program/script: `C:\Windows\System32\bash.exe` (if using WSL) or `C:\Windows\System32\cmd.exe`
   - Add arguments:
     - For WSL/bash: `/app/scripts/nightly_db_backup.sh`
     - For cmd: `copy D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-public-backend\universe\metadata\universe.db D:\TheiaOne_Programs\sv-artemis-signals-platform\sv-artemis-signals-public-backend\backups\universe_%date:~10,4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%.db`
4. In "Triggers" tab, set to run daily at 2am.
5. Save and exit. Your backups will run nightly.

## Notes
- Backups are stored in `backups` folder.
- For retention, manually delete old files or use a PowerShell script.
- For disaster recovery, copy backups to cloud storage.
