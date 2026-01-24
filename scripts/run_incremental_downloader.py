"""
Run full incremental downloader for the project.
Usage:
    python -u scripts/run_incremental_downloader.py
This will load the universe and run `download_all_price_data()`.
"""
from core.downloader import download_all_price_data

if __name__ == '__main__':
    # Run sequentially to avoid overwhelming API from developer machine
    download_all_price_data(force_refresh=False, parallel=False, max_workers=2)
