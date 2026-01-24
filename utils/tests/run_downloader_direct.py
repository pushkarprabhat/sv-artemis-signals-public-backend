import importlib.util
import sys
from types import ModuleType
from pathlib import Path

# Inject lightweight shim for `kiteconnect` to avoid importing Twisted/win32 GUI
# during test runs. This lets us run downloader logic offline for verification.
stub_kite = ModuleType('kiteconnect')
class _KiteConnect:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.access_token = None
    def set_access_token(self, token):
        self.access_token = token
    def profile(self):
        return {"user": "stub"}
    def ltp(self, instruments):
        return {ins: {"last_price": 100.0} for ins in instruments}
    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
        """Return synthetic historical OHLCV data between from_date and to_date.

        Produces daily rows for `day` interval and a few 5-minute rows per day for `5minute`.
        Ensures `previousclose` is present so downloader calculates net_change/net_change_pct.
        """
        # Normalize dates
        try:
            import datetime as _dt
            if hasattr(from_date, 'date'):
                start = from_date.date()
            else:
                start = from_date
            if hasattr(to_date, 'date'):
                end = to_date.date()
            else:
                end = to_date
        import importlib.util
        import sys
        from types import ModuleType
        from pathlib import Path

        module_path = Path(__file__).parents[2] / 'core' / 'downloader.py'

        # Inject lightweight stub for core.corporate_actions to avoid importing heavy core package
        stub_ca = ModuleType('core.corporate_actions')
        def get_corporate_actions_manager():
            class Dummy:
                def record_file_checksum(self, *args, **kwargs):
                    return True
            return Dummy()
        stub_ca.get_corporate_actions_manager = get_corporate_actions_manager
        sys.modules['core.corporate_actions'] = stub_ca

        spec = importlib.util.spec_from_file_location('downloader_direct', str(module_path))
        downloader = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(downloader)

        symbols = ['RELIANCE', 'TCS', 'INFY']
        for s in symbols:
            print('Downloading:', s)
            try:
                ok = downloader.download_price_data(s, force_refresh=False)
                print(s, '->', 'OK' if ok else 'FAILED/NO_DATA')
            except Exception as e:
                print(s, '-> EXCEPTION:', e)
                    rows.append({
