import pandas as pd
from pathlib import Path
from typing import Dict

class ValidationStats:
    """Manages validation statistics for data completeness checks"""
    
    def __init__(self):
        self.stats = {
            'last_check': None,
            'total_symbols': 0,
            'symbols_passed': 0,
            'symbols_failed': 0,
            'avg_completeness': 100.0
        }
    
    def get_summary(self) -> Dict:
        """Return summary of validation stats"""
        return self.stats

# Singleton instance
validation_stats = ValidationStats()

def get_min_records_threshold(timeframe: str) -> int:
    """Returns the minimum number of records expected for a timeframe to be considered complete"""
    thresholds = {
        'day': 200,          # At least 200 trading days (~1 year)
        '5minute': 2000,    # Approx 25 days of 5-min data
        '15minute': 750,
        '30minute': 375,
        '60minute': 200
    }
    return thresholds.get(timeframe, 100)

if __name__ == '__main__':
    # Test
    print(f"Summary: {validation_stats.get_summary()}")
    print(f"Day threshold: {get_min_records_threshold('day')}")
