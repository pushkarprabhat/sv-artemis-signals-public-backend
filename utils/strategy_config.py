# utils/strategy_config.py
"""
Strategy Configuration Management
Centralized configuration for all trading strategies with presets and defaults
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configuration storage path
CONFIG_DIR = Path("marketdata/strategy_configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STRATEGY DEFAULTS
# ============================================================================

STRATEGY_DEFAULTS = {
    'pairs_trading': {
        'z_entry': 2.0,
        'z_exit': 0.5,
        'lookback': 60,
    },
    'momentum': {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volume_sma_period': 20,
        'hold_days': 5,
    },
    'mean_reversion': {
        'ma_period': 20,
        'std_dev': 2.0,
    },
    'ma_crossover': {
        'fast_period': 20,
        'slow_period': 50,
    },
}

# ============================================================================
# PARAMETER RANGES (Min, Max, Step, Type)
# ============================================================================

STRATEGY_PARAMETER_RANGES = {
    'pairs_trading': {
        'z_entry': {'min': 0.5, 'max': 5.0, 'step': 0.1, 'type': 'float'},
        'z_exit': {'min': 0.1, 'max': 2.0, 'step': 0.1, 'type': 'float'},
        'lookback': {'min': 10, 'max': 200, 'step': 5, 'type': 'int'},
    },
    'momentum': {
        'rsi_period': {'min': 5, 'max': 30, 'step': 1, 'type': 'int'},
        'rsi_oversold': {'min': 10, 'max': 40, 'step': 1, 'type': 'int'},
        'rsi_overbought': {'min': 60, 'max': 90, 'step': 1, 'type': 'int'},
        'volume_sma_period': {'min': 10, 'max': 50, 'step': 1, 'type': 'int'},
        'hold_days': {'min': 1, 'max': 20, 'step': 1, 'type': 'int'},
    },
    'mean_reversion': {
        'ma_period': {'min': 10, 'max': 50, 'step': 1, 'type': 'int'},
        'std_dev': {'min': 1.0, 'max': 3.0, 'step': 0.5, 'type': 'float'},
    },
    'ma_crossover': {
        'fast_period': {'min': 5, 'max': 30, 'step': 1, 'type': 'int'},
        'slow_period': {'min': 30, 'max': 100, 'step': 1, 'type': 'int'},
    },
}

# ============================================================================
# STRATEGY PRESETS (Conservative, Moderate, Aggressive)
# ============================================================================

STRATEGY_PRESETS = {
    'pairs_trading': {
        'conservative': {
            'name': 'Conservative',
            'description': 'Lower entry signals, tighter stops - less frequent trades, lower risk',
            'params': {
                'z_entry': 2.5,
                'z_exit': 1.0,
                'lookback': 80,
            }
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Balanced risk/reward - standard settings',
            'params': {
                'z_entry': 2.0,
                'z_exit': 0.5,
                'lookback': 60,
            }
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Lower entry signals, wider stops - more frequent trades, higher risk',
            'params': {
                'z_entry': 1.5,
                'z_exit': 0.2,
                'lookback': 40,
            }
        },
    },
    'momentum': {
        'conservative': {
            'name': 'Conservative',
            'description': 'Stricter RSI levels, longer hold - fewer but higher quality trades',
            'params': {
                'rsi_period': 21,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'volume_sma_period': 30,
                'hold_days': 10,
            }
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Balanced settings - standard momentum strategy',
            'params': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_sma_period': 20,
                'hold_days': 5,
            }
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Looser RSI levels, shorter hold - more frequent trades, higher volume',
            'params': {
                'rsi_period': 7,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'volume_sma_period': 10,
                'hold_days': 2,
            }
        },
    },
    'mean_reversion': {
        'conservative': {
            'name': 'Conservative',
            'description': 'Wider bands, slower MA - fewer extreme trades',
            'params': {
                'ma_period': 30,
                'std_dev': 2.5,
            }
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Balanced Bollinger Bands settings',
            'params': {
                'ma_period': 20,
                'std_dev': 2.0,
            }
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Tighter bands, faster MA - more trading signals',
            'params': {
                'ma_period': 15,
                'std_dev': 1.5,
            }
        },
    },
    'ma_crossover': {
        'conservative': {
            'name': 'Conservative',
            'description': 'Slower MAs, wider crossovers - fewer but stronger signals',
            'params': {
                'fast_period': 30,
                'slow_period': 100,
            }
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Standard MA crossover settings',
            'params': {
                'fast_period': 20,
                'slow_period': 50,
            }
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Faster MAs, tighter crossovers - more frequent signals',
            'params': {
                'fast_period': 10,
                'slow_period': 30,
            }
        },
    },
}

# ============================================================================
# SCREENER PRESETS
# ============================================================================

SCREENER_PRESETS = {
    'stock_screener': {
        'conservative': {
            'name': 'Conservative',
            'description': 'Large cap, high liquidity, stable growth',
            'criteria': {
                'market_cap_min': 10000,  # Crores
                'volume_min': 1000000,
                'pe_max': 25,
                'dividend_yield_min': 1.0,
            }
        },
        'moderate': {
            'name': 'Moderate',
            'description': 'Mid cap, balanced growth',
            'criteria': {
                'market_cap_min': 1000,
                'volume_min': 500000,
                'pe_max': 35,
                'dividend_yield_min': 0.5,
            }
        },
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Small cap, high growth potential, higher risk',
            'criteria': {
                'market_cap_min': 100,
                'volume_min': 100000,
                'pe_max': 50,
                'dividend_yield_min': 0,
            }
        },
    },
}

# ============================================================================
# CONFIGURATION MANAGER CLASS
# ============================================================================

class StrategyConfigManager:
    """Manages strategy configurations, presets, and persistence"""
    
    def __init__(self):
        """Initialize configuration manager"""
        self.configs_dir = CONFIG_DIR
        self.configs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_defaults(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        if strategy_name not in STRATEGY_DEFAULTS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return STRATEGY_DEFAULTS[strategy_name].copy()
    
    def get_parameter_ranges(self, strategy_name: str) -> Dict[str, Dict]:
        """Get parameter ranges and types for a strategy"""
        if strategy_name not in STRATEGY_PARAMETER_RANGES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return STRATEGY_PARAMETER_RANGES[strategy_name].copy()
    
    def get_presets(self, strategy_name: str) -> Dict[str, Dict]:
        """Get all presets for a strategy"""
        if strategy_name not in STRATEGY_PRESETS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return STRATEGY_PRESETS[strategy_name].copy()
    
    def get_preset_params(self, strategy_name: str, preset_name: str) -> Dict[str, Any]:
        """Get parameters for a specific preset"""
        if strategy_name not in STRATEGY_PRESETS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Normalize to lowercase to handle 'Conservative' vs 'conservative'
        preset_key = preset_name.lower()
        if preset_key not in STRATEGY_PRESETS[strategy_name]:
            raise ValueError(f"Unknown preset: {preset_name}")
        return STRATEGY_PRESETS[strategy_name][preset_key]['params'].copy()
    
    def save_custom_config(self, strategy_name: str, config_name: str, params: Dict[str, Any]) -> Path:
        """Save a custom configuration"""
        config_path = self.configs_dir / f"{strategy_name}_{config_name}.json"
        
        config_data = {
            'strategy': strategy_name,
            'name': config_name,
            'params': params,
            'created': str(Path.ctime(config_path)) if config_path.exists() else None,
            'modified': str(Path.ctime(config_path)) if config_path.exists() else None,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved custom config: {strategy_name}/{config_name}")
        return config_path
    
    def load_custom_config(self, strategy_name: str, config_name: str) -> Dict[str, Any]:
        """Load a custom configuration"""
        config_path = self.configs_dir / f"{strategy_name}_{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data['params']
    
    def list_custom_configs(self, strategy_name: str) -> List[str]:
        """List all custom configurations for a strategy"""
        configs = []
        for config_file in self.configs_dir.glob(f"{strategy_name}_*.json"):
            config_name = config_file.stem.replace(f"{strategy_name}_", "")
            configs.append(config_name)
        return sorted(configs)
    
    def delete_custom_config(self, strategy_name: str, config_name: str) -> bool:
        """Delete a custom configuration"""
        config_path = self.configs_dir / f"{strategy_name}_{config_name}.json"
        
        if not config_path.exists():
            return False
        
        config_path.unlink()
        logger.info(f"Deleted custom config: {strategy_name}/{config_name}")
        return True
    
    def validate_params(self, strategy_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters against ranges"""
        if strategy_name not in STRATEGY_PARAMETER_RANGES:
            return False, f"Unknown strategy: {strategy_name}"
        
        ranges = STRATEGY_PARAMETER_RANGES[strategy_name]
        
        for param_name, param_value in params.items():
            if param_name not in ranges:
                return False, f"Unknown parameter: {param_name}"
            
            param_range = ranges[param_name]
            
            # Type check
            param_type = param_range['type']
            if param_type == 'int' and not isinstance(param_value, int):
                return False, f"{param_name} must be integer"
            if param_type == 'float' and not isinstance(param_value, (int, float)):
                return False, f"{param_name} must be float"
            
            # Range check
            if param_value < param_range['min'] or param_value > param_range['max']:
                return False, f"{param_name} out of range [{param_range['min']}, {param_range['max']}]"
        
        return True, "Valid"
    
    def export_config(self, strategy_name: str, params: Dict[str, Any], filepath: Path) -> bool:
        """Export configuration to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'strategy': strategy_name,
                    'params': params,
                }, f, indent=2)
            logger.info(f"Exported config to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False
    
    def import_config(self, filepath: Path) -> tuple[str, Dict[str, Any]]:
        """Import configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Imported config from {filepath}")
            return data['strategy'], data['params']
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            raise


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_config_manager = None

def get_strategy_config_manager() -> StrategyConfigManager:
    """Get singleton instance of configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = StrategyConfigManager()
    return _config_manager
