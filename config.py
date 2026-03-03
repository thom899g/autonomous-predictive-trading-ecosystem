"""
Configuration and environment management for the Autonomous Predictive Trading Ecosystem.
Centralizes constants, paths, and environment variables with validation.
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Supported asset classes for trading."""
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    sequence_length: int = 60
    batch_size: int = 32
    lstm_units: int = 50
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    validation_split: float = 0.15

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_size_pct: float = 0.05
    stop_loss_pct: float = 0.02
    var_confidence_level: float = 0.95
    max_daily_drawdown: float = 0.03

class Config:
    """Main configuration singleton."""
    
    # Firebase
    FIREBASE_CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
    FIRESTORE_COLLECTION_PREFIX: str = "trading_ecosystem"
    
    # Data
    DEFAULT_TIMEFRAME: str = "1h"
    MAX_DATA_POINTS: int = 10000
    DATA_RETENTION_DAYS: int = 365
    
    # API Keys (validate existence)
    ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
    CCXT_EXCHANGE: str = "binance"
    
    # Paths
    MODEL_SAVE_DIR: str = "./saved_models"
    DATA_CACHE_DIR: str = "./data_cache"
    
    # Initialized configurations
    model_config: ModelConfig = ModelConfig()
    risk_config: RiskConfig = RiskConfig()
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        self._validate_paths()
        self._validate_api_keys()
        
    def _validate_paths(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)
        
        if not os.path.exists(self.FIREBASE_CREDENTIALS_PATH):
            logger.error(f"Firebase credentials not found at {self.FIREBASE_CREDENTIALS_PATH}")
            # In production, this would trigger a human intervention request
    
    def _validate_api_keys(self) -> None:
        """Log warnings for missing API keys."""
        if not self.ALPHA_VANTAGE_KEY:
            logger.warning("ALPHA_VANTAGE_KEY not set. Stock data collection limited.")

# Global config instance
config = Config()