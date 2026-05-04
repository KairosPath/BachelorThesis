"""
Configuration file for DXY Prediction Project
==============================================
Diploma Thesis: Econometric Regression Methods for Dollar Index Prediction
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class DataConfig:
    """Configuration for data fetching and processing"""
    
    start_date: str = "2005-01-01"
    end_date: str = "2026-03-31"  # fixed thesis sample end (all downloads / panels)
    
    target_ticker: str = "DX-Y.NYB"
    
    feature_tickers: List[str] = field(default_factory=lambda: [
        "EEM",       # iShares MSCI Emerging Markets ETF (replaces ^IRX)
        "HG=F",      # Copper futures (replaces ^VIX)
        "^TNX",
        "GC=F",
        "CL=F",
        "EURUSD=X",
        "GBPUSD=X",
        "CAD=X",     # USD/CAD — DXY basket FX; stronger USD link, weaker EM co-move than small-cap equity
        "^GSPC",
    ])
    
    frequency: str = "daily"
    
    fill_method: str = "ffill"
    max_missing_ratio: float = 0.1

@dataclass
class ARIMAConfig:
    """ARIMA model configuration"""
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal: bool = False
    information_criterion: str = "aic"
    stepwise: bool = True

@dataclass
class GARCHConfig:
    """GARCH/EGARCH model configuration"""
    p: int = 1
    q: int = 1
    o: int = 0
    power: float = 2.0
    vol_model: str = "GARCH"
    dist: str = "normal"

@dataclass
class RegressionConfig:
    """Penalized regression configuration"""
    lasso_alpha_range: tuple = (0.001, 100)
    lasso_n_alphas: int = 100
    ridge_alpha_range: tuple = (0.001, 100_000.0)
    ridge_n_alphas: int = 100
    cv_folds: int = 5
    
    use_lasso_selection: bool = True
    min_features: int = 3

class Config:
    """Global configuration container"""
    data = DataConfig()
    arima = ARIMAConfig()
    garch = GARCHConfig()
    regression = RegressionConfig()
    
    RANDOM_SEED = 42
    
    LOG_LEVEL = "INFO"

config = Config()

