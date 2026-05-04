"""
GARCH Model Module
==================
GARCH, EGARCH, and GJR-GARCH models for volatility modeling
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Union
import pickle
from pathlib import Path
import logging
import warnings

from arch import arch_model

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, MODELS_DIR

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class GARCHModel:
    """
    GARCH Family Models for volatility forecasting
    
    Supported models:
    - GARCH(p, q)
    - EGARCH(p, o, q) - Exponential GARCH with asymmetry
    - GJR-GARCH(p, o, q) - GJR-GARCH with leverage effect
    
    Features:
    - Multiple distribution options (normal, t, skew-t, GED)
    - Combined mean + volatility modeling
    - Volatility forecasting
    - VaR calculation
    """
    
    def __init__(
        self,
        vol_model: str = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        mean_model: str = "Constant",
        dist: str = "normal"
    ):
        """
        Initialize GARCH Model
        
        Args:
            vol_model: Volatility model type ('GARCH', 'EGARCH', 'GJR-GARCH')
            p: GARCH order (lagged squared returns)
            o: EGARCH/GJR order (asymmetric term)
            q: ARCH order (lagged variance)
            mean_model: Mean model type ('Zero', 'Constant', 'AR')
            dist: Error distribution ('normal', 't', 'skewt', 'ged')
        """
        self.vol_model_type = vol_model
        self.p = p
        self.o = o
        self.q = q
        self.mean_model = mean_model
        self.dist = dist
        
        self.model = None
        self.fitted_model = None
        self.config = config.garch
    
    def _create_model(self, returns: pd.Series) -> arch_model:
        """
        Create arch model based on configuration
        
        Args:
            returns: Returns series
            
        Returns:
            arch_model instance
        """
        returns_scaled = returns * 100
        
        model = arch_model(
            returns_scaled,
            mean=self.mean_model,
            vol=self.vol_model_type,
            p=self.p,
            o=self.o if self.vol_model_type in ['EGARCH', 'GJR-GARCH'] else 0,
            q=self.q,
            dist=self.dist
        )
        
        return model
    
    def fit(
        self,
        returns: pd.Series,
        update_freq: int = 0,
        show_warning: bool = False
    ) -> 'GARCHModel':
        """
        Fit GARCH model to returns data
        
        Args:
            returns: Returns series
            update_freq: Frequency for progress updates
            show_warning: Whether to show optimization warnings
            
        Returns:
            Self
        """
        returns_clean = returns.dropna()
        
        logger.info(f"Fitting {self.vol_model_type}({self.p},{self.o},{self.q}) model...")
        
        self.model = self._create_model(returns_clean)
        
        try:
            self.fitted_model = self.model.fit(
                update_freq=update_freq,
                disp='off',
                show_warning=show_warning
            )
            
            logger.info("Model fitted successfully")
            logger.info(f"Log-Likelihood: {self.fitted_model.loglikelihood:.2f}")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise
        
        return self
    
    def predict(
        self,
        horizon: int = 1,
        reindex: bool = True
    ) -> pd.DataFrame:
        """
        Forecast volatility
        
        Args:
            horizon: Forecast horizon
            reindex: Whether to reindex forecasts
            
        Returns:
            DataFrame with variance and volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = self.fitted_model.forecast(horizon=horizon, reindex=reindex)
        
        variance = forecasts.variance
        volatility = np.sqrt(variance)
        
        volatility = volatility / 100
        variance = variance / 10000
        
        results = pd.DataFrame({
            'variance': variance.values.flatten()[:horizon] if hasattr(variance, 'values') else variance,
            'volatility': volatility.values.flatten()[:horizon] if hasattr(volatility, 'values') else volatility
        })
        
        return results
    
    def rolling_forecast(
        self,
        returns: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 1
    ) -> pd.DataFrame:
        """
        Perform rolling out-of-sample volatility forecast.
        
        Uses arch's last_obs: fit on data up to t-1, forecast for t.
        Each forecast is strictly out-of-sample (no future data used).
        
        Args:
            returns: Full returns series
            train_size: Initial training size (first forecast at index train_size)
            horizon: Forecast horizon (1-step ahead)
            refit_every: Refit model every N steps (1 = every step, 21 = weekly)
            
        Returns:
            DataFrame with forecast_vol, realized_vol, index=dates
        """
        logger.info(f"Starting rolling OOS forecast (train_size={train_size}, refit_every={refit_every})")
        
        returns_clean = returns.dropna()
        n = len(returns_clean)
        index = returns_clean.index
        
        am = self._create_model(returns_clean)
        
        vol_forecasts = []
        realized_vol = []
        dates = []
        
        for i in range(train_size, n):
            if (i - train_size) % refit_every != 0:
                continue
            
            try:
                res = am.fit(last_obs=index[i], disp='off')
            except Exception as e:
                logger.warning(f"Fit failed at i={i}: {e}")
                continue
            
            try:
                fc = res.forecast(horizon=horizon)
                var_val = fc.variance.iloc[0].values[0]
                vol_val = float(np.sqrt(var_val) / 100)
                vol_forecasts.append(vol_val)
                
                realized = float(np.abs(returns_clean.iloc[i]))
                realized_vol.append(realized)
                dates.append(index[i])
                
            except Exception as e:
                logger.warning(f"Forecast failed at i={i}: {e}")
                continue
        
        results = pd.DataFrame({
            'forecast_vol': vol_forecasts,
            'realized_vol': realized_vol
        }, index=dates)
        
        logger.info(f"Rolling OOS forecast complete: {len(results)} predictions")
        
        return results
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get fitted conditional volatility
        
        Returns:
            Conditional volatility series
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.conditional_volatility / 100
    
    def get_standardized_residuals(self) -> pd.Series:
        """
        Get standardized residuals
        
        Returns:
            Standardized residuals series
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.std_resid
    
    def summary(self) -> str:
        """
        Get model summary
        
        Returns:
            Summary string
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())
    
    def save(self, filename: str = "garch_model.pkl") -> None:
        """Save model to disk"""
        filepath = MODELS_DIR / filename
        
        model_data = {
            'vol_model_type': self.vol_model_type,
            'p': self.p,
            'o': self.o,
            'q': self.q,
            'mean_model': self.mean_model,
            'dist': self.dist,
            'fitted_model': self.fitted_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: str = "garch_model.pkl") -> 'GARCHModel':
        """Load model from disk"""
        filepath = MODELS_DIR / filename
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vol_model_type = model_data['vol_model_type']
        self.p = model_data['p']
        self.o = model_data['o']
        self.q = model_data['q']
        self.mean_model = model_data['mean_model']
        self.dist = model_data['dist']
        self.fitted_model = model_data['fitted_model']
        
        logger.info(f"Model loaded from {filepath}")
        
        return self
    
    def plot_volatility(
        self,
        returns: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot volatility analysis
        
        Args:
            returns: Original returns (for comparison)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        from src.thesis_plot_rc import apply_thesis_style

        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        apply_thesis_style()
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        cond_vol = self.get_conditional_volatility()
        std_resid = self.get_standardized_residuals()
        
        axes[0].plot(cond_vol.index, cond_vol.values, linewidth=0.8)
        axes[0].set_title('Conditional Volatility')
        axes[0].set_ylabel('Volatility')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(std_resid.index, std_resid.values, linewidth=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].axhline(y=2, color='orange', linestyle='--', linewidth=0.5)
        axes[1].axhline(y=-2, color='orange', linestyle='--', linewidth=0.5)
        axes[1].set_title('Standardized Residuals')
        axes[1].set_ylabel('Std Residual')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(std_resid.dropna(), bins=50, density=True, alpha=0.7)
        x = np.linspace(-4, 4, 100)
        from scipy.stats import norm
        axes[2].plot(x, norm.pdf(x), 'r-', linewidth=2, label='Normal')
        axes[2].set_title('Distribution of Standardized Residuals')
        axes[2].set_xlabel('Standardized Residual')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        
        model_name = f"{self.vol_model_type}({self.p},{self.o},{self.q})"
        plt.suptitle(f'{model_name} Model Analysis')
        plt.tight_layout()
        
        return fig

