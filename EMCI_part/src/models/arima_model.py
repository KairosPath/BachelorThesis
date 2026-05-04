"""
ARIMA Model Module
==================
ARIMA and Auto-ARIMA implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import pickle
from pathlib import Path
import logging
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, MODELS_DIR

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)

class ARIMAModel:
    """
    ARIMA Model for time series forecasting
    
    Features:
    - Automatic order selection using AIC/BIC
    - Manual order specification
    - Rolling forecast
    - Model diagnostics
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        auto_select: bool = True
    ):
        """
        Initialize ARIMA Model
        
        Args:
            order: (p, d, q) tuple for manual specification
            auto_select: Whether to automatically select order
        """
        self.order = order
        self.auto_select = auto_select
        self.model = None
        self.fitted_model = None
        self.config = config.arima
        self.history = []
    
    def _determine_d(self, series: pd.Series, max_d: int = 2) -> int:
        """
        Determine the differencing order d
        
        Args:
            series: Time series data
            max_d: Maximum differencing order
            
        Returns:
            Recommended d
        """
        current_series = series.dropna()
        
        for d in range(max_d + 1):
            result = adfuller(current_series)
            if result[1] < 0.05:
                return d
            current_series = current_series.diff().dropna()
        
        return max_d
    
    def auto_select_order(
        self,
        series: pd.Series,
        seasonal: bool = False
    ) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using AIC/BIC
        
        Args:
            series: Time series data
            seasonal: Whether to include seasonal component
            
        Returns:
            Optimal (p, d, q) order
        """
        logger.info("Auto-selecting ARIMA order...")
        
        series_clean = series.dropna()
        
        auto_model = pm.auto_arima(
            series_clean,
            start_p=0,
            start_q=0,
            max_p=self.config.max_p,
            max_q=self.config.max_q,
            max_d=self.config.max_d,
            seasonal=seasonal,
            stepwise=self.config.stepwise,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            information_criterion=self.config.information_criterion
        )
        
        order = auto_model.order
        logger.info(f"Selected order: ARIMA{order}")
        logger.info(f"AIC: {auto_model.aic():.2f}, BIC: {auto_model.bic():.2f}")
        
        return order
    
    def fit(
        self,
        series: pd.Series,
        order: Optional[Tuple[int, int, int]] = None
    ) -> 'ARIMAModel':
        """
        Fit ARIMA model to data
        
        Args:
            series: Time series data
            order: (p, d, q) order (uses auto-select if None)
            
        Returns:
            Self
        """
        series_clean = series.dropna()
        
        if order is not None:
            self.order = order
        elif self.auto_select:
            self.order = self.auto_select_order(series_clean)
        elif self.order is None:
            d = self._determine_d(series_clean)
            self.order = (1, d, 1)
        
        logger.info(f"Fitting ARIMA{self.order} model...")
        
        self.model = ARIMA(series_clean, order=self.order)
        self.fitted_model = self.model.fit()
        
        logger.info("Model fitted successfully")
        logger.info(f"AIC: {self.fitted_model.aic:.2f}")
        logger.info(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self
    
    def predict(
        self,
        steps: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Make forecast predictions
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Predictions (and optionally confidence intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast.predicted_mean
        
        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            return predictions, conf_int
        
        return predictions
    
    def rolling_forecast(
        self,
        series: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 1
    ) -> pd.DataFrame:
        """
        Perform rolling forecast
        
        Args:
            series: Full time series
            train_size: Initial training size
            horizon: Forecast horizon
            refit_every: Refit model every N steps
            
        Returns:
            DataFrame with actual and predicted values
        """
        logger.info(f"Starting rolling forecast with horizon={horizon}")
        
        series_clean = series.dropna()
        n = len(series_clean)
        
        if train_size >= n:
            raise ValueError("train_size must be less than series length")
        
        predictions = []
        actuals = []
        dates = []
        
        for i in range(train_size, n - horizon + 1):
            train_data = series_clean.iloc[:i]
            
            if (i - train_size) % refit_every == 0:
                try:
                    self.fit(train_data, order=self.order)
                except Exception as e:
                    logger.warning(f"Refit failed at step {i}: {e}")
            
            try:
                pred = self.predict(steps=horizon)
                if horizon == 1:
                    predictions.append(pred.values[0])
                else:
                    predictions.append(pred.values[-1])
                
                actual_idx = i + horizon - 1
                actuals.append(series_clean.iloc[actual_idx])
                dates.append(series_clean.index[actual_idx])
                
            except Exception as e:
                logger.warning(f"Prediction failed at step {i}: {e}")
                continue
        
        results = pd.DataFrame({
            'actual': actuals,
            'predicted': predictions
        }, index=dates)
        
        logger.info(f"Rolling forecast complete: {len(results)} predictions")
        
        return results
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals
        
        Returns:
            Residuals series
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.resid
    
    def summary(self) -> str:
        """
        Get model summary
        
        Returns:
            Summary string
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())
    
    def save(self, filename: str = "arima_model.pkl") -> None:
        """
        Save model to disk
        
        Args:
            filename: Filename for saving
        """
        filepath = MODELS_DIR / filename
        
        model_data = {
            'order': self.order,
            'fitted_model': self.fitted_model,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: str = "arima_model.pkl") -> 'ARIMAModel':
        """
        Load model from disk
        
        Args:
            filename: Filename to load from
            
        Returns:
            Self
        """
        filepath = MODELS_DIR / filename
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.order = model_data['order']
        self.fitted_model = model_data['fitted_model']
        
        logger.info(f"Model loaded from {filepath}")
        
        return self
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot model diagnostics
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        from scipy import stats
        from src.thesis_plot_rc import apply_thesis_style

        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        residuals = self.get_residuals()
        
        apply_thesis_style()
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        std_resid = (residuals - residuals.mean()) / residuals.std()
        axes[0, 0].plot(std_resid)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Standardized Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        
        axes[0, 1].hist(std_resid, bins=40, density=True, alpha=0.7)
        x = np.linspace(std_resid.min(), std_resid.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Density')
        
        stats.probplot(std_resid, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        plot_acf(residuals, ax=axes[1, 1], lags=30)
        axes[1, 1].set_title('ACF of Residuals')
        
        plt.suptitle(f'ARIMA{self.order} Diagnostics')
        plt.tight_layout()
        
        return fig

class ARIMAXModel:
    """
    ARIMAX Model — ARIMA with exogenous variables.
    Tests semi-strong EMH: do cross-asset returns (DXY, VIX, Gold, Oil, FX, etc.) help predict EEM?
    Uses 9 base feature returns as exog (parsimonious vs Ridge/Lasso's 103 features).
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        auto_select: bool = True
    ):
        self.order = order
        self.auto_select = auto_select
        self.fitted_model = None
        self.config = config.arima

    def _determine_d(self, series: pd.Series, max_d: int = 2) -> int:
        current_series = series.dropna()
        for d in range(max_d + 1):
            result = adfuller(current_series)
            if result[1] < 0.05:
                return d
            current_series = current_series.diff().dropna()
        return max_d

    def auto_select_order(
        self,
        series: pd.Series,
        exog: pd.DataFrame,
        seasonal: bool = False
    ) -> Tuple[int, int, int]:
        series_clean = series.dropna()
        exog_clean = exog.loc[series_clean.index].dropna()
        common_idx = series_clean.index.intersection(exog_clean.index)
        series_clean = series_clean.loc[common_idx]
        exog_clean = exog_clean.loc[common_idx]

        auto_model = pm.auto_arima(
            series_clean,
            X=exog_clean,
            start_p=0, start_q=0,
            max_p=self.config.max_p,
            max_q=self.config.max_q,
            max_d=self.config.max_d,
            seasonal=seasonal,
            stepwise=self.config.stepwise,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            information_criterion=self.config.information_criterion
        )
        order = auto_model.order
        logger.info(f"ARIMAX selected order: {order}")
        return order

    def fit(
        self,
        series: pd.Series,
        exog: pd.DataFrame,
        order: Optional[Tuple[int, int, int]] = None
    ) -> 'ARIMAXModel':
        series_clean = series.dropna()
        exog_clean = exog.loc[series_clean.index].dropna()
        common_idx = series_clean.index.intersection(exog_clean.index)
        series_clean = series_clean.loc[common_idx]
        exog_clean = exog_clean.loc[common_idx]

        if order is not None:
            self.order = order
        elif self.auto_select:
            self.order = self.auto_select_order(series_clean, exog_clean)
        else:
            d = self._determine_d(series_clean)
            self.order = self.order or (1, d, 1)

        logger.info(f"Fitting ARIMAX{self.order} with {exog_clean.shape[1]} exog variables...")
        self.fitted_model = SARIMAX(
            series_clean,
            exog=exog_clean,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        logger.info(f"ARIMAX fitted. AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
        return self

    def predict(
        self,
        steps: int = 1,
        exog: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        if exog is None:
            raise ValueError("ARIMAX requires exog for prediction")
        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)
        return forecast.predicted_mean

    def rolling_forecast(
        self,
        series: pd.Series,
        exog: pd.DataFrame,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21
    ) -> pd.DataFrame:
        series_clean = series.dropna()
        exog_clean = exog.reindex(series_clean.index).dropna()
        common_idx = series_clean.index.intersection(exog_clean.index)
        series_clean = series_clean.loc[common_idx]
        exog_clean = exog_clean.loc[common_idx]
        n = len(series_clean)
        index = series_clean.index

        predictions, actuals, dates = [], [], []

        for i in range(train_size, n - horizon + 1):
            if (i - train_size) % refit_every == 0:
                try:
                    self.fit(series_clean.iloc[:i], exog_clean.iloc[:i], order=self.order)
                except Exception as e:
                    logger.warning(f"ARIMAX refit failed at i={i}: {e}")
                    continue

            try:
                exog_fcast = exog_clean.iloc[i : i + horizon]
                pred = self.predict(steps=horizon, exog=exog_fcast)
                predictions.append(pred.values[-1])
                actuals.append(series_clean.iloc[i + horizon - 1])
                dates.append(index[i + horizon - 1])
            except Exception as e:
                logger.warning(f"ARIMAX forecast failed at i={i}: {e}")
                continue

        results = pd.DataFrame({'actual': actuals, 'predicted': predictions}, index=dates)
        logger.info(f"ARIMAX rolling forecast: {len(results)} predictions")
        return results

    def summary(self) -> str:
        if self.fitted_model is None:
            return "ARIMAX not fitted yet"
        return str(self.fitted_model.summary())

    def save(self, filename: str = "arimax_model.pkl") -> None:
        filepath = MODELS_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({'order': self.order, 'fitted_model': self.fitted_model}, f)
        logger.info(f"ARIMAX saved to {filepath}")

