"""
Metrics Module
==============
Comprehensive metrics for model evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Container for metrics results"""
    rmse: float
    mae: float
    mape: float
    r2: float
    directional_accuracy: float
    hit_ratio: float
    mse: float
    correlation: float
    
    def to_dict(self) -> Dict:
        return {
            'RMSE': self.rmse,
            'MAE': self.mae,
            'MAPE': self.mape,
            'R²': self.r2,
            'Directional Accuracy': self.directional_accuracy,
            'Hit Ratio': self.hit_ratio,
            'MSE': self.mse,
            'Correlation': self.correlation
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  MODEL PERFORMANCE METRICS",
            "=" * 50,
            f"  RMSE:                 {self.rmse:.6f}",
            f"  MAE:                  {self.mae:.6f}",
            f"  MAPE:                 {self.mape:.2f}%",
            f"  R²:                   {self.r2:.4f}",
            f"  Directional Accuracy: {self.directional_accuracy:.2f}%",
            f"  Hit Ratio:            {self.hit_ratio:.2f}%",
            f"  Correlation:          {self.correlation:.4f}",
            "=" * 50
        ]
        return "\n".join(lines)

class MetricsCalculator:
    """
    Calculates various performance metrics for predictions
    
    Metrics:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R²: Coefficient of Determination
    - Directional Accuracy: % of correct direction predictions
    - Hit Ratio: % of profitable signals
    """
    
    def __init__(self):
        pass
    
    def rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size == 0:
            return float("nan")
        return float(np.sqrt(mean_squared_error(actual, predicted)))

    def mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size == 0:
            return float("nan")
        return float(mean_absolute_error(actual, predicted))
    
    def mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = actual != 0
        if not mask.any():
            return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared"""
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size < 2:
            return float("nan")
        return float(r2_score(actual, predicted))
    
    def directional_accuracy(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy
        
        Measures how often the predicted direction matches actual direction
        """
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size == 0:
            return float("nan")
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)

        correct = actual_direction == predicted_direction

        return float(np.mean(correct) * 100)
    
    def hit_ratio(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """
        Calculate hit ratio (profitable predictions)
        
        A prediction is a "hit" if:
        - Predicted positive and actual positive
        - Predicted negative and actual negative
        """
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size == 0:
            return float("nan")
        signals = np.sign(predicted)

        strategy_returns = signals * actual

        return float(np.mean(strategy_returns > 0) * 100)
    
    def correlation(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Pearson correlation"""
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        if actual.size < 2:
            return float("nan")
        c = np.corrcoef(actual, predicted)[0, 1]
        return float(c) if not np.isnan(c) else float("nan")
    
    def diebold_mariano_test(
        self,
        actual: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        loss: str = 'squared',
        h: int = 1
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        Uses Harvey, Leybourne & Newbold (1997) small-sample correction:
        t_{n-1} distribution instead of standard normal.

        H0: Equal predictive accuracy
        H1: Unequal predictive accuracy

        Args:
            actual: Actual values
            pred1: Predictions from model 1 (challenger)
            pred2: Predictions from model 2 (benchmark)
            loss: Loss function ('squared' or 'absolute')
            h: Forecast horizon (1 for one-step-ahead)

        Returns:
            Tuple of (DM statistic, p-value). The p-value is **two-sided**
            (H0: E[d]=0 vs H1: E[d]≠0), using `2 * (1 - t.cdf(|t|, df))`.
        """
        from scipy import stats

        if loss == 'squared':
            e1 = (actual - pred1) ** 2
            e2 = (actual - pred2) ** 2
        else:
            e1 = np.abs(actual - pred1)
            e2 = np.abs(actual - pred2)

        d = e1 - e2
        n = len(d)
        if n < 2:
            return float("nan"), float("nan")
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        if var_d == 0 or not np.isfinite(var_d):
            return float("nan"), float("nan")

        hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = mean_d / np.sqrt(var_d / n) * hln_correction

        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

        return float(dm_stat), float(p_value)
    
    def calculate_all(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray]
    ) -> MetricsResult:
        """
        Calculate all metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MetricsResult with all metrics
        """
        paired = pd.DataFrame({"actual": actual, "predicted": predicted})
        paired = paired.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if paired.empty:
            nan = float("nan")
            return MetricsResult(
                rmse=nan,
                mae=nan,
                mape=nan,
                r2=nan,
                directional_accuracy=nan,
                hit_ratio=nan,
                mse=nan,
                correlation=nan,
            )
        actual = paired["actual"].to_numpy(dtype=float, copy=False)
        predicted = paired["predicted"].to_numpy(dtype=float, copy=False)

        return MetricsResult(
            rmse=self.rmse(actual, predicted),
            mae=self.mae(actual, predicted),
            mape=self.mape(actual, predicted),
            r2=self.r2(actual, predicted),
            directional_accuracy=self.directional_accuracy(actual, predicted),
            hit_ratio=self.hit_ratio(actual, predicted),
            mse=mean_squared_error(actual, predicted),
            correlation=self.correlation(actual, predicted)
        )
    
    def summary_report(
        self,
        results: Dict[str, pd.DataFrame],
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate summary report comparing multiple models
        
        Args:
            results: Dictionary mapping model names to DataFrames with 'actual' and 'predicted'
            model_names: Optional custom names for models
            
        Returns:
            Summary DataFrame
        """
        summary = []
        
        for name, df in results.items():
            metrics = self.calculate_all(df['actual'], df['predicted'])
            row = metrics.to_dict()
            row['Model'] = name
            summary.append(row)
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.set_index('Model')
        
        return summary_df

