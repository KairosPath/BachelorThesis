"""
Penalized Regression Models Module
==================================
Ridge, Lasso, and ElasticNet regression for prediction
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import pickle
from pathlib import Path
import logging

from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, MODELS_DIR

logger = logging.getLogger(__name__)

class RidgeLassoModel:
    """
    Penalized Regression Models for time series prediction
    
    Models:
    - Ridge: L2 regularization (shrinks coefficients)
    - Lasso: L1 regularization (feature selection)
    - ElasticNet: Combined L1 + L2 regularization
    
    Features:
    - Cross-validation for hyperparameter selection
    - Time series aware cross-validation
    - Feature importance analysis
    - Coefficient stability analysis
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        alpha: Optional[float] = None,
        l1_ratio: float = 0.5,
        cv_folds: int = 5
    ):
        """
        Initialize Penalized Regression Model
        
        Args:
            model_type: 'ridge', 'lasso', or 'elasticnet'
            alpha: Regularization strength (None for CV selection)
            l1_ratio: ElasticNet L1/L2 ratio (0-1)
            cv_folds: Number of CV folds
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.config = config.regression
        
        self._validate_model_type()
    
    def _validate_model_type(self):
        """Validate model type"""
        valid_types = ['ridge', 'lasso', 'elasticnet']
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
    
    def _cv_alpha_grid(self) -> np.ndarray:
        """Log-spaced α grid: extended upper bound for Ridge; Lasso (and ElasticNet) unchanged."""
        if self.model_type == "ridge":
            lo, hi = self.config.ridge_alpha_range
            n = self.config.ridge_n_alphas
        else:
            lo, hi = self.config.lasso_alpha_range
            n = self.config.lasso_n_alphas
        return np.logspace(np.log10(lo), np.log10(hi), n)
    
    def _create_model(self, use_cv: bool = True):
        """
        Create model instance
        
        Args:
            use_cv: Whether to use cross-validation for alpha selection
            
        Returns:
            Model instance
        """
        alphas = self._cv_alpha_grid()
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        if self.model_type == 'ridge':
            if use_cv and self.alpha is None:
                model = RidgeCV(alphas=alphas, cv=tscv)
            else:
                model = Ridge(alpha=self.alpha or 1.0)
                
        elif self.model_type == 'lasso':
            if use_cv and self.alpha is None:
                model = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
            else:
                model = Lasso(alpha=self.alpha or 1.0, max_iter=10000)
                
        else:
            if use_cv and self.alpha is None:
                model = ElasticNetCV(
                    alphas=alphas,
                    l1_ratio=self.l1_ratio,
                    cv=tscv,
                    max_iter=10000
                )
            else:
                model = ElasticNet(
                    alpha=self.alpha or 1.0,
                    l1_ratio=self.l1_ratio,
                    max_iter=10000
                )
        
        return model
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_cv: bool = True
    ) -> 'RidgeLassoModel':
        """
        Fit model to data
        
        Args:
            X: Feature DataFrame
            y: Target Series
            use_cv: Whether to use CV for alpha selection
            
        Returns:
            Self
        """
        logger.info(f"Fitting {self.model_type.upper()} model...")
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = self._create_model(use_cv=use_cv)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'alpha_'):
            self.alpha = self.model.alpha_
            logger.info(f"Selected alpha: {self.alpha:.6f}")
        
        n_nonzero = np.sum(self.model.coef_ != 0)
        logger.info(f"Non-zero coefficients: {n_nonzero}/{len(self.model.coef_)}")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def rolling_forecast(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21
    ) -> pd.DataFrame:
        """
        Perform rolling forecast
        
        Args:
            X: Full feature DataFrame
            y: Full target Series
            train_size: Initial training size
            horizon: Forecast horizon
            refit_every: Refit model every N steps
            
        Returns:
            DataFrame with actual and predicted values
        """
        logger.info(f"Starting rolling forecast with horizon={horizon}")
        
        n = len(X)
        
        predictions = []
        actuals = []
        dates = []
        
        for i in range(train_size, n - horizon + 1):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            if (i - train_size) % refit_every == 0:
                try:
                    self.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Refit failed at step {i}: {e}")
            
            try:
                X_test = X.iloc[i:i+horizon]
                pred = self.predict(X_test)
                
                if horizon == 1:
                    predictions.append(pred[0])
                else:
                    predictions.append(pred[-1])
                
                actual_idx = i + horizon - 1
                actuals.append(y.iloc[actual_idx])
                dates.append(y.index[actual_idx])
                
            except Exception as e:
                logger.warning(f"Prediction failed at step {i}: {e}")
                continue
        
        results = pd.DataFrame({
            'actual': actuals,
            'predicted': predictions
        }, index=dates)
        
        logger.info(f"Rolling forecast complete: {len(results)} predictions")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficient magnitudes)
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        importance = importance.sort_values('abs_coefficient', ascending=False)
        importance['rank'] = range(1, len(importance) + 1)
        
        return importance
    
    def get_selected_features(self, threshold: float = 1e-6) -> List[str]:
        """
        Get features selected by Lasso (non-zero coefficients)
        
        Args:
            threshold: Minimum absolute coefficient value
            
        Returns:
            List of selected feature names
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        selected = []
        for name, coef in zip(self.feature_names, self.model.coef_):
            if np.abs(coef) > threshold:
                selected.append(name)
        
        return selected
    
    def save(self, filename: Optional[str] = None) -> None:
        """Save model to disk"""
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = MODELS_DIR / filename
        
        model_data = {
            'model_type': self.model_type,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: Optional[str] = None) -> 'RidgeLassoModel':
        """Load model from disk"""
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = MODELS_DIR / filename
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.alpha = model_data['alpha']
        self.l1_ratio = model_data['l1_ratio']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        
        return self
    
    def plot_coefficients(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot coefficient values
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = self.get_feature_importance()
        top_features = importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
        
        ax.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'{self.model_type.upper()} Coefficients (α={self.alpha:.4f})')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        return fig
    
