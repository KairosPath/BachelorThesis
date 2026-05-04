"""
ML Models Module
================
Generic sklearn/XGBoost estimator wrapper providing the same interface
as RidgeLassoModel: fit, predict, rolling_forecast, save.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR

logger = logging.getLogger(__name__)


class MLModel:
    """
    Wrapper for any sklearn-compatible estimator (Random Forest, XGBoost, SVR, …).

    Provides the same interface as RidgeLassoModel so that notebooks can call
    .fit(), .predict(), .rolling_forecast() and .save() uniformly across all
    model types.

    Internal StandardScaler is re-fitted on each call to .fit(), so every
    rolling window is scaled independently — no look-ahead leakage.
    """

    def __init__(self, estimator, name: str = "MLModel"):
        """
        Parameters
        ----------
        estimator : sklearn-compatible estimator
            Unfitted estimator instance (will be cloned on each fit).
        name : str
            Human-readable label used for logging and file names.
        """
        self.estimator = estimator
        self.name = name
        self.scaler = StandardScaler()
        self.fitted_estimator = None
        self.feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModel":
        """
        Fit scaler and estimator on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.

        Returns
        -------
        self
        """
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        X_scaled = self.scaler.fit_transform(X)
        self.fitted_estimator = clone(self.estimator)
        self.fitted_estimator.fit(X_scaled, np.asarray(y))
        logger.debug(f"{self.name}: fitted on {len(y)} samples")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for X using the fitted scaler and estimator.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training set).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.fitted_estimator is None:
            raise ValueError(f"{self.name}: call fit() before predict()")
        X_scaled = self.scaler.transform(X)
        return self.fitted_estimator.predict(X_scaled)

    # ------------------------------------------------------------------
    # Rolling forecast
    # ------------------------------------------------------------------

    def rolling_forecast(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21,
    ) -> pd.DataFrame:
        """
        Perform a rolling one-step-ahead out-of-sample forecast.

        Replicates the logic of RidgeLassoModel.rolling_forecast so that
        results can be passed directly to ModelComparator.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature DataFrame (train + test).
        y : pd.Series
            Full target series (train + test).
        train_size : int
            Number of observations in the initial training window.
        horizon : int
            Forecast horizon (default 1).
        refit_every : int
            Refit the model every N steps (default 21 trading days).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['actual', 'predicted'] indexed by date.
        """
        logger.info(
            f"{self.name}: starting rolling forecast "
            f"(train_size={train_size}, refit_every={refit_every})"
        )

        n = len(X)
        predictions, actuals, dates = [], [], []

        for i in range(train_size, n - horizon + 1):
            if (i - train_size) % refit_every == 0:
                try:
                    self.fit(X.iloc[:i], y.iloc[:i])
                except Exception as exc:
                    logger.warning(f"{self.name}: refit failed at step {i}: {exc}")

            try:
                pred = self.predict(X.iloc[i : i + horizon])
                predictions.append(pred[0] if horizon == 1 else pred[-1])
                actual_idx = i + horizon - 1
                actuals.append(y.iloc[actual_idx])
                dates.append(y.index[actual_idx])
            except Exception as exc:
                logger.warning(f"{self.name}: prediction failed at step {i}: {exc}")

        results = pd.DataFrame(
            {"actual": actuals, "predicted": predictions}, index=dates
        )
        logger.info(f"{self.name}: rolling forecast complete — {len(results)} steps")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> None:
        """
        Serialize the fitted estimator and scaler to disk.

        Parameters
        ----------
        filename : str, optional
            File name inside MODELS_DIR.  Defaults to '<name>_model.pkl'.
        """
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_model.pkl"

        filepath = MODELS_DIR / filename
        payload = {
            "name": self.name,
            "estimator": self.estimator,
            "fitted_estimator": self.fitted_estimator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info(f"{self.name}: saved to {filepath}")

    def load(self, filename: Optional[str] = None) -> "MLModel":
        """
        Load a previously saved MLModel from disk.

        Parameters
        ----------
        filename : str, optional
            File name inside MODELS_DIR.

        Returns
        -------
        self
        """
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_model.pkl"

        filepath = MODELS_DIR / filename
        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)

        self.name = payload["name"]
        self.estimator = payload["estimator"]
        self.fitted_estimator = payload["fitted_estimator"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        logger.info(f"{self.name}: loaded from {filepath}")
        return self
