"""
Two-step ARIMA(p,d,q) + GARCH(1,1) model.

Step 1 (Mean): pmdarima.auto_arima selects the full ARIMA(p,d,q) order.
               Refit with statsmodels SARIMAX for a proper log-likelihood (LRT).
               Supports optional exogenous variables (ARIMAX).

Step 2 (Variance): arch GARCH(1,1) with Student-t on ARIMA residuals.
               Fitted in-sample; standardized residuals available for diagnostics.

Rolling forecast: ARIMA order is fixed from in-sample fit; only SARIMAX
               coefficients are refit every `refit_every` steps — fast, no bias.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR

logger = logging.getLogger(__name__)


class TwoStepARIMAGARCHModel:
    """
    Two-step ARIMA(p,d,q) + GARCH(1,1) model.

    Step 1 — Mean equation
        pmdarima.auto_arima selects the full ARIMA(p,d,q) order (BIC by default).
        The model is refit with statsmodels SARIMAX so that a proper
        log-likelihood is available for Likelihood Ratio Tests.

    Step 2 — Variance equation
        arch GARCH(1,1) with Student-t innovations on ARIMA residuals
        (mean='Zero' — mean already captured by Step 1).

    Rolling forecast
        The ARIMA(p,d,q) order is fixed from the initial fit.
        Only SARIMAX coefficients are refit at each window — fast and unbiased.

    Parameters
    ----------
    max_p, max_q, max_d : int
        Search bounds for auto_arima.
    vol_p, vol_q : int
        GARCH(p, q) order.
    dist : str
        Innovation distribution for GARCH ('t' = Student-t).
    information_criterion : str
        'bic' or 'aic' for auto_arima order selection.
    """

    def __init__(
        self,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 1,
        vol_p: int = 1,
        vol_q: int = 1,
        dist: str = "t",
        information_criterion: str = "aic",
    ):
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.vol_p = vol_p
        self.vol_q = vol_q
        self.dist = dist
        self.information_criterion = information_criterion

        self._arima_order: Optional[Tuple[int, int, int]] = None
        self._arima_result = None   # statsmodels SARIMAXResults
        self._garch_result = None   # arch ARCHModelResult

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def order(self) -> Optional[Tuple[int, int, int]]:
        return self._arima_order

    @property
    def arima_llf(self) -> Optional[float]:
        return self._arima_result.llf if self._arima_result is not None else None

    @property
    def garch_llf(self) -> Optional[float]:
        return self._garch_result.loglikelihood if self._garch_result is not None else None

    @property
    def std_resid(self) -> Optional[pd.Series]:
        return self._garch_result.std_resid if self._garch_result is not None else None

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        order: Optional[Tuple[int, int, int]] = None,
    ) -> "TwoStepARIMAGARCHModel":
        """
        Fit the two-step model.

        Parameters
        ----------
        y : pd.Series
            Log-return series (undifferenced, aligned index).
        exog : pd.DataFrame, optional
            Exogenous regressors aligned to y (already lag-shifted in dataset).
        order : tuple (p, d, q), optional
            If given, fixes (p,d,q) and skips auto_arima. If None, auto_arima runs
            on y alone (no exog) or on y with exog — use None for separate ARIMAX
            order selection.
        """
        import pmdarima as pm
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_clean = y.dropna()
        exog_arr = exog.loc[y_clean.index].values if exog is not None else None

        # ── Step 1a: order selection ─────────────────────────────────────────
        if order is not None:
            self._arima_order = order
        else:
            auto = pm.auto_arima(
                y_clean,
                X=exog_arr,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                information_criterion=self.information_criterion,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            self._arima_order = auto.order

        # ── Step 1b: refit with SARIMAX for proper LLF ──────────────────────
        sm = SARIMAX(
            y_clean,
            exog=exog_arr,
            order=self._arima_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._arima_result = sm.fit(disp=False, method="lbfgs")

        # ── Step 2: GARCH(1,1) on ARIMA residuals ───────────────────────────
        resid = pd.Series(
            self._arima_result.resid,
            index=y_clean.index[-len(self._arima_result.resid):],
        ).dropna()

        gm = arch_model(
            resid * 100,
            mean="Zero",
            vol="GARCH",
            p=self.vol_p,
            q=self.vol_q,
            dist=self.dist,
        )
        self._garch_result = gm.fit(disp="off", show_warning=False)

        tag = "ARIMAX" if exog is not None else "ARIMA"
        logger.info(
            f"TwoStepARIMAGARCH: {tag}{self._arima_order}+GARCH({self.vol_p},{self.vol_q})"
            f"  ARIMA_LL={self.arima_llf:.2f}  GARCH_LL={self.garch_llf:.2f}"
        )
        return self

    # ── rolling forecast ────────────────────────────────────────────────────

    def rolling_forecast(
        self,
        y: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21,
        exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rolling one-step-ahead mean forecast (fixed ARIMA order).

        The ARIMA(p,d,q) order from fit() is used unchanged throughout.
        Only SARIMAX coefficients are refit every `refit_every` steps.
        exog.iloc[i] is the value already shifted by one day in the dataset —
        no look-ahead bias.

        Returns
        -------
        pd.DataFrame with columns ['actual', 'predicted'].
        """
        if self._arima_order is None:
            raise ValueError("Call fit() before rolling_forecast().")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        n = len(y)
        preds, actuals, dates = [], [], []
        sarimax_res = None

        for i in range(train_size, n - horizon + 1):
            if (i - train_size) % refit_every == 0:
                try:
                    y_w = y.iloc[:i]
                    exog_w = exog.iloc[:i].values if exog is not None else None
                    m = SARIMAX(
                        y_w,
                        exog=exog_w,
                        order=self._arima_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    sarimax_res = m.fit(disp=False, method="lbfgs")
                except Exception as exc:
                    logger.warning(f"TwoStepARIMAGARCH refit failed at step {i}: {exc}")
                    continue

            if sarimax_res is None:
                continue

            try:
                exog_next = exog.iloc[i: i + 1].values if exog is not None else None
                fc = sarimax_res.forecast(steps=horizon, exog=exog_next)
                pred = float(fc.iloc[-1])
                preds.append(pred)
                actuals.append(y.iloc[i + horizon - 1])
                dates.append(y.index[i + horizon - 1])
            except Exception as exc:
                logger.warning(f"TwoStepARIMAGARCH forecast failed at step {i}: {exc}")

        return pd.DataFrame({"actual": actuals, "predicted": preds}, index=dates)

    # ── diagnostics / IO ────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = []
        if self._arima_result is not None:
            lines.append(f"=== Step 1: ARIMA{self._arima_order} (statsmodels SARIMAX) ===")
            lines.append(str(self._arima_result.summary()))
        if self._garch_result is not None:
            lines.append(
                f"\n=== Step 2: GARCH({self.vol_p},{self.vol_q}) "
                "on ARIMA residuals (arch) ==="
            )
            lines.append(str(self._garch_result.summary()))
        return "\n".join(lines) if lines else "Not fitted"

    def save(self, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = "twostep_arimagarch.pkl"
        filepath = MODELS_DIR / filename
        payload = {
            "max_p": self.max_p, "max_q": self.max_q, "max_d": self.max_d,
            "vol_p": self.vol_p, "vol_q": self.vol_q,
            "dist": self.dist,
            "information_criterion": self.information_criterion,
            "_arima_order": self._arima_order,
            "_arima_result": self._arima_result,
            "_garch_result": self._garch_result,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info(f"TwoStepARIMAGARCHModel saved to {filepath}")

    def load(self, filename: Optional[str] = None) -> "TwoStepARIMAGARCHModel":
        if filename is None:
            filename = "twostep_arimagarch.pkl"
        filepath = MODELS_DIR / filename
        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)
        for k, v in payload.items():
            setattr(self, k, v)
        logger.info(f"TwoStepARIMAGARCHModel loaded from {filepath}")
        return self
