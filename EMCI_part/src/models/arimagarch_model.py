"""
ARIMA+GARCH Models
==================
Two implementations for joint mean-variance modelling:

1. ARIMAGARCHModel  — Legacy: joint AR(p)+GARCH(1,1) via MLE (arch library).
   AR order is BIC-selected over {0…max_lags}; no MA or differencing terms.

2. TwoStepARIMAGARCHModel  — Preferred: full two-step approach.
   Step 1 (Mean): pmdarima.auto_arima selects ARIMA(p,d,q) order; the model
                  is then refit with statsmodels SARIMAX for a proper LLF.
   Step 2 (Variance): arch GARCH(1,1) with Student-t on ARIMA residuals.
   For rolling forecasts the order is fixed from the in-sample fit and only
   the SARIMAX parameters are refit at each window — order stays constant.

Returns for arch models are scaled ×100 internally for numerical stability.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from arch import arch_model

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR

logger = logging.getLogger(__name__)


class ARIMAGARCHModel:
    """
    Joint ARIMA(p) + GARCH(1,1) model estimated by MLE.

    The AR lag order is selected by BIC over {0, 1, …, max_lags}.
    When exog is provided in fit(), the mean equation becomes ARX
    (ARIMAX+GARCH); in that case set select_lags=False and assign
    best_lags manually to compare at the same AR order as ARIMA+GARCH.

    rolling_forecast() covers the pure AR+GARCH case (no exog).
    ARIMAX+GARCH is used in-sample only — for the Likelihood Ratio Test.
    """

    def __init__(
        self,
        max_lags: int = 5,
        vol_model: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        information_criterion: str = "bic",
    ):
        self.max_lags   = max_lags
        self.vol_model  = vol_model
        self.p          = p
        self.q          = q
        self.dist       = dist
        self.information_criterion = information_criterion.lower()
        self.best_lags: Optional[int] = None
        self.fitted_model = None

    # ------------------------------------------------------------------
    def _build(
        self,
        y_s: pd.Series,
        lags: int,
        exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        """Construct an arch_model with the given AR lags and optional exog."""
        if exog is not None:
            mean = "ARX"
        elif lags > 0:
            mean = "AR"
        else:
            mean = "Constant"

        kw: dict = {}
        if lags > 0:
            kw["lags"] = lags
        if exog is not None:
            kw["x"] = exog

        return arch_model(
            y_s, mean=mean,
            vol=self.vol_model, p=self.p, q=self.q,
            dist=self.dist, **kw
        )

    @staticmethod
    def _arch_forecast_x_from_result(
        res,
        block: pd.DataFrame,
        horizon: int,
    ) -> Union[dict, np.ndarray]:
        """
        Build ``x`` for ``res.forecast``.  Use ``res.model._x.shape[1]`` as the
        number of exogenous regressors — do not rely on ``len(_x_names)`` alone
        (it can disagree with ``_x`` in edge cases).  Multiple regressors: dict
        ``{name: (1, horizon)}`` as in arch docs; single: 2D ``(1, horizon)``.
        """
        x_m = getattr(res.model, "_x", None)
        if x_m is None or not hasattr(x_m, "shape") or x_m.ndim < 2:
            nx_exog = int(block.shape[1])
        else:
            nx_exog = int(x_m.shape[1])

        x_names = list(getattr(res.model, "_x_names", None) or [])
        if block.shape[1] != nx_exog:
            raise ValueError(
                f"Exog mismatch: model nx={nx_exog}, forecast block has "
                f"{block.shape[1]} columns"
            )

        def _col_for_name(nm: str):
            if nm in block.columns:
                return nm
            if nm.isdigit():
                k = int(nm)
                if k in block.columns:
                    return k
            for c in block.columns:
                if str(c) == nm:
                    return c
            raise KeyError(
                f"No column matching arch exog name {nm!r} in block; "
                f"have {list(block.columns)}"
            )

        if nx_exog == 1:
            nm0 = x_names[0] if x_names else str(block.columns[0])
            c = _col_for_name(nm0)
            return np.asarray(block[c], dtype=float).reshape(1, horizon)

        # Dict keys must match arch's self._x_names exactly; if lengths disagree,
        # fall back to 3D array (j-th panel = j-th exog column).
        if len(x_names) == nx_exog:
            return {
                x_names[j]: np.asarray(
                    block[_col_for_name(x_names[j])], dtype=float
                ).reshape(1, horizon)
                for j in range(nx_exog)
            }

        panels = [
            np.asarray(block.iloc[:, j], dtype=float).reshape(1, horizon)
            for j in range(nx_exog)
        ]
        return np.ascontiguousarray(np.array(panels, dtype=float))

    # ------------------------------------------------------------------
    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        select_lags: bool = True,
    ) -> "ARIMAGARCHModel":
        """
        Fit ARIMA+GARCH (or ARIMAX+GARCH) on training data.

        Parameters
        ----------
        y : pd.Series
            Log-return series (undifferenced).
        exog : pd.DataFrame, optional
            Exogenous variables aligned to y's index (already shifted).
        select_lags : bool
            If True, select AR lag order by BIC. If False, use self.best_lags.
        """
        y_s = y.dropna() * 100
        exog_fit = exog.loc[y_s.index] if exog is not None else None

        if select_lags:
            best_ic, best_lags = np.inf, 0
            use_aic = self.information_criterion == "aic"
            for lags in range(0, self.max_lags + 1):
                try:
                    m   = self._build(y_s, lags, exog_fit)
                    res = m.fit(disp="off", show_warning=False)
                    ic = res.aic if use_aic else res.bic
                    if ic < best_ic:
                        best_ic, best_lags = ic, lags
                except Exception:
                    continue
            self.best_lags = best_lags

        if self.best_lags is None:
            self.best_lags = 0

        m = self._build(y_s, self.best_lags, exog_fit)
        self.fitted_model = m.fit(disp="off", show_warning=False)

        tag = "ARX" if exog is not None else "AR"
        ic_name = "AIC" if self.information_criterion == "aic" else "BIC"
        ic_val = self.aic if self.information_criterion == "aic" else self.bic
        logger.info(
            f"ARIMAGARCHModel fitted: {tag}({self.best_lags})+{self.vol_model}(1,1)"
            f"  LL={self.loglikelihood:.2f}  {ic_name}={ic_val:.2f}"
        )
        return self

    # ------------------------------------------------------------------
    def rolling_forecast(
        self,
        y: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21,
        exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rolling one-step-ahead conditional mean forecast.

        Supports both AR+GARCH (no exog) and ARX+GARCH (with exog).
        Refits every `refit_every` steps using only data up to that point.
        When exog is provided, rows ``exog.iloc[i : i + horizon]`` (already
        shifted in the dataset) are passed to arch's ``forecast`` as a dict
        of columns (required for multiple regressors) — no look-ahead bias.

        Returns a DataFrame with columns ['actual', 'predicted'] compatible
        with ModelComparator.add_model_results().
        """
        if self.best_lags is None:
            raise ValueError("Call fit() before rolling_forecast().")

        n = len(y)
        preds, actuals, dates = [], [], []
        res = None

        for i in range(train_size, n - horizon + 1):
            # If there is no fitted model yet, refit every day until one succeeds.
            # Otherwise we would skip refit_every-1 days after a failed first fit.
            need_refit = (res is None) or ((i - train_size) % refit_every == 0)
            if need_refit:
                try:
                    y_w    = y.iloc[:i] * 100
                    exog_w = exog.iloc[:i] if exog is not None else None
                    m      = self._build(y_w, self.best_lags, exog_w)
                    res     = m.fit(disp="off", show_warning=False)
                except Exception as exc:
                    logger.warning(f"ARIMAGARCHModel refit failed at step {i}: {exc}")
                    if res is None:
                        continue

            if res is None:
                continue

            try:
                if exog is not None:
                    block = exog.iloc[i : i + horizon]
                    x_f = self._arch_forecast_x_from_result(res, block, horizon)
                    fc = res.forecast(horizon=horizon, reindex=False, x=x_f)
                else:
                    fc = res.forecast(horizon=horizon, reindex=False)
                pred = float(fc.mean.iloc[-1, 0]) / 100
                preds.append(pred)
                actuals.append(y.iloc[i + horizon - 1])
                dates.append(y.index[i + horizon - 1])
            except Exception as exc:
                logger.warning(f"ARIMAGARCHModel forecast failed at step {i}: {exc}")

        if not preds:
            logger.error(
                "ARIMAGARCHModel.rolling_forecast: no successful forecasts "
                "(check refit/forecast warnings above)."
            )
        return pd.DataFrame({"actual": actuals, "predicted": preds}, index=dates)

    # ------------------------------------------------------------------
    @property
    def loglikelihood(self) -> Optional[float]:
        return self.fitted_model.loglikelihood if self.fitted_model else None

    @property
    def aic(self) -> Optional[float]:
        return self.fitted_model.aic if self.fitted_model else None

    @property
    def bic(self) -> Optional[float]:
        return self.fitted_model.bic if self.fitted_model else None

    @property
    def order(self) -> tuple:
        return (self.best_lags or 0, 0, 0)

    def summary(self) -> str:
        return str(self.fitted_model.summary()) if self.fitted_model else "Not fitted"

    # ------------------------------------------------------------------
    def save(self, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = "arimagarch_model.pkl"
        filepath = MODELS_DIR / filename
        payload  = {
            "max_lags":     self.max_lags,
            "vol_model":    self.vol_model,
            "p":            self.p,
            "q":            self.q,
            "dist":         self.dist,
            "information_criterion": self.information_criterion,
            "best_lags":    self.best_lags,
            "fitted_model": self.fitted_model,
        }
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info(f"ARIMAGARCHModel saved to {filepath}")

    def load(self, filename: Optional[str] = None) -> "ARIMAGARCHModel":
        if filename is None:
            filename = "arimagarch_model.pkl"
        filepath = MODELS_DIR / filename
        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)
        self.max_lags     = payload["max_lags"]
        self.vol_model    = payload["vol_model"]
        self.p            = payload["p"]
        self.q            = payload["q"]
        self.dist         = payload["dist"]
        self.information_criterion = payload.get("information_criterion", "bic")
        self.best_lags    = payload["best_lags"]
        self.fitted_model = payload["fitted_model"]
        return self


# ──────────────────────────────────────────────────────────────────────────────
class TwoStepARIMAGARCHModel:
    """
    Two-step ARIMA(p,d,q) + GARCH(1,1) model.

    Step 1 — Mean equation
        pmdarima.auto_arima selects the full ARIMA(p,d,q) order (BIC by default).
        The model is then refit with statsmodels SARIMAX so that a proper
        log-likelihood is available for Likelihood Ratio Tests.
        Supports optional exogenous variables (ARIMAX).

    Step 2 — Variance equation
        arch GARCH(1,1) with Student-t innovations fitted on ARIMA residuals
        (mean='Zero' because the mean is already captured by Step 1).

    Rolling forecast
        The ARIMA(p,d,q) order is fixed from the initial fit (in-sample order
        selection). In each rolling window only the SARIMAX coefficients are
        refit, giving fast and unbiased OOS forecasts.
        The GARCH step is not refit in rolling mode; it is used for in-sample
        diagnostics and LRT only.

    Parameters
    ----------
    max_p, max_q, max_d : int
        Search bounds for auto_arima.
    vol_p, vol_q : int
        GARCH(p,q) order (default 1,1).
    dist : str
        Innovation distribution for GARCH ('t' for Student-t).
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
        information_criterion: str = "bic",
    ):
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.vol_p = vol_p
        self.vol_q = vol_q
        self.dist  = dist
        self.information_criterion = information_criterion

        self._arima_order:  Optional[Tuple[int, int, int]] = None
        self._arima_result = None   # statsmodels SARIMAXResults
        self._garch_result = None   # arch ARCHModelResult

    # ── properties ───────────────────────────────────────────────────────────
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

    # ── fit ──────────────────────────────────────────────────────────────────
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
        order : tuple (p,d,q), optional
            Override auto_arima — useful for nesting ARIMAX inside ARIMA for LRT.
        """
        import pmdarima as pm
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_clean   = y.dropna()
        exog_arr  = exog.loc[y_clean.index].values if exog is not None else None

        # ── Step 1a: order selection ────────────────────────────────────────
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
        p, d, q = self._arima_order
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
            f"TwoStepARIMAGARCHModel: {tag}{self._arima_order}+GARCH({self.vol_p},{self.vol_q})"
            f"  ARIMA_LL={self.arima_llf:.2f}  GARCH_LL={self.garch_llf:.2f}"
        )
        return self

    # ── rolling forecast ─────────────────────────────────────────────────────
    def rolling_forecast(
        self,
        y: pd.Series,
        train_size: int,
        horizon: int = 1,
        refit_every: int = 21,
        exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rolling one-step-ahead mean forecast using a fixed ARIMA(p,d,q) order.

        The order is taken from fit(); only the SARIMAX coefficients are refit
        every `refit_every` steps.  No look-ahead bias: exog.iloc[i] is the
        value already shifted by one trading day in the dataset.

        Returns
        -------
        pd.DataFrame with columns ['actual', 'predicted'].
        """
        if self._arima_order is None:
            raise ValueError("Call fit() before rolling_forecast().")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        n           = len(y)
        preds, actuals, dates = [], [], []
        sarimax_res = None

        for i in range(train_size, n - horizon + 1):
            if (i - train_size) % refit_every == 0:
                try:
                    y_w    = y.iloc[:i]
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
                exog_next = exog.iloc[i : i + 1].values if exog is not None else None
                fc   = sarimax_res.forecast(steps=horizon, exog=exog_next)
                pred = float(fc.iloc[-1])
                preds.append(pred)
                actuals.append(y.iloc[i + horizon - 1])
                dates.append(y.index[i + horizon - 1])
            except Exception as exc:
                logger.warning(f"TwoStepARIMAGARCH forecast failed at step {i}: {exc}")

        return pd.DataFrame({"actual": actuals, "predicted": preds}, index=dates)

    # ── diagnostics / IO ─────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = []
        if self._arima_result is not None:
            lines.append(f"=== Step 1: ARIMA{self._arima_order} (statsmodels SARIMAX) ===")
            lines.append(str(self._arima_result.summary()))
        if self._garch_result is not None:
            lines.append(
                f"\n=== Step 2: GARCH({self.vol_p},{self.vol_q}) "
                f"on ARIMA residuals (arch) ==="
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
            "dist":  self.dist,
            "information_criterion": self.information_criterion,
            "_arima_order":  self._arima_order,
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
