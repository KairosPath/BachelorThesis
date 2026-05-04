"""
Statistical Tests Module
========================
Implements tests for stationarity, autocorrelation, and heteroskedasticity
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]] = None
    conclusion: str = ""
    additional_info: Optional[Dict] = None

class StatisticalTests:
    """
    Collection of statistical tests for time series analysis
    
    Tests included:
    - ADF (Augmented Dickey-Fuller) - unit root test
    - KPSS - stationarity test
    - Ljung-Box - autocorrelation test
    - ARCH-LM - heteroskedasticity test
    - Jarque-Bera - normality test
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize StatisticalTests
        
        Args:
            significance_level: Significance level for hypothesis testing
        """
        self.alpha = significance_level
    
    def adf_test(
        self,
        series: pd.Series,
        regression: str = "c",
        maxlag: Optional[int] = None
    ) -> TestResult:
        """
        Augmented Dickey-Fuller test for unit root
        
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        
        Args:
            series: Time series data
            regression: Type of regression ('c', 'ct', 'ctt', 'n')
            maxlag: Maximum lag order
            
        Returns:
            TestResult with ADF test results
        """
        series_clean = series.dropna()
        
        result = adfuller(series_clean, regression=regression, maxlag=maxlag)
        
        adf_stat, p_value, used_lag, nobs, critical_values, icbest = result
        
        if p_value < self.alpha:
            conclusion = f"STATIONARY (p={p_value:.4f} < {self.alpha})"
        else:
            conclusion = f"NON-STATIONARY (p={p_value:.4f} >= {self.alpha})"
        
        return TestResult(
            test_name="Augmented Dickey-Fuller Test",
            statistic=adf_stat,
            p_value=p_value,
            critical_values=critical_values,
            conclusion=conclusion,
            additional_info={
                "used_lag": used_lag,
                "nobs": nobs,
                "ic_best": icbest
            }
        )
    
    def kpss_test(
        self,
        series: pd.Series,
        regression: str = "c",
        nlags: str = "auto"
    ) -> TestResult:
        """
        KPSS test for stationarity
        
        H0: Series is stationary
        H1: Series has a unit root (non-stationary)
        
        Note: This is the opposite of ADF!
        
        Args:
            series: Time series data
            regression: Type of regression ('c' or 'ct')
            nlags: Number of lags ('auto' or integer)
            
        Returns:
            TestResult with KPSS test results
        """
        series_clean = series.dropna()
        
        result = kpss(series_clean, regression=regression, nlags=nlags)
        
        kpss_stat, p_value, lags, critical_values = result
        
        if p_value < self.alpha:
            conclusion = f"NON-STATIONARY (p={p_value:.4f} < {self.alpha})"
        else:
            conclusion = f"STATIONARY (p={p_value:.4f} >= {self.alpha})"
        
        return TestResult(
            test_name="KPSS Test",
            statistic=kpss_stat,
            p_value=p_value,
            critical_values=critical_values,
            conclusion=conclusion,
            additional_info={"lags_used": lags}
        )
    
    def ljung_box_test(
        self,
        series: pd.Series,
        lags: Union[int, List[int]] = 10
    ) -> TestResult:
        """
        Ljung-Box test for autocorrelation
        
        H0: No autocorrelation up to lag k
        H1: Autocorrelation exists
        
        Args:
            series: Time series data (usually residuals)
            lags: Number of lags to test
            
        Returns:
            TestResult with Ljung-Box test results
        """
        series_clean = series.dropna()
        
        result = acorr_ljungbox(series_clean, lags=lags, return_df=True)
        
        lb_stat = result['lb_stat'].iloc[-1]
        p_value = result['lb_pvalue'].iloc[-1]
        
        if p_value < self.alpha:
            conclusion = f"AUTOCORRELATION DETECTED (p={p_value:.4f} < {self.alpha})"
        else:
            conclusion = f"NO SIGNIFICANT AUTOCORRELATION (p={p_value:.4f} >= {self.alpha})"
        
        return TestResult(
            test_name="Ljung-Box Test",
            statistic=lb_stat,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={
                "all_results": result.to_dict()
            }
        )
    
    def arch_test(
        self,
        series: pd.Series,
        nlags: int = 5
    ) -> TestResult:
        """
        ARCH-LM test for heteroskedasticity
        
        H0: No ARCH effects (homoskedasticity)
        H1: ARCH effects present (heteroskedasticity)
        
        Args:
            series: Time series data (usually returns)
            nlags: Number of lags
            
        Returns:
            TestResult with ARCH test results
        """
        series_clean = series.dropna()
        
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(series_clean, nlags=nlags)
        
        if lm_pvalue < self.alpha:
            conclusion = f"ARCH EFFECTS DETECTED (p={lm_pvalue:.4f} < {self.alpha})"
        else:
            conclusion = f"NO SIGNIFICANT ARCH EFFECTS (p={lm_pvalue:.4f} >= {self.alpha})"
        
        return TestResult(
            test_name="ARCH-LM Test",
            statistic=lm_stat,
            p_value=lm_pvalue,
            conclusion=conclusion,
            additional_info={
                "f_statistic": f_stat,
                "f_pvalue": f_pvalue,
                "nlags": nlags
            }
        )
    
    def jarque_bera_test(
        self,
        series: pd.Series
    ) -> TestResult:
        """
        Jarque-Bera test for normality
        
        H0: Data is normally distributed
        H1: Data is not normally distributed
        
        Args:
            series: Time series data
            
        Returns:
            TestResult with Jarque-Bera test results
        """
        series_clean = series.dropna()
        
        result = stats.jarque_bera(series_clean)
        jb_stat = result.statistic if hasattr(result, 'statistic') else result[0]
        p_value = result.pvalue if hasattr(result, 'pvalue') else result[1]
        
        skew = stats.skew(series_clean)
        kurtosis = stats.kurtosis(series_clean)
        
        if p_value < self.alpha:
            conclusion = f"NOT NORMALLY DISTRIBUTED (p={p_value:.4f} < {self.alpha})"
        else:
            conclusion = f"NORMALLY DISTRIBUTED (p={p_value:.4f} >= {self.alpha})"
        
        return TestResult(
            test_name="Jarque-Bera Test",
            statistic=jb_stat,
            p_value=p_value,
            conclusion=conclusion,
            additional_info={
                "skewness": skew,
                "kurtosis": kurtosis
            }
        )
    
    def run_all_tests(
        self,
        series: pd.Series,
        series_name: str = "Series"
    ) -> Dict[str, TestResult]:
        """
        Run all available tests on a series
        
        Args:
            series: Time series data
            series_name: Name for logging
            
        Returns:
            Dictionary of test results
        """
        logger.info(f"Running all tests for {series_name}")
        
        results = {}
        
        results['adf'] = self.adf_test(series)
        results['kpss'] = self.kpss_test(series)
        
        results['ljung_box'] = self.ljung_box_test(series)
        
        results['arch'] = self.arch_test(series)
        
        results['jarque_bera'] = self.jarque_bera_test(series)
        
        logger.info("=" * 50)
        logger.info(f"Test Results Summary for {series_name}")
        logger.info("=" * 50)
        for test_name, result in results.items():
            logger.info(f"{result.test_name}: {result.conclusion}")
        
        return results
    
    def get_acf_pacf(
        self,
        series: pd.Series,
        nlags: int = 40
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ACF and PACF values with confidence intervals
        
        Args:
            series: Time series data
            nlags: Number of lags
            
        Returns:
            Tuple of (acf_values, acf_confint, pacf_values, pacf_confint)
        """
        series_clean = series.dropna()
        
        acf_values, acf_confint = acf(series_clean, nlags=nlags, alpha=self.alpha)
        pacf_values, pacf_confint = pacf(series_clean, nlags=nlags, alpha=self.alpha)
        
        return acf_values, acf_confint, pacf_values, pacf_confint
    
    def determine_integration_order(
        self,
        series: pd.Series,
        max_diff: int = 2
    ) -> int:
        """
        Determine the order of integration (d in ARIMA)
        
        Args:
            series: Time series data
            max_diff: Maximum differencing order to try
            
        Returns:
            Integration order d
        """
        current_series = series.dropna()
        
        for d in range(max_diff + 1):
            adf_result = self.adf_test(current_series)
            kpss_result = self.kpss_test(current_series)
            
            if adf_result.p_value < self.alpha and kpss_result.p_value >= self.alpha:
                logger.info(f"Integration order d = {d}")
                return d
            
            current_series = current_series.diff().dropna()
        
        logger.warning(f"Could not achieve stationarity with d <= {max_diff}")
        return max_diff

def print_test_result(result: TestResult) -> None:
    """Pretty print a test result"""
    print("\n" + "=" * 60)
    print(f"  {result.test_name}")
    print("=" * 60)
    print(f"  Test Statistic: {result.statistic:.6f}")
    print(f"  P-Value: {result.p_value:.6f}")
    
    if result.critical_values:
        print("  Critical Values:")
        for key, value in result.critical_values.items():
            print(f"    {key}: {value:.4f}")
    
    print(f"\n  ➤ Conclusion: {result.conclusion}")
    print("=" * 60)

