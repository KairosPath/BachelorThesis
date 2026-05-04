"""
Exploratory Analysis Module
===========================
Comprehensive exploratory data analysis for time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, FIGURES_DIR
from src.thesis_plot_rc import apply_thesis_style, as_pdf_filename
from .statistical_tests import StatisticalTests, print_test_result

logger = logging.getLogger(__name__)

apply_thesis_style()
sns.set_palette("husl")

class TimeSeriesAnalyzer:
    """
    Comprehensive time series exploratory analysis
    
    Features:
    - Descriptive statistics
    - Distribution analysis
    - Autocorrelation analysis
    - Trend and seasonality decomposition
    - Statistical tests
    - Visualization
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize TimeSeriesAnalyzer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.tests = StatisticalTests()
    
    def descriptive_stats(
        self,
        series: pd.Series,
        name: str = "Series"
    ) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            series: Time series data
            name: Series name
            
        Returns:
            DataFrame with statistics
        """
        series_clean = series.dropna()
        
        stats_dict = {
            'Count': len(series_clean),
            'Mean': series_clean.mean(),
            'Std Dev': series_clean.std(),
            'Variance': series_clean.var(),
            'Min': series_clean.min(),
            '25%': series_clean.quantile(0.25),
            'Median': series_clean.median(),
            '75%': series_clean.quantile(0.75),
            'Max': series_clean.max(),
            'Range': series_clean.max() - series_clean.min(),
            'IQR': series_clean.quantile(0.75) - series_clean.quantile(0.25),
            'Skewness': series_clean.skew(),
            'Kurtosis': series_clean.kurtosis(),
            'Start Date': series_clean.index.min(),
            'End Date': series_clean.index.max(),
        }
        
        stats_df = pd.DataFrame([stats_dict], index=[name]).T
        
        return stats_df
    
    def plot_time_series(
        self,
        series: pd.Series,
        title: str = "Time Series",
        ylabel: str = "Value",
        save: bool = False,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series with trend line
        
        Args:
            series: Time series data
            title: Plot title
            ylabel: Y-axis label
            save: Whether to save the figure
            filename: Custom filename for saving
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(series.index, series.values, linewidth=1, alpha=0.8, label='Original')
        
        if len(series) > 21:
            rolling_mean = series.rolling(window=21).mean()
            ax.plot(rolling_mean.index, rolling_mean.values, 
                   linewidth=2, color='red', label='21-day MA')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(filename or f"{title.replace(' ', '_').lower()}.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
            logger.info(f"Saved figure: {fname}")
        
        return fig
    
    def plot_distribution(
        self,
        series: pd.Series,
        title: str = "Distribution",
        save: bool = False,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution with histogram and KDE
        
        Args:
            series: Time series data
            title: Plot title
            save: Whether to save the figure
            filename: Custom filename
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        series_clean = series.dropna()
        
        axes[0].hist(series_clean, bins=50, density=True, alpha=0.7, 
                    color='steelblue', edgecolor='white')
        series_clean.plot.kde(ax=axes[0], color='red', linewidth=2)
        axes[0].set_title('Histogram + KDE')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        
        axes[1].boxplot(series_clean, vert=True)
        axes[1].set_title('Box Plot')
        axes[1].set_ylabel('Value')
        
        from scipy import stats
        stats.probplot(series_clean, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (vs Normal)')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(filename or f"{title.replace(' ', '_').lower()}_dist.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
        
        return fig
    
    def plot_acf_pacf(
        self,
        series: pd.Series,
        lags: int = 40,
        title: str = "ACF/PACF Analysis",
        save: bool = False,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ACF and PACF
        
        Args:
            series: Time series data
            lags: Number of lags
            title: Plot title
            save: Whether to save
            filename: Custom filename
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        series_clean = series.dropna()
        
        plot_acf(series_clean, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Correlation')
        
        plot_pacf(series_clean, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Correlation')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(filename or f"{title.replace(' ', '_').lower()}_acf_pacf.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
        
        return fig
    
    def plot_returns_analysis(
        self,
        returns: pd.Series,
        title: str = "Returns Analysis",
        save: bool = False
    ) -> plt.Figure:
        """
        Comprehensive returns analysis plot
        
        Args:
            returns: Returns series
            title: Plot title
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        returns_clean = returns.dropna()
        
        axes[0, 0].plot(returns_clean.index, returns_clean.values, linewidth=0.8)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Returns Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Return')
        
        squared_returns = returns_clean ** 2
        axes[0, 1].plot(squared_returns.index, squared_returns.values, linewidth=0.8)
        axes[0, 1].set_title('Squared Returns (Volatility)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Squared Return')
        
        axes[1, 0].hist(returns_clean, bins=50, density=True, alpha=0.7,
                       color='steelblue', edgecolor='white')
        returns_clean.plot.kde(ax=axes[1, 0], color='red', linewidth=2)
        
        x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
        from scipy import stats
        normal_pdf = stats.norm.pdf(x, returns_clean.mean(), returns_clean.std())
        axes[1, 0].plot(x, normal_pdf, 'g--', linewidth=2, label='Normal')
        axes[1, 0].legend()
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Density')
        
        rolling_vol = returns_clean.rolling(window=21).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_vol.index, rolling_vol.values, linewidth=1)
        axes[1, 1].set_title('Rolling Volatility (21-day, Annualized)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(f"{title.replace(' ', '_').lower()}.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
        save: bool = False,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            df: DataFrame with multiple series
            title: Plot title
            save: Whether to save
            filename: Custom filename
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr_matrix = df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5)
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(filename or f"{title.replace(' ', '_').lower()}.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
        
        return fig
    
    def decompose_series(
        self,
        series: pd.Series,
        period: int = 252,
        model: str = "additive",
        title: str = "Time Series Decomposition",
        save: bool = False
    ) -> plt.Figure:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            series: Time series data
            period: Period for seasonal component
            model: 'additive' or 'multiplicative'
            title: Plot title
            save: Whether to save
            
        Returns:
            Matplotlib figure
        """
        series_clean = series.dropna()
        
        decomposition = seasonal_decompose(series_clean, model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        axes[0].plot(series_clean.index, series_clean.values)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Value')
        
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_title('Trend')
        axes[1].set_ylabel('Value')
        
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_title('Seasonal')
        axes[2].set_ylabel('Value')
        
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_title('Residual')
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Date')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if save:
            fname = as_pdf_filename(f"{title.replace(' ', '_').lower()}.pdf")
            fig.savefig(FIGURES_DIR / fname, bbox_inches='tight')
        
        return fig
    
    def full_analysis(
        self,
        series: pd.Series,
        name: str = "Series",
        is_returns: bool = False,
        save_figures: bool = True
    ) -> Dict:
        """
        Perform complete exploratory analysis
        
        Args:
            series: Time series data
            name: Series name for titles
            is_returns: Whether series is returns (affects analysis type)
            save_figures: Whether to save all figures
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Starting full analysis for: {name}")
        
        results = {}
        
        logger.info("Computing descriptive statistics...")
        results['stats'] = self.descriptive_stats(series, name)
        print("\n" + "="*60)
        print(f"  DESCRIPTIVE STATISTICS: {name}")
        print("="*60)
        print(results['stats'].to_string())
        
        logger.info("Running statistical tests...")
        results['tests'] = self.tests.run_all_tests(series, name)
        for test_name, result in results['tests'].items():
            print_test_result(result)
        
        logger.info("Determining integration order...")
        results['integration_order'] = self.tests.determine_integration_order(series)
        print(f"\n  ➤ Recommended Integration Order (d): {results['integration_order']}")
        
        logger.info("Generating plots...")
        
        results['fig_ts'] = self.plot_time_series(
            series, title=f"{name} - Time Series",
            save=save_figures
        )
        
        results['fig_dist'] = self.plot_distribution(
            series, title=f"{name} - Distribution",
            save=save_figures
        )
        
        results['fig_acf'] = self.plot_acf_pacf(
            series, title=f"{name} - ACF/PACF",
            save=save_figures
        )
        
        if is_returns:
            results['fig_returns'] = self.plot_returns_analysis(
                series, title=f"{name} - Returns Analysis",
                save=save_figures
            )
        
        logger.info(f"Analysis complete for: {name}")
        
        return results

