"""
Model Comparison Module
=======================
Comprehensive model comparison and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .metrics import MetricsCalculator, MetricsResult

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, FIGURES_DIR, RESULTS_DIR
from src.thesis_plot_rc import apply_thesis_style, as_pdf_filename

logger = logging.getLogger(__name__)

class ModelComparator:
    """
    Compare multiple models and generate comparison reports
    
    Features:
    - Multi-model comparison
    - Time period analysis
    - Statistical tests for significance
    - Visualization
    """
    
    def __init__(self):
        self.metrics = MetricsCalculator()
        self.results = {}
    
    def add_model_results(
        self,
        name: str,
        actual: pd.Series,
        predicted: pd.Series
    ) -> None:
        """
        Add model results for comparison
        
        Args:
            name: Model name
            actual: Actual values
            predicted: Predicted values
        """
        df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
        self.results[name] = df

        logger.info(f"Added results for model: {name} ({len(df)} aligned rows)")
    
    def compare_all(self) -> pd.DataFrame:
        """
        Compare all added models
        
        Returns:
            Comparison DataFrame
        """
        if not self.results:
            raise ValueError("No model results added yet")
        
        return self.metrics.summary_report(self.results)
    
    def statistical_significance_test(
        self,
        model1: str,
        model2: str,
        loss: str = 'squared'
    ) -> Dict:
        """
        Test statistical significance of difference between two models
        
        Args:
            model1: First model name
            model2: Second model name
            loss: Loss function for DM test
            
        Returns:
            Dictionary with test results
        """
        if model1 not in self.results or model2 not in self.results:
            raise ValueError("Both models must be added first")
        
        df1 = self.results[model1]
        df2 = self.results[model2]
        
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) == 0:
            nan = float("nan")
            return {
                "model1": model1,
                "model2": model2,
                "dm_statistic": nan,
                "dm_pvalue": nan,
                "rmse1": nan,
                "rmse2": nan,
                "conclusion": "No overlapping dates between models",
            }
        actual = df1.loc[common_idx, "actual"].values
        pred1 = df1.loc[common_idx, "predicted"].values
        pred2 = df2.loc[common_idx, "predicted"].values

        dm_stat, dm_pvalue = self.metrics.diebold_mariano_test(
            actual, pred1, pred2, loss=loss
        )
        
        rmse1 = self.metrics.rmse(actual, pred1)
        rmse2 = self.metrics.rmse(actual, pred2)
        
        if dm_pvalue < 0.05:
            if rmse1 < rmse2:
                conclusion = f"{model1} is significantly better"
            else:
                conclusion = f"{model2} is significantly better"
        else:
            conclusion = "No significant difference"
        
        return {
            'model1': model1,
            'model2': model2,
            'dm_statistic': dm_stat,
            'dm_pvalue': dm_pvalue,
            'rmse1': rmse1,
            'rmse2': rmse2,
            'conclusion': conclusion
        }
    
    def plot_predictions(
        self,
        model_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 8),
        save: bool = False
    ) -> plt.Figure:
        """
        Plot predictions vs actual for multiple models
        
        Args:
            model_names: Models to plot (None = all)
            start_date: Start date for plot
            end_date: End date for plot
            figsize: Figure size
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        apply_thesis_style()
        fig, ax = plt.subplots(figsize=figsize)
        
        first_model = model_names[0]
        actual = self.results[first_model]['actual']
        
        if start_date:
            actual = actual[actual.index >= start_date]
        if end_date:
            actual = actual[actual.index <= end_date]
        
        ax.plot(actual.index, actual.values, 'k-', linewidth=1.5, 
               label='Actual', alpha=0.8)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for model_name, color in zip(model_names, colors):
            predicted = self.results[model_name]['predicted']
            
            if start_date:
                predicted = predicted[predicted.index >= start_date]
            if end_date:
                predicted = predicted[predicted.index <= end_date]
            
            ax.plot(predicted.index, predicted.values, '--', 
                   linewidth=1, label=model_name, color=color, alpha=0.7)
        
        ax.set_title('Model Predictions vs Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(FIGURES_DIR / as_pdf_filename('predictions_comparison.pdf'), bbox_inches='tight')
        
        return fig
    
    def generate_report(
        self,
        save: bool = True
    ) -> str:
        """
        Generate comprehensive comparison report
        
        Args:
            save: Whether to save report to file
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("  MODEL COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        comparison = self.compare_all()
        lines.append("OVERALL PERFORMANCE")
        lines.append("-" * 40)
        lines.append(comparison.to_string())
        lines.append("")
        
        lines.append("BEST MODELS BY METRIC")
        lines.append("-" * 40)
        
        lower_better = ['RMSE', 'MAE', 'MAPE', 'MSE']
        for col in comparison.columns:
            if col in lower_better:
                best = comparison[col].idxmin()
            else:
                best = comparison[col].idxmax()
            lines.append(f"  {col}: {best}")
        
        lines.append("")
        
        if len(self.results) >= 2:
            lines.append("STATISTICAL SIGNIFICANCE TESTS")
            lines.append("-" * 40)
            
            model_names = list(self.results.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    test = self.statistical_significance_test(
                        model_names[i], model_names[j]
                    )
                    lines.append(f"  {test['model1']} vs {test['model2']}:")
                    lines.append(f"    DM statistic: {test['dm_statistic']:.4f}")
                    lines.append(f"    p-value: {test['dm_pvalue']:.4f}")
                    lines.append(f"    Conclusion: {test['conclusion']}")
                    lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save:
            with open(RESULTS_DIR / 'model_comparison_report.txt', 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {RESULTS_DIR / 'model_comparison_report.txt'}")
        
        return report

