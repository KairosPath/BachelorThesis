"""
MSCI EM Prediction Project - Main Script
=========================================
Diploma Thesis: Econometric Regression Methods for MSCI Emerging Markets Index Prediction

This script demonstrates the complete workflow:
1. Data loading and preprocessing
2. Exploratory analysis and statistical tests
3. Model fitting (ARIMA, GARCH, Ridge, Lasso)

Author: Diploma Student
Date: 2025
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

from config import config
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.analysis.statistical_tests import StatisticalTests
from src.models.arima_model import ARIMAModel
from src.models.garch_model import GARCHModel
from src.models.regression_models import RidgeLassoModel
from src.validation.model_comparison import ModelComparator

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("  MSCI EM PREDICTION PROJECT")
    print("  Diploma Thesis: Econometric Regression Methods")
    print("=" * 70)
    
    print("\n[STEP 1] Loading and preparing data...")
    
    loader = DataLoader()
    processor = DataProcessor()
    
    eem = loader.fetch_target(start_date='2005-01-01')
    features = loader.fetch_yahoo_data(config.data.feature_tickers)
    
    if eem.empty:
        logger.error("Failed to load EEM data. Please check your internet connection.")
        return
    
    print(f"  ✓ EEM data: {len(eem)} records")
    print(f"  ✓ Features: {features.shape}")
    
    eem_clean = processor.clean_data(eem)
    features_clean = processor.clean_data(features)
    
    X, y = processor.prepare_dataset(eem_clean, features_clean)
    
    X_shifted = X.shift(1).dropna()
    y_aligned = y.loc[X_shifted.index]
    
    print(f"  ✓ Final dataset: {len(X_shifted)} samples, {X_shifted.shape[1]} features (shifted, no leakage)")
    
    print("\n[STEP 2] Running statistical tests...")
    
    tests = StatisticalTests()
    test_results = tests.run_all_tests(y_aligned, "EEM Returns")
    
    d = tests.determine_integration_order(eem_clean['Close'])
    print(f"  ✓ Recommended integration order: d = {d}")
    
    print("\n[STEP 3] Splitting data...")
    
    train_size = int(len(y_aligned) * 0.8)
    X_train, X_test = X_shifted.iloc[:train_size], X_shifted.iloc[train_size:]
    y_train, y_test = y_aligned.iloc[:train_size], y_aligned.iloc[train_size:]
    
    print(f"  ✓ Training: {len(y_train)} samples")
    print(f"  ✓ Testing: {len(y_test)} samples")
    
    print("\n[STEP 4] Fitting models...")
    
    print("  → Fitting ARIMA(0,0,0) baseline (random walk)...")
    baseline = ARIMAModel(order=(0, 0, 0), auto_select=False)
    baseline.fit(y_train)
    print(f"    ✓ ARIMA(0,0,0) baseline fitted")
    
    print("  → Fitting ARIMA (auto-select)...")
    arima = ARIMAModel(auto_select=True)
    arima.fit(y_train)
    print(f"    ✓ ARIMA{arima.order} fitted")
    
    print("  → Fitting GARCH...")
    garch = GARCHModel(vol_model='GARCH', p=1, q=1, dist='t')
    garch.fit(y_train)
    print(f"    ✓ GARCH(1,1) fitted")
    
    print("  → Fitting EGARCH...")
    egarch = GARCHModel(vol_model='EGARCH', p=1, o=1, q=1, dist='t')
    egarch.fit(y_train)
    print(f"    ✓ EGARCH(1,1,1) fitted")
    
    print("  → Fitting Ridge regression...")
    ridge = RidgeLassoModel(model_type='ridge')
    ridge.fit(X_train, y_train)
    print(f"    ✓ Ridge (α={ridge.alpha:.6f}) fitted")
    
    print("  → Fitting Lasso regression...")
    lasso = RidgeLassoModel(model_type='lasso')
    lasso.fit(X_train, y_train)
    n_selected = len(lasso.get_selected_features())
    print(f"    ✓ Lasso (α={lasso.alpha:.6f}) fitted, {n_selected} features selected")
    
    print("\n[STEP 5] Generating rolling forecasts...")
    
    baseline_results = baseline.rolling_forecast(y_aligned, train_size=train_size, horizon=1, refit_every=21)
    arima_results = arima.rolling_forecast(y_aligned, train_size=train_size, horizon=1, refit_every=21)
    ridge_results = ridge.rolling_forecast(X_shifted, y_aligned, train_size=train_size, horizon=1, refit_every=21)
    lasso_results = lasso.rolling_forecast(X_shifted, y_aligned, train_size=train_size, horizon=1, refit_every=21)
    
    print(f"  ✓ Baseline ARIMA(0,0,0) predictions: {len(baseline_results)}")
    print(f"  ✓ ARIMA{arima.order} predictions: {len(arima_results)}")
    print(f"  ✓ Ridge predictions: {len(ridge_results)}")
    print(f"  ✓ Lasso predictions: {len(lasso_results)}")
    
    print("\n[STEP 6] Comparing models...")
    
    arima_label = f"ARIMA{arima.order}"
    
    comparator = ModelComparator()
    comparator.add_model_results('RW baseline', baseline_results['actual'], baseline_results['predicted'])
    comparator.add_model_results(arima_label, arima_results['actual'], arima_results['predicted'])
    comparator.add_model_results('Ridge', ridge_results['actual'], ridge_results['predicted'])
    comparator.add_model_results('Lasso', lasso_results['actual'], lasso_results['predicted'])
    
    comparison = comparator.compare_all()
    print("\n  Model Comparison:")
    print(comparison.to_string())
    
    report = comparator.generate_report(save=True)
    
    print("\n[STEP 7] Saving models and results...")
    
    baseline.save('baseline_arima000.pkl')
    arima.save('arima_model.pkl')
    garch.save('garch_model.pkl')
    egarch.save('egarch_model.pkl')
    ridge.save('ridge_model.pkl')
    lasso.save('lasso_model.pkl')
    
    processor.save_processed_data(X, y, name='eem_dataset')
    
    print("  ✓ All models saved")
    print("  ✓ Data saved")
    
    print("\n" + "=" * 70)
    print("  PROJECT COMPLETE!")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - models/saved/")
    print("  - data/processed/")
    print("  - results/")
    print("\nRun the Jupyter notebooks for detailed analysis:")
    print("  - notebooks/01_data_exploration.ipynb")
    print("  - notebooks/02_model_fitting.ipynb")
    print("  - notebooks/03_ML_evaluation.ipynb")
    
    return {
        'comparison': comparison,
    }

if __name__ == "__main__":
    results = main()

