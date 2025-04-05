5D Market Regime Classification for Bitcoin using XGBoost
Overview
This project develops a robust 5D market regime classification system for Bitcoin (BTC) using the XGBoost machine learning model. The system computes market regime vectors across multiple timeframes (4-hour, daily, and weekly) to generate a 5D vector consisting of Direction, Momentum, Volatility, Participation, and Confidence. The goal is to provide high-signal, low-noise features for training XGBoost models, enabling accurate market regime classification. This 5D vector serves as a foundational tool for downstream applications, such as determining position sizing, selecting trading strategies, or providing market context for other models.

The project includes statistical validation, feature optimization, and a multi-timeframe approach to ensure robustness. Detailed market research and feature selection were conducted at 07:16 PM PDT on Monday, March 24, 2025, drawing on financial literature, machine learning best practices, and empirical studies.

Project Structure
The project is organized under the mrdroot directory with the following structure:

core/: Core functionality for the project.
data_processing/: Scripts for processing raw Bitcoin OHLCV data.
mrd_egg_info/: Metadata for the project package.
statistical_analysis/: Scripts for statistical validation and feature evaluation.
tabs/: Utility scripts for tabulating results.
__pycache__/: Compiled Python bytecode (ignored by Git).
app.py: Main application script to run the pipeline.
logging_config.py: Configuration for logging.
utils.py: Utility functions for data processing and analysis.
__init__.py: Marks directories as Python packages.
app.log: Log file for application runtime.
daily_stats_output.json: Output file for daily statistical results.
debug_import.py: Debugging script for imports.
printer.py: Script for formatted output printing.
setup.py: Script for package installation and setup.
structure.py: Defines the project structure and schema.
verify_package.py: Script to verify package integrity.
Objective
The primary objective is to create a 5D market regime vector for Bitcoin, where:

Direction: Indicates bullish or bearish bias (price trend direction).
Momentum: Measures the rate of price change (e.g., acceleration or deceleration).
Volatility: Captures market noise and volatility levels.
Participation: Assesses volume conviction and price confirmation.
Confidence: Provides a confidence measure from the XGBoost model, used as the fifth dimension.
This 5D vector is designed for use in downstream models, such as:

Determining position sizing for trades.
Selecting appropriate trading strategies based on market regimes.
Providing market context for other predictive models.
Methodology
Background
The system computes market regime vectors at three timeframes: 4-hour, daily, and weekly. The initial feature set included 33 features, which were reduced to 15 high-signal, low-noise features after extensive market research and statistical validation. The research leveraged sources like Investopedia, QuantInsti, Kaggle, and academic papers to prioritize features with empirical evidence of predictive power.

Feature Selection
The feature selection process involved:

Market Research: Reviewed financial literature, machine learning studies, and empirical backtests to identify high-signal features.
Feature Evaluation: Assessed features for signal-to-noise ratio, computational efficiency, and out-of-sample performance.
Feature Reduction: Reduced the feature set from 33 to 15 by removing noisy or redundant metrics.
Retained Features
The following table summarizes the retained features across different timeframes:

Vector	Feature	Description	Purpose
4-Hour	directional_momentum	Log return of price change	Direction, Momentum
4-Hour	volatility_signature	Range/ATR, percentile scaled	Volatility
4-Hour	volume_intensity	Volume/SMA, percentile scaled	Participation
4-Hour	body_proportion	(Close - Open)/(High - Low)	Direction, Momentum
4-Hour	range_position	(Close - Low)/(High - Low)	Momentum
Daily	direction_slope	Linear regression slope of 4-hour closes	Direction
Daily	volatility_trajectory	Change in volatility from start to end	Volatility
Daily	volume_distribution_skew	Skew of intraday volume	Participation
Daily	body_consistency	Proportion of 4-hour candles matching daily direction	Direction, Confidence
Daily	price_level_30d	Close relative to 30-day range	Direction, Momentum
Daily	volume_zscore	Volume anomaly relative to 20-day window	Participation
Weekly	direction_value	Volume-weighted average of daily slopes	Direction
Weekly	strength_value	Proportion of days matching weekly direction	Momentum
Weekly	volatility_value	Weekly range/12-week EMA, percentile scaled	Volatility
Weekly	volume_intensity	Weekly volume/20-week SMA	Participation
Target Labels
Future price changes and volatilities at 4-hour, daily, and weekly horizons are retained as target labels for supervised learning.
Database Breakdown
The database is structured to store raw 4-hour Bitcoin data and computed features for 4-hour, daily, and weekly vectors. Each feature is calculated using statistically sound methods and validated for significance.

1. 4-Hour Vector
Table: four_hour_vectors
Columns:
timestamp (DATETIME): Start of the 4-hour period.
directional_momentum (FLOAT): Log return of price change.
volatility_signature (FLOAT): Normalized volatility (range/ATR).
volume_intensity (FLOAT): Volume z-score over 20 periods.
body_proportion (FLOAT): (Close - Open)/(High - Low).
range_position (FLOAT): (Close - Low)/(High - Low).
future_price_change_4h (FLOAT): Next 4-hour price change (target).
Validation:
Correlation matrix to detect multicollinearity.
Feature importance using XGBoost.
CSV Report: four_hour_validation.csv
2. Daily Vector
Table: daily_vectors
Columns:
date (DATE): Start of the day.
direction_slope (FLOAT): Linear regression slope of 4-hour closes.
volatility_trajectory (FLOAT): Change in volatility across the day.
volume_distribution_skew (FLOAT): Skewness of 4-hour volumes.
body_consistency (FLOAT): Proportion of 4-hour candles matching daily direction.
price_level_30d (FLOAT): Close relative to 30-day range.
volume_zscore (FLOAT): Daily volume z-score over 20 days.
future_price_change_1d (FLOAT): Next day’s price change (target).
Validation:
Variance Inflation Factor (VIF) to check multicollinearity.
Feature stability using ANOVA.
CSV Report: daily_validation.csv
3. Weekly Vector
Table: weekly_vectors
Columns:
week_start_date (DATE): Start of the week.
direction_value (FLOAT): Volume-weighted average of daily direction slopes.
strength_value (FLOAT): Proportion of days matching weekly direction.
volatility_value (FLOAT): Weekly range/12-week EMA, scaled by percentile rank.
volume_intensity (FLOAT): Weekly volume relative to 20-week SMA.
future_price_change_1w (FLOAT): Next week’s price change (target).
Validation:
Principal Component Analysis (PCA) to ensure unique variance.
Feature ablation to measure impact on validation error.
CSV Report: weekly_validation.csv
Multi-Timeframe XGBoost Models
To capture a comprehensive view of market regimes, the project implements multiple XGBoost models combining different timeframes:

4H + Daily: Captures short-term trends and intraday dynamics.
4H + Weekly: Balances short-term signals with longer-term trends.
Daily + Weekly: Provides a medium-to-long-term perspective for regime stability.
Model Architecture
Input Features:
Trend: direction_4h, direction_1d, trend_persistence.
Momentum: momentum_4h, roc_1d, weekly_roc.
Volatility: volatility_4h, vol_regime_1w, range_ratio.
Participation: participation_4h, volume_profile, vwap_dev.
Confidence: Derived as 1 - prediction std dev from the XGBoost ensemble.
Target Variable:
A 5D vector (direction, momentum, volatility, participation, confidence) is predicted using supervised learning.
Unsupervised clustering (OPTICS) is used pre-training to define "true" regimes and map them to ideal 5D vectors.
Model Configuration:
python

Collapse

Wrap

Copy
import xgboost as xgb

params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'alpha': 0.5,
    'lambda': 1.2,
    'n_estimators': 1200
}

model = xgb.XGBRegressor(**params)
model.fit(X_train[features], y_train[['direction', 'momentum', 'volatility', 'participation']])
Confidence Dimension
The fifth dimension, confidence, is computed as 1 - prediction std dev from the XGBoost tree ensemble, providing a measure of prediction uncertainty.
Validation Protocol
Walk-Forward Testing
Train on an expanding window and test on the next 30 days.
Log performance metrics (MAE, regime accuracy, confidence calibration).
Key Metrics
Vector MAE: Target < 0.15 (mean absolute error across all 5 dimensions).
Regime Accuracy: Target > 70% (match to OPTICS cluster labels).
Confidence Calibration: Target < 0.1 Brier score (confidence aligns with prediction accuracy).
Shapley Consistency: Target > 0.8 (feature importance stability across folds).
Implementation Checklist
Data Pipeline:
Compute 4H features → 1D aggregates → 1W aggregates.
Store raw numerical values in the database.
Regime Labeling:
Run OPTICS clustering weekly to define regimes.
Update vector_5d via centroid mapping.
Model Training:
Retrain daily with a 90-day lookback window.
Track feature importance drift.
Confidence Layer:
Monitor calibration curve monthly.
Alert if confidence Brier score > 0.15.
Installation
Clone the repository:
text

Collapse

Wrap

Copy
git clone https://github.com/exploor/XGBoostData.git
cd XGBoostData
Install dependencies:
text

Collapse

Wrap

Copy
pip install -r requirements.txt
Run the setup script:
text

Collapse

Wrap

Copy
python setup.py install
Usage
Run the main application:
text

Collapse

Wrap

Copy
python app.py
Check logs for runtime information:
text

Collapse

Wrap

Copy
cat app.log
Review daily statistical outputs:
text

Collapse

Wrap

Copy
cat daily_stats_output.json
Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Why This Works
Multi-Timeframe Context: Combining 4H, daily, and weekly features prevents myopic predictions and captures both short-term and long-term dynamics.
Unsupervised Ground Truth: OPTICS clustering provides an unbiased method for defining market regimes.
Confidence as Uncertainty: The confidence dimension leverages XGBoost’s tree-based standard deviation for a free uncertainty metric.
Regularized Simplicity: Conservative XGBoost parameters prevent overfitting to Bitcoin’s volatility.
This pipeline transforms raw 4-hour Bitcoin data into a robust 5D regime vector, validated for stability and predictive power, ready for use in downstream trading applications.
