import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import logging
from typing import Dict, Any, Optional, Tuple, List

# Set non-interactive backend for matplotlib
plt.switch_backend('Agg')

from mrd.statistical_analysis.summary.four_hour_stats import generate_four_hour_stats
from mrd.data_processing.data_loader import load_four_hour_features

# Configure logger for this module
logger = logging.getLogger(__name__)

# Visualization settings (modify these to change formatting across all plots)
FIGURE_SIZE = (2, 1.5)           # Width, height in inches for all figures
TITLE_FONTSIZE = 5             # Font size for plot titles
AXIS_LABEL_FONTSIZE = 4          # Font size for axis labels (x and y)
TICK_LABEL_FONTSIZE = 3          # Font size for tick labels on axes
LEGEND_FONTSIZE = 3              # Font size for legends
GRID_ALPHA = 0.3                 # Transparency of grid lines (0 to 1)
ERROR_TEXT_FONTSIZE = 3          # Font size for error messages in plots
ERROR_TEXT_COLOR = 'red'         # Color for error message text

def analyze_four_hour_data(stats_results: Dict[str, Dict[str, Any]], 
                           df: pd.DataFrame,
                           start_date: str,
                           end_date: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process statistical results for 4-hour market data, generating visualizations
    and explanations for each feature and test.

    Parameters:
    - stats_results: Dict containing statistical test results from four_hour_stats
    - df: DataFrame with 4-hour market data
    - start_date: Start date string in format 'YYYY-MM-DD'
    - end_date: End date string in format 'YYYY-MM-DD'

    Returns:
    - Dict with structure {feature: {test_type: {visualization, explanation, hypothesis, significance, model_implications}}}
    """
    logger.info(f"Starting analysis for period {start_date} to {end_date}")
    
    if not stats_results or 'error' in stats_results:
        error_msg = stats_results.get('error', 'Unknown error in stats_results')
        logger.error(f"Cannot analyze: {error_msg}")
        return {}
        
    analysis_results = {}
    feature_descriptions = {
        "directional_momentum": "Measures the log return of price changes over 4-hour periods, indicating the strength and direction of price movement.",
        "volatility_signature": "Captures normalized volatility relative to recent history, using range divided by ATR.",
        "volume_intensity": "Quantifies trading volume relative to recent averages, reflecting market participation.",
        "body_proportion": "Represents the proportion of the candle body to its range, indicating momentum strength.",
        "range_position": "Measures where the closing price lies within the high-low range, reflecting momentum."
    }

    # Validate required columns in DataFrame
    required_columns = ["future_price_change_4h", "high_volatility"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in DataFrame: {missing_columns}")
        return {}

    for feature, tests in stats_results.items():
        if feature == 'error':
            logger.error(f"Error in stats_results: {tests}")
            continue
            
        logger.info(f"Processing feature: {feature}")
        analysis_results[feature] = {}
        
        # Generate feature summary
        summary_text = feature_descriptions.get(feature, "No description available.")
        analysis_results[feature]["summary"] = {"explanation": summary_text}

        # Analyze each test for the feature
        for test_type, test_results in tests.items():
            logger.info(f"Processing test: {test_type} for feature: {feature}")
            try:
                if 'error' in test_results:
                    error_msg = test_results['error']
                    logger.error(f"Error in test results for {feature} - {test_type}: {error_msg}")
                    analysis_results[feature][test_type] = handle_analysis_error(feature, test_type, Exception(error_msg))
                    continue
                    
                # Process test results based on type
                if test_type == "correlation":
                    analysis_results[feature][test_type] = generate_correlation_analysis(feature, test_results, df)
                elif test_type == "distribution":
                    analysis_results[feature][test_type] = generate_distribution_analysis(feature, test_results, df)
                elif test_type == "stationarity":
                    analysis_results[feature][test_type] = generate_stationarity_analysis(feature, test_results, df)
                elif test_type == "mutual_information":
                    analysis_results[feature][test_type] = generate_mutual_information_analysis(feature, test_results, df)
                elif test_type == "t_test":
                    analysis_results[feature][test_type] = generate_t_test_analysis(feature, test_results, df)
                else:
                    logger.warning(f"Unknown test type: {test_type}")
                    analysis_results[feature][test_type] = {
                        "visualization": None,
                        "explanation": f"Unknown test type: {test_type}",
                        "hypothesis": "N/A",
                        "significance": "N/A",
                        "model_implications": "N/A"
                    }
            except Exception as e:
                logger.error(f"Error analyzing {test_type} for {feature}: {str(e)}", exc_info=True)
                analysis_results[feature][test_type] = handle_analysis_error(feature, test_type, e)

    logger.info("Completed four hour analysis")
    return analysis_results

def generate_correlation_analysis(feature_name: str,
                                  test_results: Dict[str, Any],
                                  df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for correlation test results.
    """
    try:
        target = test_results.get('target', 'future_price_change_4h')
        fig = create_scatter_plot(df[feature_name], df[target],
                                  x_label=feature_name, y_label=target)
        explanation, hypothesis, significance = generate_correlation_explanation(test_results, feature_name, target)
        model_implications = generate_model_implications(feature_name, "correlation", test_results)
        return {
            "visualization": fig,
            "explanation": explanation,
            "hypothesis": hypothesis,
            "significance": significance,
            "model_implications": model_implications
        }
    except Exception as e:
        logger.error(f"Error generating correlation analysis for {feature_name}: {str(e)}", exc_info=True)
        raise

def generate_distribution_analysis(feature_name: str,
                                   test_results: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for distribution test results.
    """
    try:
        fig = create_histogram_plot(df[feature_name], feature_name)
        explanation, hypothesis, significance = generate_distribution_explanation(test_results, feature_name)
        model_implications = generate_model_implications(feature_name, "distribution", test_results)
        return {
            "visualization": fig,
            "explanation": explanation,
            "hypothesis": hypothesis,
            "significance": significance,
            "model_implications": model_implications
        }
    except Exception as e:
        logger.error(f"Error generating distribution analysis for {feature_name}: {str(e)}", exc_info=True)
        raise

def generate_stationarity_analysis(feature_name: str,
                                   test_results: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for stationarity test results.
    """
    try:
        fig = create_rolling_statistics_plot(df[feature_name])
        explanation, hypothesis, significance = generate_stationarity_explanation(test_results, feature_name)
        model_implications = generate_model_implications(feature_name, "stationarity", test_results)
        return {
            "visualization": fig,
            "explanation": explanation,
            "hypothesis": hypothesis,
            "significance": significance,
            "model_implications": model_implications
        }
    except Exception as e:
        logger.error(f"Error generating stationarity analysis for {feature_name}: {str(e)}", exc_info=True)
        raise

def generate_mutual_information_analysis(feature_name: str,
                                         test_results: Dict[str, Any],
                                         df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate explanation for mutual information test results (no visualization).
    """
    try:
        target = test_results.get('target', 'future_price_change_4h')
        explanation, hypothesis, significance = generate_mutual_information_explanation(test_results, feature_name, target)
        model_implications = generate_model_implications(feature_name, "mutual_information", test_results)
        return {
            "visualization": None,
            "explanation": explanation,
            "hypothesis": hypothesis,
            "significance": significance,
            "model_implications": model_implications
        }
    except Exception as e:
        logger.error(f"Error generating mutual information analysis for {feature_name}: {str(e)}", exc_info=True)
        raise

def generate_t_test_analysis(feature_name: str,
                             test_results: Dict[str, Any],
                             df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for t-test results.
    """
    try:
        fig = create_box_plot(df, feature_name, "high_volatility")
        explanation, hypothesis, significance = generate_t_test_explanation(test_results, feature_name)
        model_implications = generate_model_implications(feature_name, "t_test", test_results)
        return {
            "visualization": fig,
            "explanation": explanation,
            "hypothesis": hypothesis,
            "significance": significance,
            "model_implications": model_implications
        }
    except Exception as e:
        logger.error(f"Error generating t-test analysis for {feature_name}: {str(e)}", exc_info=True)
        raise

def create_time_series_plot(df: pd.DataFrame, feature_name: str, title: str = None) -> plt.Figure:
    """Create a time series plot for a feature with consistent formatting."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(df.index, df[feature_name])
    ax.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(feature_name.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title or f"{feature_name.replace('_', ' ').title()} Over Time", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    return fig

def create_histogram_plot(series: pd.Series, feature_name: str, bins: int = 30, kde: bool = True) -> plt.Figure:
    """Create a histogram with optional KDE for a feature with consistent formatting."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.histplot(series, bins=bins, kde=kde, ax=ax)
    ax.set_xlabel(feature_name.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"Distribution of {feature_name.replace('_', ' ').title()}", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    return fig

def create_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str, title: str = None) -> plt.Figure:
    """Create a scatter plot with regression line and consistent formatting."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.5})
    ax.set_xlabel(x_label.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title or f"{x_label.replace('_', ' ').title()} vs {y_label.replace('_', ' ').title()}", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    return fig

def create_box_plot(data: pd.DataFrame, feature: str, groupby: str, title: str = None) -> plt.Figure:
    """Create a box plot comparing a feature across groups with consistent formatting."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.boxplot(x=groupby, y=feature, data=data, ax=ax)
    ax.set_xlabel(groupby.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(feature.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title or f"{feature.replace('_', ' ').title()} by {groupby.replace('_', ' ').title()}", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.tight_layout()
    return fig

def create_rolling_statistics_plot(series: pd.Series, window: int = 20, title: str = None) -> plt.Figure:
    """Create a plot with original series and rolling mean/std with consistent formatting."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(series, label="Original")
    ax.plot(rolling_mean, label=f"Rolling Mean (window={window})")
    ax.plot(rolling_std, label=f"Rolling Std (window={window})")
    ax.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(series.name.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title or f"Rolling Statistics of {series.name.replace('_', ' ').title()}", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    return fig

def generate_correlation_explanation(test_results: Dict[str, Any], feature_name: str,
                                     target_name: str = "future_price_change_4h") -> Tuple[str, str, str]:
    """Generate explanation text for correlation results."""
    pearson_r = test_results.get("pearson_r", 0)
    pearson_p = test_results.get("pearson_p", 1)
    spearman_r = test_results.get("spearman_r", 0)
    spearman_p = test_results.get("spearman_p", 1)
    
    explanation = (f"The Pearson correlation between {feature_name} and {target_name} is {pearson_r:.2f} "
                   f"(p-value: {format_p_value(pearson_p)}). This indicates a {format_correlation_strength(pearson_r)} "
                   f"linear relationship. The Spearman rank correlation is {spearman_r:.2f} "
                   f"(p-value: {format_p_value(spearman_p)}), indicating a {format_correlation_strength(spearman_r)} "
                   f"monotonic relationship.")
    
    hypothesis = f"H0: No linear correlation between {feature_name} and {target_name}."
    significance = interpret_significance(min(pearson_p, spearman_p))
    return explanation, hypothesis, significance

def generate_distribution_explanation(test_results: Dict[str, Any], feature_name: str) -> Tuple[str, str, str]:
    """Generate explanation text for distribution results."""
    results = test_results.get("results", {})
    mean = results.get("mean", 0)
    std = results.get("std", 0)
    skew = results.get("skew", 0)
    kurt = results.get("kurtosis", 0)
    
    explanation = (f"The distribution of {feature_name} has a mean of {mean:.2f}, std of {std:.2f}, "
                   f"skewness of {skew:.2f}, and kurtosis of {kurt:.2f}. ")
    
    if abs(skew) < 0.5:
        explanation += "The distribution is approximately symmetric. "
    elif skew > 0:
        explanation += "The distribution is right-skewed (positive skew), with a longer tail to the right. "
    else:
        explanation += "The distribution is left-skewed (negative skew), with a longer tail to the left. "
    
    if kurt > 3:
        explanation += "It has heavier tails than a normal distribution (leptokurtic), suggesting more outliers."
    elif kurt < 3:
        explanation += "It has lighter tails than a normal distribution (platykurtic), suggesting fewer outliers."
    else:
        explanation += "It has tail weight similar to a normal distribution (mesokurtic)."
    
    hypothesis = "H0: The feature follows a normal distribution."
    significance = "Descriptive statistics provided; no significance test applied."
    return explanation, hypothesis, significance

def generate_stationarity_explanation(test_results: Dict[str, Any], feature_name: str) -> Tuple[str, str, str]:
    """Generate explanation text for stationarity results."""
    results = test_results.get("results", {})
    adf_statistic = results.get("adf_statistic", 0)
    p_value = results.get("p_value", 1)
    is_stationary = results.get("is_stationary", False)
    
    explanation = (f"The ADF test for {feature_name} yields a test statistic of {adf_statistic:.2f} "
                   f"and a p-value of {format_p_value(p_value)}. ")
    
    if is_stationary or p_value < 0.05:
        explanation += ("The feature is stationary, meaning its statistical properties do not change over time. "
                        "This suggests it can be used directly in the model without differencing or transformation.")
    else:
        explanation += ("The feature is non-stationary, meaning its statistical properties change over time. "
                        "This may require differencing or transformation before use in the model.")
    
    hypothesis = f"H0: {feature_name} is non-stationary (has a unit root)."
    significance = interpret_significance(p_value)
    return explanation, hypothesis, significance

def generate_mutual_information_explanation(test_results: Dict[str, Any], feature_name: str,
                                            target_name: str) -> Tuple[str, str, str]:
    """Generate explanation text for mutual information results."""
    results = test_results.get("results", {})
    mi = results.get("mutual_information", 0)
    
    explanation = (f"The mutual information between {feature_name} and {target_name} is {mi:.4f}. "
                   f"This measures the amount of information obtained about one variable by observing the other. ")
    
    if mi < 0.01:
        explanation += "The very low score suggests minimal dependency between these variables."
    elif mi < 0.1:
        explanation += "The low score suggests weak dependency between these variables."
    elif mi < 0.5:
        explanation += "The moderate score suggests meaningful dependency between these variables."
    else:
        explanation += "The high score suggests strong dependency between these variables."
    
    hypothesis = f"H0: No dependency between {feature_name} and {target_name}."
    significance = "Mutual information is a measure; no p-value provided."
    return explanation, hypothesis, significance

def generate_t_test_explanation(test_results: Dict[str, Any], feature_name: str,
                                condition_name: str = "high_volatility") -> Tuple[str, str, str]:
    """Generate explanation text for t-test results."""
    results = test_results.get("results", {})
    t_stat = results.get("t_statistic", 0)
    p_value = results.get("p_value", 1)
    significant = results.get("significant", False)
    
    explanation = (f"The t-test comparing {feature_name} under different {condition_name} conditions "
                   f"yields a t-statistic of {t_stat:.2f} with a p-value of {format_p_value(p_value)}. ")
    
    if significant or p_value < 0.05:
        explanation += (f"There is a statistically significant difference in {feature_name} "
                        f"between different {condition_name} conditions.")
    else:
        explanation += (f"There is no statistically significant difference in {feature_name} "
                        f"between different {condition_name} conditions.")
    
    hypothesis = f"H0: No difference in {feature_name} means across {condition_name} conditions."
    significance = interpret_significance(p_value)
    return explanation, hypothesis, significance

def generate_model_implications(feature_name: str, test_name: str, test_results: Dict[str, Any]) -> str:
    """Generate explanations of how this feature/test relates to the XGBoost model."""
    dimensions = get_feature_dimension_mapping().get(feature_name, ["Unknown"])
    implications = f"This feature influences the {', '.join(dimensions)} dimensions in the market regime classification. "
    
    if test_name == "correlation":
        results = test_results.get("results", {})
        pearson_r = results.get("pearson_r", 0) if isinstance(results, dict) else test_results.get("pearson_r", 0)
        p_value = results.get("pearson_p", 1) if isinstance(results, dict) else test_results.get("pearson_p", 1)
        if abs(pearson_r) > 0.3:
            implications += "The strong correlation suggests high predictive power for these dimensions."
        elif abs(pearson_r) > 0.1:
            implications += "The moderate correlation suggests useful predictive power for these dimensions."
        else:
            implications += "The weak correlation suggests limited linear predictive power for these dimensions."
        if p_value < 0.05:
            implications += " The statistical significance supports its inclusion in the model."
        else:
            implications += " The lack of statistical significance suggests caution when using this feature."
    elif test_name == "stationarity":
        results = test_results.get("results", {})
        is_stationary = results.get("is_stationary", False) if isinstance(results, dict) else test_results.get("is_stationary", False)
        if is_stationary:
            implications += "Its stationarity means it can be used directly in the model without transformation."
        else:
            implications += "Its non-stationarity suggests it may need differencing or transformation before use."
    elif test_name == "mutual_information":
        results = test_results.get("results", {})
        mi = results.get("mutual_information", 0) if isinstance(results, dict) else test_results.get("mutual_information", 0)
        if mi > 0.5:
            implications += f"The high mutual information score of {mi:.2f} indicates strong non-linear predictive power."
        elif mi > 0.1:
            implications += f"The moderate mutual information score of {mi:.2f} indicates useful non-linear relationships."
        else:
            implications += f"The low mutual information score of {mi:.2f} suggests limited non-linear predictive power."
    elif test_name == "t_test":
        results = test_results.get("results", {})
        p_value = results.get("p_value", 1) if isinstance(results, dict) else test_results.get("p_value", 1)
        if p_value < 0.05:
            implications += "The significant differences across market conditions enhance its utility for regime classification."
        else:
            implications += "The lack of significant differences across market conditions may limit its usefulness."
    else:  # distribution analysis
        results = test_results.get("results", {})
        skew = results.get("skew", 0) if isinstance(results, dict) else test_results.get("skew", 0)
        if abs(skew) > 1:
            implications += "Its skewed distribution suggests potential need for transformation before model training."
        else:
            implications += "Its distribution characteristics are suitable for direct use in the model."
    
    return implications

def get_feature_dimension_mapping() -> Dict[str, List[str]]:
    """Returns a mapping of features to the 5D dimensions they influence."""
    return {
        "directional_momentum": ["Direction", "Momentum"],
        "volatility_signature": ["Volatility"],
        "volume_intensity": ["Participation"],
        "body_proportion": ["Momentum", "Direction"],
        "range_position": ["Momentum"]
    }

def format_p_value(p_value: float) -> str:
    """Format p-value with appropriate scientific notation."""
    if p_value < 0.001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.3f}"

def interpret_significance(p_value: float, threshold: float = 0.05) -> str:
    """Interpret the statistical significance of a p-value."""
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < threshold:
        return f"Significant (p < {threshold})"
    else:
        return f"Not significant (p ≥ {threshold})"

def format_correlation_strength(correlation: float) -> str:
    """Interpret the strength of a correlation coefficient."""
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        return "strong"
    elif abs_corr > 0.3:
        return "moderate"
    elif abs_corr > 0.1:
        return "weak"
    else:
        return "negligible"

def handle_analysis_error(feature_name: str, test_name: str, error: Exception) -> Dict[str, Any]:
    """Handle errors in analysis process and return fallback content with consistent formatting."""
    logger.error(f"Error analyzing {test_name} for {feature_name}: {str(error)}", exc_info=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.text(0.5, 0.5, f"Analysis error: {str(error)}", ha='center', va='center', 
            fontsize=ERROR_TEXT_FONTSIZE, color=ERROR_TEXT_COLOR)
    ax.set_axis_off()
    plt.tight_layout()
    return {
        "visualization": fig,
        "explanation": f"Error during analysis: {str(error)}",
        "hypothesis": "Analysis could not be completed",
        "significance": "Unknown due to error",
        "model_implications": "Cannot determine implications due to error"
    }

def analyze_and_visualize(stats_results, df, start_date=None, end_date=None):
    """
    Alias for analyze_four_hour_data to maintain backward compatibility.
    Automatically uses current date range if dates not provided.
    """
    if start_date is None or end_date is None:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        logger.warning(f"No date range provided, using default: {start_date} to {end_date}")
    
    logger.info(f"analyze_and_visualize called with dates: {start_date} to {end_date}")
    return analyze_four_hour_data(stats_results, df, start_date, end_date)

def test_analyze_four_hour_data():
    """Test the analyze_four_hour_data function with sample data."""
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    df = load_four_hour_features(start_date, end_date)
    if df.empty:
        print("No data available for the selected date range.")
        return

    stats_results = generate_four_hour_stats(start_date, end_date)
    analysis_results = analyze_four_hour_data(stats_results, df, start_date, end_date)

    for feature, tests in analysis_results.items():
        print(f"\nFeature: {feature}")
        for test_type, test_analysis in tests.items():
            if test_type == "summary":
                continue
            print(f"  Test: {test_type}")
            print(f"    Explanation: {test_analysis['explanation']}")
            print(f"    Hypothesis: {test_analysis['hypothesis']}")
            print(f"    Significance: {test_analysis['significance']}")
            print(f"    Model Implications: {test_analysis['model_implications']}")
            if test_analysis['visualization']:
                print("    Visualization available")

if __name__ == "__main__":
    test_analyze_four_hour_data()