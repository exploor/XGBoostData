import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Points to 'mrdroot'
sys.path.insert(0, str(project_root))
st.set_page_config(layout="wide")

# Absolute imports for 4-hour, daily, and weekly analysis
from mrd.data_processing.scripts.compute_4h_features import compute_4h_features, save_features_to_db as save_4h_features_to_db
from mrd.data_processing.scripts.compute_daily_features import compute_daily_features, save_features_to_db as save_daily_features_to_db
from mrd.statistical_analysis.summary.four_hour_stats import generate_four_hour_stats
from mrd.statistical_analysis.summary.daily_stats import generate_daily_stats
from mrd.statistical_analysis.summary.four_hour_analysis import analyze_four_hour_data
from mrd.statistical_analysis.summary.daily_analysis import analyze_daily_data
from mrd.statistical_analysis.summary.weekly_stats import generate_weekly_stats  # Added for weekly tab
from mrd.statistical_analysis.summary.weekly_analysis import analyze_weekly_data  # Added for weekly tab

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_data
def fetch_daily_data(start_date_str, end_date_str):
    """Fetch daily data from the database."""
    try:
        user = "postgres"
        password = "postgres"
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        query = f"SELECT * FROM daily_vectors WHERE date BETWEEN '{start_date_str}' AND '{end_date_str}' LIMIT 1000"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching daily data: {e}")
        st.error("Failed to fetch daily data from database.")
        return pd.DataFrame()

@st.cache_data
def fetch_four_hour_data(start_date_str, end_date_str):
    """Fetch 4-hour data from the database."""
    try:
        user = "postgres"
        password = "postgres"
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        query = f"SELECT * FROM four_hour_vectors WHERE timestamp BETWEEN '{start_date_str}' AND '{end_date_str}' LIMIT 1000"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching 4-hour data: {e}")
        st.error("Failed to fetch 4-hour data from database.")
        return pd.DataFrame()

@st.cache_data
def fetch_weekly_data(start_date_str, end_date_str):
    """Fetch weekly data from the database."""
    try:
        user = "postgres"
        password = "postgres"
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        query = f"SELECT * FROM weekly_vectors WHERE timestamp BETWEEN '{start_date_str}' AND '{end_date_str}' LIMIT 1000"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching weekly data: {e}")
        st.error("Failed to fetch weekly data from database.")
        return pd.DataFrame()

def display_daily_tab(start_date_str, end_date_str):
    """Display the daily analysis tab with data and statistical results."""
    st.header("Daily Statistical Analysis")
    df = fetch_daily_data(start_date_str, end_date_str)
    if not df.empty:
        sort_column = st.selectbox("Sort by column", df.columns, key='daily_sort_column')
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], key='daily_sort_order')
        ascending = True if sort_order == "Ascending" else False
        sorted_df = df.sort_values(by=sort_column, ascending=ascending)
        st.write(f"Top 10 rows sorted by {sort_column} in {sort_order} order:")
        st.dataframe(sorted_df.head(10))
    else:
        st.write("No daily data available for the selected date range.")
    
    if st.session_state.analysis_run and 'daily' in st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results['daily']
        if isinstance(analysis_results, dict):
            for feature, feature_analysis in analysis_results.items():
                if isinstance(feature_analysis, dict):
                    st.subheader(feature.replace('_', ' ').title())
                    if "summary" in feature_analysis:
                        st.write(feature_analysis["summary"].get("explanation", "No summary available."))
                    for test, test_analysis in feature_analysis.items():
                        if test == "summary":
                            continue
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if "visualization" in test_analysis and test_analysis["visualization"]:
                                try:
                                    fig = test_analysis["visualization"]
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error displaying visualization: {str(e)}")
                                    logger.error(f"Visualization error: {str(e)}")
                        with col2:
                            st.write(f"**{test.replace('_', ' ').title()}**")
                            st.write(f"**Explanation**: {test_analysis.get('explanation', 'N/A')}")
                            st.write(f"**Hypothesis**: {test_analysis.get('hypothesis', 'N/A')}")
                            st.write(f"**Significance**: {test_analysis.get('significance', 'N/A')}")
                            st.write(f"**Model Implications**: {test_analysis.get('model_implications', 'N/A')}")
                else:
                    st.warning(f"Invalid analysis format for feature: {feature}")
        else:
            st.warning("Daily analysis results are not in the expected format.")
    else:
        st.write("Please select a date range and click 'Run Analysis' in the Home tab to see results.")

def display_four_hour_tab(start_date_str, end_date_str):
    """Display the 4-hour analysis tab with data and statistical results."""
    st.header("4-Hour Statistical Analysis")
    df = fetch_four_hour_data(start_date_str, end_date_str)
    if not df.empty:
        sort_column = st.selectbox("Sort by column", df.columns, key='four_hour_sort_column')
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], key='four_hour_sort_order')
        ascending = True if sort_order == "Ascending" else False
        sorted_df = df.sort_values(by=sort_column, ascending=ascending)
        st.write(f"Top 10 rows sorted by {sort_column} in {sort_order} order:")
        st.dataframe(sorted_df.head(10))
    else:
        st.write("No data available for the selected date range.")
    
    if st.session_state.analysis_run and '4h' in st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results['4h']
        if isinstance(analysis_results, dict):
            for feature, feature_analysis in analysis_results.items():
                if isinstance(feature_analysis, dict):
                    st.subheader(feature.replace('_', ' ').title())
                    if "summary" in feature_analysis:
                        st.write(feature_analysis["summary"].get("explanation", "No summary available."))
                    for test, test_analysis in feature_analysis.items():
                        if test == "summary":
                            continue
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if "visualization" in test_analysis and test_analysis["visualization"]:
                                try:
                                    fig = test_analysis["visualization"]
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error displaying visualization: {str(e)}")
                                    logger.error(f"Visualization error: {str(e)}")
                        with col2:
                            st.write(f"**{test.replace('_', ' ').title()}**")
                            if "key_metric" in test_analysis and test_analysis["key_metric"] != "N/A":
                                st.write(f"**Key Metric**: {test_analysis['key_metric']}")
                            st.write(f"**Explanation**: {test_analysis.get('explanation', 'N/A')}")
                            st.write(f"**Hypothesis**: {test_analysis.get('hypothesis', 'N/A')}")
                            st.write(f"**Significance**: {test_analysis.get('significance', 'N/A')}")
                            st.write(f"**Model Implications**: {test_analysis.get('model_implications', 'N/A')}")
                else:
                    st.warning(f"Invalid analysis format for feature: {feature}")
        else:
            st.warning("4-hour analysis results are not in the expected format.")
    else:
        st.write("Please select a date range and click 'Run Analysis' in the Home tab to see results.")

def display_weekly_tab(start_date_str, end_date_str):
    """Display the weekly analysis tab with data and statistical results."""
    st.header("Weekly Statistical Analysis")
    df = fetch_weekly_data(start_date_str, end_date_str)
    if not df.empty:
        sort_column = st.selectbox("Sort by column", df.columns, key='weekly_sort_column')
        sort_order = st.radio("Sort order", ["Ascending", "Descending"], key='weekly_sort_order')
        ascending = True if sort_order == "Ascending" else False
        sorted_df = df.sort_values(by=sort_column, ascending=ascending)
        st.write(f"Top 10 rows sorted by {sort_column} in {sort_order} order:")
        st.dataframe(sorted_df.head(10))
    else:
        st.write("No weekly data available for the selected date range.")
    
    if st.session_state.analysis_run and 'weekly' in st.session_state.analysis_results:
        analysis_results = st.session_state.analysis_results['weekly']
        if isinstance(analysis_results, dict):
            for feature, feature_analysis in analysis_results.items():
                if isinstance(feature_analysis, dict):
                    st.subheader(feature.replace('_', ' ').title())
                    if "summary" in feature_analysis:
                        st.write(feature_analysis["summary"].get("explanation", "No summary available."))
                    for test, test_analysis in feature_analysis.items():
                        if test == "summary":
                            continue
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if "visualization" in test_analysis and test_analysis["visualization"]:
                                try:
                                    fig = test_analysis["visualization"]
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error displaying visualization: {str(e)}")
                                    logger.error(f"Visualization error: {str(e)}")
                        with col2:
                            st.write(f"**{test.replace('_', ' ').title()}**")
                            st.write(f"**Explanation**: {test_analysis.get('explanation', 'N/A')}")
                            st.write(f"**Hypothesis**: {test_analysis.get('hypothesis', 'N/A')}")
                            st.write(f"**Significance**: {test_analysis.get('significance', 'N/A')}")
                            st.write(f"**Model Implications**: {test_analysis.get('model_implications', 'N/A')}")
                else:
                    st.warning(f"Invalid analysis format for feature: {feature}")
        else:
            st.warning("Weekly analysis results are not in the expected format.")
    else:
        st.write("Please select a date range and click 'Run Analysis' in the Home tab to see results.")

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

    st.title("Statistical Analysis Dashboard")

    # Create tabs: Home, Daily Analysis, 4-Hour Analysis, Weekly Analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Home", "Daily Analysis", "4-Hour Analysis", "Weekly Analysis"])

    # Home Tab: Date selection and analysis trigger
    with tab1:
        st.header("Home")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        if st.button("Run Analysis"):
            with st.spinner("Computing features and running analysis..."):
                try:
                    # Assume start_date_str and end_date_str come from date inputs
                    start_date_str = st.session_state.get('start_date', '2022-03-01')
                    end_date_str = st.session_state.get('end_date', '2022-04-30')

                    # Initialize analysis_results if not already done
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = {}

                    # Existing 4-hour Analysis (example, adjust as per your code)
                    start_date_4h = f"{start_date_str} 00:00:00"
                    end_date_4h = f"{end_date_str} 23:59:59"
                    df_4h = compute_4h_features(start_date_4h, end_date_4h)  # Your function
                    if not df_4h.empty:
                        save_4h_features_to_db(df_4h)  # Your function
                        stats_4h = generate_four_hour_stats(start_date_str, end_date_str)
                        analysis_4h = analyze_four_hour_data(stats_4h, df_4h, start_date_str, end_date_str)
                        st.session_state.analysis_results['4h'] = analysis_4h

                    # Existing Daily Analysis (example, adjust as per your code)
                    df_daily = compute_daily_features(start_date_str, end_date_str)  # Your function
                    if not df_daily.empty:
                        save_daily_features_to_db(df_daily)  # Your function
                        stats_daily = generate_daily_stats(start_date_str, end_date_str)
                        analysis_daily = analyze_daily_data(stats_daily, df_daily, start_date_str, end_date_str)
                        st.session_state.analysis_results['daily'] = analysis_daily

                    # Add Weekly Analysis
                    df_weekly = fetch_weekly_data(start_date_str, end_date_str)
                    if not df_weekly.empty:
                        stats_weekly = generate_weekly_stats(start_date_str, end_date_str)
                        analysis_weekly = analyze_weekly_data(stats_weekly, df_weekly, start_date_str, end_date_str)
                        st.session_state.analysis_results['weekly'] = analysis_weekly
                    else:
                        st.warning("No weekly data available for the selected date range.")

                    # Mark analysis as run
                    st.session_state.analysis_run = True
                    st.success("Analysis completed. Check the tabs for results.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error during analysis: {str(e)}", exc_info=True)
    # Daily Analysis Tab
    with tab2:
        display_daily_tab(start_date_str, end_date_str)

    # 4-Hour Analysis Tab
    with tab3:
        display_four_hour_tab(start_date_str, end_date_str)

    # Weekly Analysis Tab
    with tab4:
        display_weekly_tab(start_date_str, end_date_str)