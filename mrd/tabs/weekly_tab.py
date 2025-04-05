import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def fetch_weekly_data(start_date_str, end_date_str):
    """Fetch weekly data from the database."""
    try:
        # Adjust connection details as per your setup
        user = "postgres"
        password = "postgres"
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        query = f"""
            SELECT * FROM weekly_vectors 
            WHERE timestamp BETWEEN '{start_date_str}' AND '{end_date_str}' 
            LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching weekly data: {e}")
        st.error("Failed to fetch weekly data from database.")
        return pd.DataFrame()

def display_weekly_tab(start_date_str, end_date_str):
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
        st.write("Please select a date range and click 'Run Analysis' to see results.")