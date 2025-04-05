import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Function to fetch data from the database, cached for performance
@st.cache_data
def fetch_data(start_date_str, end_date_str):
    try:
        # Hardcoded database credentials (WARNING: Not secure for production)
        user = "postgres"  # Replace with your actual username
        password = "postgres"  # Replace with your actual password
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        # Create connection string
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        # Query to fetch data within the date range, limited to 1000 rows
        query = f"SELECT * FROM four_hour_vectors WHERE timestamp BETWEEN '{start_date_str}' AND '{end_date_str}' LIMIT 1000"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching data from database: {e}")
        st.error("Failed to fetch data from database. Please check the logs.")
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to display the 4-hour tab content
def display_four_hour_tab(start_date_str, end_date_str):
    st.header("4-Hour Statistical Analysis")
    
    # Fetch data from the database
    df = fetch_data(start_date_str, end_date_str)
    
    # Display database view at the top
    if not df.empty:
        # User selects the column to sort by
        sort_column = st.selectbox("Sort by column", df.columns)
        
        # User selects the sort order
        sort_order = st.radio("Sort order", ["Ascending", "Descending"])
        
        # Sort the DataFrame based on user input
        ascending = True if sort_order == "Ascending" else False
        sorted_df = df.sort_values(by=sort_column, ascending=ascending)
        
        # Display the top 10 rows
        st.write(f"Top 10 rows sorted by {sort_column} in {sort_order} order:")
        st.dataframe(sorted_df.head(10))
    else:
        st.write("No data available for the selected date range.")
    
    # Existing analysis results (unchanged)
    if st.session_state.analysis_run and st.session_state.analysis_results:
        for feature, feature_analysis in st.session_state.analysis_results.items():
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
                            logger.error(f"Visualization error for {feature} - {test}: {str(e)}")
                with col2:
                    st.write(f"**{test.replace('_', ' ').title()}**")
                    if "key_metric" in test_analysis and test_analysis["key_metric"] != "N/A":
                        st.write(f"**Key Metric**: {test_analysis['key_metric']}")
                    st.write(f"**Explanation**: {test_analysis.get('explanation', 'No explanation available.')}")
                    st.write(f"**Hypothesis**: {test_analysis.get('hypothesis', 'No hypothesis available.')}")
                    st.write(f"**Significance**: {test_analysis.get('significance', 'No significance available.')}")
                    st.write(f"**Model Implications**: {test_analysis.get('model_implications', 'No implications available.')}")
    else:
        st.write("Please select a date range and click 'Run Analysis' to see results.")