import streamlit as st

class StateManager:
    def __init__(self):
        """Initialize state manager with Streamlit session state."""
        if 'date_range' not in st.session_state:
            st.session_state.date_range = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

    def set_date_range(self, start_date: str, end_date: str) -> None:
        """Set the date range in session state."""
        try:
            st.session_state.date_range = (start_date, end_date)
            st.session_state.data_loaded = False  # Reset data loaded flag
        except Exception as e:
            st.error(f"Error setting date range: {e}")

    def get_date_range(self) -> tuple:
        """Get the current date range from session state."""
        return st.session_state.date_range if st.session_state.date_range else (None, None)

state = StateManager()