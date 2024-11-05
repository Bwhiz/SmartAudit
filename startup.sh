#!/bin/bash

# Set Streamlit server settings
export STREAMLIT_PORT=${PORT:-8501}
export STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
streamlit run main.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0