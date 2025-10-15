
# -*- coding: utf-8 -*-
# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DataForge: Synthetic Data Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLES ---
st.markdown("""
<style>
.main-title {
    font-size: 2.8em;
    font-weight: bold;
    color: #4CAF50;
    margin-bottom: 10px;
}
.page-intro {
    font-size: 1.1em;
    color: #333333;
    margin-bottom: 25px;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
for key, default in {
    "uploaded_data": None,
    "synthetic_data": None,
    "data_key": 0,
    "uploaded_file_name": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 DataForge Home",
    "⬆️ Upload Data",
    "🧪 Synthesization",
    "📊 Analysis",
    "📘 User Guide",
    "📩 Contact"
])

# --- FUNCTIONS ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type. Please use CSV, Excel (.xlsx), or JSON.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def generate_faker_data(original_df, num_rows):
    fake = Faker()
    synthetic_data = {}

    for col in original_df.columns:
        dtype = original_df[col].dtype
        if np.issubdtype(dtype, np.number):
            min_val, max_val = original_df[col].min(), original_df[col].max()
            low, high = min_val * 0.9 if pd.notna(min_val) else 0, max_val * 1.1 if pd.notna(max_val) else 100
            synthetic_data[col] = (
                np.random.randint(int(low), int(high) + 1, num_rows)
                if np.issubdtype(dtype, np.integer)
                else np.random.uniform(low, high, num_rows)
            )
        elif col.lower() in ["name", "full_name"]:
            synthetic_data[col] = [fake.name() for _ in range(num_rows)]
        elif col.lower() in ["email", "email_address"]:
            synthetic_data[col] = [fake.email() for _ in range(num_rows)]
        elif col.lower() in ["job", "occupation"]:
            synthetic_data[col] = [fake.job() for _ in range(num_rows)]
        elif col.lower() in ["id", "user_id", "index"]:
            synthetic_data[col] = range(1, num_rows + 1)
        else:
            unique_values = original_df[col].dropna().unique()
            synthetic_data[col] = (
                np.ra

