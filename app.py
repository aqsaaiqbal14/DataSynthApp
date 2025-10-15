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
    font-size: 3em;
    font-weight: bold;
    color: #4CAF50;
    margin-bottom: 10px;
}
.page-intro {
    font-size: 1.5em;
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
                np.random.choice(unique_values, num_rows)
                if len(unique_values) > 0
                else [f"Synthetic_{i}" for i in range(num_rows)]
            )

    final_df = pd.DataFrame(synthetic_data)
    return final_df[original_df.columns] if all(c in final_df.columns for c in original_df.columns) else final_df


def generate_ctgan_data(df, num_rows, epochs=100):
    try:
        from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
        from ydata_synthetic.synthesizers.regular import RegularSynthesizer

        # --- Identify numeric and categorical columns ---
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in df.columns if c not in num_cols]

        if not num_cols and not cat_cols:
            st.error("No valid columns found for CTGAN training. Falling back to Faker demo.")
            return generate_faker_data(df, num_rows)

        # --- Define model parameters ---
        model_args = ModelParameters(
            batch_size=500,
            lr=2e-4,
            betas=(0.5, 0.9),
            noise_dim=128,
            layers_dim=128
        )
        train_args = TrainParameters(epochs=epochs)

        # --- Train CTGAN ---
        synth = RegularSynthesizer(modelname='ctgan', model_parameters=model_args)
        synth.fit(data=df, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

        # --- Sample synthetic data ---
        synth_data = synth.sample(num_rows)

        return synth_data

    except Exception as e:
        st.error(f"CTGAN training failed: {e}. Falling back to Faker demo.")
        return generate_faker_data(df, num_rows)


# --- MAIN CONTENT ---
if page == "🏠 DataForge Home":
    st.markdown('<div class="main-title">Welcome to DataForge</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="main-title">DataForge</div>
<p class="page-intro">
DataForge is a next-generation synthetic data generator and analytics platform built for privacy-first innovation.  
Whether you are a data scientist, developer, or researcher, DataForge empowers you to:
</p>

<ul style="font-size:1.2em; line-height:1.6;">
    <li>⚙️ <b>Generate high-quality synthetic datasets</b> using rule-based (Faker) or AI-powered (CTGAN) engines.</li>
    <li>📤 <b>Upload real datasets securely</b> across CSV, Excel, or JSON formats.</li>
    <li>📊 <b>Visually compare real vs synthetic data</b> through histograms and correlation heatmaps.</li>
    <li>⬇️ <b>Export anonymized synthetic data</b> for testing, modeling, or collaboration — risk-free.</li>
</ul>

<p class="page-intro">
Ready to begin? Head over to <b>⬆️ Upload Data</b> in the sidebar to get started!
</p>
""", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/1843/1843544.png", width=120)
    st.success("Start by navigating to **⬆️ Upload Data** in the sidebar to begin!")
elif page == "⬆️ Upload Data":
    st.markdown('<div class="main-title">⬆️ Upload Your Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Upload your dataset in CSV, Excel, or JSON format. DataForge will securely load and prepare it for synthetic generation.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"], key=f"uploader_{st.session_state.data_key}")

    if uploaded_file:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.synthetic_data = None
                st.session_state.data_key += 1
                st.success("File uploaded successfully!")

        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            st.subheader("Original Data Preview")
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
    else:
        st.info("Please upload a dataset to start.")

elif page == "🧪 Synthesization":
    st.markdown('<div class="main-title">🧪 Generate Synthetic Data</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Choose your preferred method to generate synthetic data — either the fast Faker Demo or CTGAN. Define the number of rows to generate below.</p>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first.")
    else:
        df_original = st.session_state.uploaded_data
        method = st.selectbox("Select Synthesization Method", ["Faker Demo", "CTGAN"])
        num_rows = st.slider("Rows to generate", 100, len(df_original) * 5, len(df_original), 100)

        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                if "Faker" in method:
                    synth_df = generate_faker_data(df_original, num_rows)
                else:
                    synth_df = generate_ctgan_data(df_original, num_rows)

                st.session_state.synthetic_data = synth_df
                st.success("Synthetic data generated successfully!")
                st.dataframe(synth_df.head())
                st.download_button("Download Synthetic CSV", synth_df.to_csv(index=False), "synthetic_data.csv", "text/csv")

elif page == "📊 Analysis":
    st.markdown('<div class="main-title">📊 Data Analysis & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Compare the statistical properties of your original and synthetic datasets using interactive histograms and correlation heatmaps.</p>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None or st.session_state.synthetic_data is None:
        st.warning("Please upload and generate data first.")
    else:
        df_orig, df_synth = st.session_state.uploaded_data, st.session_state.synthetic_data
        common_numeric = list(set(df_orig.select_dtypes(np.number).columns) & set(df_synth.select_dtypes(np.number).columns))
        if common_numeric:
            col = st.selectbox("Select numeric column to compare:", common_numeric)
            df_plot = pd.DataFrame({
                col: pd.concat([df_orig[col].dropna(), df_synth[col].dropna()]),
                "Type": ["Original"] * len(df_orig[col].dropna()) + ["Synthetic"] * len(df_synth[col].dropna())
            })
            fig = px.histogram(df_plot, x=col, color="Type", barmode="overlay", histnorm="density", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

            corr_orig = df_orig[common_numeric].corr()
            corr_synth = df_synth[common_numeric].corr()
            st.subheader("Correlation Heatmaps")
            c1, c2 = st.columns(2)
            c1.plotly_chart(go.Figure(data=go.Heatmap(z=corr_orig, x=corr_orig.columns, y=corr_orig.index, colorscale="Blues")), use_container_width=True)
            c2.plotly_chart(go.Figure(data=go.Heatmap(z=corr_synth, x=corr_synth.columns, y=corr_synth.index, colorscale="Reds")), use_container_width=True)
        else:
            st.info("No common numeric columns found.")

elif page == "📘 User Guide":
    st.markdown('<div class="main-title">📘 User Guide</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Follow this step-by-step guide to use DataForge effectively:</p>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload** your dataset (.csv, .xlsx, or .json).  
    2. **Generate** synthetic data using **Faker** or **CTGAN**.  
    3. **Analyze** the similarity between real and synthetic data.  
    4. **Download** your synthetic dataset for future use.
    """)

elif page == "📩 Contact":
    st.markdown('<div class="main-title">📩 Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">We’d love to hear from you! Contact our development team for support or collaboration.</p>', unsafe_allow_html=True)
    st.markdown("""
    **Developers:**  
    - Aqsa Iqbal  
    - Faiqa Rizwan  
    - Nimra Khan  
    - Mahnoor Siddiqui  
    - Laiba Sikander  

    **Email:** support@datasynthapp.com
    """)

st.sidebar.markdown("---")
st.sidebar.info("Powered by DataForge — Synthetic Data Generation Platform")

