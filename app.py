# -*- coding: utf-8 -*-
# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from scipy.stats import ks_2samp
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
    "⚙️ Pre-processing",
    "🧪 Synthesization",
    "📊 Analysis",
    "✨ Post-processing",
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

def preprocess_data(df, missing_threshold_pct):
    df_clean = df.copy()

    # 1. Drop columns with too many missing values
    missing_percent = df_clean.isna().mean() * 100
    cols_to_drop = missing_percent[missing_percent > missing_threshold_pct].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # 2. Impute remaining missing values
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            dtype = df_clean[col].dtype
            if np.issubdtype(dtype, np.number):
                # Fill numeric NaNs with the mean (if possible)
                if df_clean[col].dropna().shape[0] > 0:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                else:
                    df_clean[col].fillna(0, inplace=True)
            else:
                # Fill categorical NaNs with the mode (most frequent value)
                if not df_clean[col].mode().empty:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                else:
                    df_clean[col].fillna("Missing", inplace=True)

    return df_clean, cols_to_drop

# Faker Generation (uses processed data)
def generate_faker_data(original_df, num_rows):
    fake = Faker()
    synthetic_data = {}

    for col in original_df.columns:
        dtype = original_df[col].dtype
        col_lower = col.lower()

        # Numeric columns
        if np.issubdtype(dtype, np.number):
            series = original_df[col].dropna()
            if series.shape[0] == 0:
                # No information — generate zeros
                if np.issubdtype(dtype, np.integer):
                    synthetic_data[col] = [0] * num_rows
                else:
                    synthetic_data[col] = [0.0] * num_rows
                continue

            min_val, max_val = series.min(), series.max()
            # Provide sensible low/high bounds even when min==max
            if pd.isna(min_val) or pd.isna(max_val):
                low, high = 0, 100
            else:
                if min_val == max_val:
                    # expand a small range around the single value
                    if np.issubdtype(dtype, np.integer):
                        low = int(min_val) - 5
                        high = int(max_val) + 5
                    else:
                        low = float(min_val) - abs(float(min_val)) * 0.1 - 1
                        high = float(max_val) + abs(float(max_val)) * 0.1 + 1
                else:
                    low = min_val * 0.9
                    high = max_val * 1.1

            # Ensure bounds are valid integers for randint
            if np.issubdtype(dtype, np.integer):
                low_i = int(np.floor(low))
                high_i = int(np.ceil(high))
                if low_i >= high_i:
                    low_i = int(low_i) - 1
                    high_i = int(high_i) + 1
                synthetic_data[col] = list(np.random.randint(low_i, high_i + 1, size=num_rows))
            else:
                low_f = float(low)
                high_f = float(high)
                if low_f >= high_f:
                    low_f = low_f - 1.0
                    high_f = high_f + 1.0
                synthetic_data[col] = list(np.random.uniform(low_f, high_f, size=num_rows))

        # Common textual columns handled by Faker
        elif col_lower in ["name", "full_name", "fullname"]:
            synthetic_data[col] = [fake.name() for _ in range(num_rows)]
        elif col_lower in ["email", "email_address", "emailaddress"]:
            synthetic_data[col] = [fake.email() for _ in range(num_rows)]
        elif col_lower in ["job", "occupation"]:
            synthetic_data[col] = [fake.job() for _ in range(num_rows)]
        elif col_lower in ["id", "user_id", "index"]:
            synthetic_data[col] = list(range(1, num_rows + 1))
        else:
            # For other categorical/text columns, sample from observed values if any
            unique_values = original_df[col].dropna().unique()
            if len(unique_values) > 0:
                synthetic_data[col] = list(np.random.choice(unique_values, size=num_rows, replace=True))
            else:
                synthetic_data[col] = [f"Synthetic_{i}" for i in range(1, num_rows + 1)]

    final_df = pd.DataFrame(synthetic_data)

    # Reorder columns to match original if possible
    try:
        final_df = final_df[original_df.columns]
    except Exception:
        pass
    return final_df


def generate_ctgan_data(df, num_rows, epochs=10):
    try:
        # Placeholder logic
        st.warning("CTGAN placeholder running demo data. Actual CTGAN training not implemented here.")
        return generate_faker_data(df, num_rows)
    except Exception:
        st.error("CTGAN not available. Falling back to Faker demo.")
        return generate_faker_data(df, num_rows)



# --- MAIN CONTENT ---
if page == "🏠 DataForge Home":
    st.markdown('<div class="main-title">Welcome to DataForge</div>', unsafe_allow_html=True)
    st.markdown("""
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
        
# --- NEW PRE-PROCESSING PAGE ---
elif page == "⚙️ Pre-processing":
    st.markdown('<div class="main-title">⚙️ Data Pre-processing (Cleaning)</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Clean and prepare your uploaded dataset. This step handles missing values and drops overly sparse columns before synthesis.</p>', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first in the **⬆️ Upload Data** section.")
    else:
        df_orig = st.session_state.uploaded_data
        st.subheader("Missing Value Summary (Original Data)")
        missing_summary = pd.DataFrame({
            "Missing Count": df_orig.isna().sum(),
            "Missing %": (df_orig.isna().mean() * 100).round(2)
        }).sort_values(by="Missing %", ascending=False)

        if missing_summary["Missing Count"].sum() > 0:
            st.dataframe(missing_summary[missing_summary["Missing Count"] > 0], use_container_width=True)
        else:
            st.info("No missing values found in the uploaded data!")

        # User control for dropping columns
        threshold = st.slider(
            "Column Drop Threshold (% Missing)",
            min_value=5, max_value=100, value=20, step=5,
            help="Columns with a missing percentage higher than this value will be dropped. Remaining NaNs will be filled (mean/mode)."
        )

        if st.button("Apply Pre-processing (Clean Data)"):
            df_clean, dropped_cols = preprocess_data(df_orig, threshold)
            st.session_state.processed_data = df_clean

            if dropped_cols:
                st.warning(f"Dropped {len(dropped_cols)} column(s): {', '.join(dropped_cols)}")

            st.success("Data cleaned and ready for synthesis! Proceed to **🧪 Synthesization**.")
            st.subheader("Processed Data Preview")
            st.dataframe(df_clean.head())
            mc1, mc2 = st.columns(2)
            mc1.metric("Final Columns", len(df_clean.columns))
            mc2.metric("Remaining Missing Cells", int(df_clean.isna().sum().sum()))
        else:
            if st.session_state.processed_data is None:
                st.info("Click 'Apply Pre-processing' to clean the data.")
            else:
                st.success("Data is already processed. Proceed to **🧪 Synthesization**.")

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
            
# --- NEW POST-PROCESSING PAGE ---
elif page == "✨ Post-processing":
    st.markdown('<div class="main-title">✨ Post-processing (Quantitative Validation)</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Perform quantitative metrics to assess how well the synthetic data preserves the statistical properties of the original data.</p>', unsafe_allow_html=True)

    if st.session_state.processed_data is None or st.session_state.synthetic_data is None:
        st.warning("Please upload, process, and generate data first.")
    else:
        df_orig, df_synth = st.session_state.processed_data, st.session_state.synthetic_data

        # Identify common numeric columns
        numeric_cols_orig = df_orig.select_dtypes(include=np.number).columns
        numeric_cols_synth = df_synth.select_dtypes(include=np.number).columns
        common_numeric = list(set(numeric_cols_orig) & set(numeric_cols_synth))

        if common_numeric:
            st.subheader("Kolmogorov-Smirnov (KS) Test for Distribution Similarity")
            st.markdown("""
            The **KS Statistic** measures the maximum distance between the cumulative distributions of the original and synthetic data.
            A value closer to **0** indicates a better match in distribution.
            """)

            ks_results = []
            for col in common_numeric:
                # KS test is only reliable on continuous numerical data
                try:
                    orig_vals = df_orig[col].dropna()
                    synth_vals = df_synth[col].dropna()
                    if orig_vals.shape[0] < 2 or synth_vals.shape[0] < 2:
                        raise ValueError("Too few data points")
                    ks_stat, p_val = ks_2samp(orig_vals, synth_vals)
                    ks_results.append({
                        "Column": col,
                        "KS Statistic": float(ks_stat),
                        "P-Value": float(p_val)
                    })
                except Exception:
                    ks_results.append({
                        "Column": col,
                        "KS Statistic": None,
                        "P-Value": None
                    })

            ks_df = pd.DataFrame(ks_results)
            # Format for display
            display_df = ks_df.copy()
            display_df["KS Statistic"] = display_df["KS Statistic"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_df["P-Value"] = display_df["P-Value"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            st.dataframe(display_df, use_container_width=True)

            # Display a summary based on KS scores
            avg_ks = ks_df["KS Statistic"].dropna().mean()
            if pd.notna(avg_ks):
                if avg_ks < 0.15:
                    st.success(f"Average KS Statistic: {avg_ks:.3f}. Excellent distributional match!")
                elif avg_ks < 0.30:
                    st.warning(f"Average KS Statistic: {avg_ks:.3f}. Good match, but some variance exists.")
                else:
                    st.error(f"Average KS Statistic: {avg_ks:.3f}. The distributions differ significantly.")
        else:
            st.info("No common numeric columns found to run the KS test.")
            
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

