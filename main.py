
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import model modules
from models.linear_models import LinearModelsModule
from models.time_series_models import TimeSeriesModule
from models.panel_models import PanelDataModule
from models.advanced_models import AdvancedModelsModule
from utils.statistical_tests import StatisticalTestsModule
from utils.data_generator import DataGenerator
from utils.visualizations import VisualizationModule

# Page configuration
st.set_page_config(
    page_title="Advanced Econometric Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #2c3e50, #34495e);
        color: white;
        margin-top: 3rem;
        border-radius: 1rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# Header
st.markdown('<div class="main-header">📊 Advanced Econometric Simulator</div>', unsafe_allow_html=True)

# Initialize modules
data_gen = DataGenerator()
viz_module = VisualizationModule()
linear_module = LinearModelsModule()
ts_module = TimeSeriesModule()
panel_module = PanelDataModule()
advanced_module = AdvancedModelsModule()
stats_module = StatisticalTestsModule()

# Sidebar
st.sidebar.title("🎛️ Control Panel")

# Theme toggle
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])

# Display mode toggle
display_mode = st.sidebar.selectbox("🖥️ Display Mode", ["Desktop", "Mobile"])
if theme == "Dark":
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .main-header {
            color: white !important;
        }
        .section-header {
            color: #ecf0f1 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply mobile mode styles
if display_mode == "Mobile":
    st.markdown("""
    <style>
        .main-header {
            font-size: 2rem;
        }
        .section-header {
            font-size: 1.4rem;
        }
        .stButton>button {
            padding: 0.4rem 1rem;
            font-size: 0.9rem;
        }
        div[data-testid="stSidebar"] {
            width: 200px !important;
        }
        .css-1d391kg {
            padding: 1rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Model category selection
st.sidebar.markdown("---")
model_category = st.sidebar.selectbox(
    "📈 Model Category",
    ["🏠 Home", "📊 Linear Models", "⏰ Time Series Models", 
     "📋 Panel Data Models", "🔬 Advanced Models", "🧪 Statistical Tests"]
)

# Data input section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Input")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["📊 Sample Data", "📤 Upload File", "✏️ Manual Entry", "🌐 API Data"]
)

# Handle data input
if data_source == "📊 Sample Data":
    sample_type = st.sidebar.selectbox(
        "Sample data type:",
        ["Linear Regression", "Time Series", "Panel Data", "Classification"]
    )
    if st.sidebar.button("Generate Sample Data"):
        st.session_state.data = data_gen.generate_sample_data(sample_type.lower().replace(" ", "_"))
        st.success("Sample data generated successfully!")

elif data_source == "📤 Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

elif data_source == "✏️ Manual Entry":
    st.sidebar.info("Manual data entry feature")
    n_rows = st.sidebar.number_input("Number of rows:", min_value=5, max_value=100, value=10)
    n_cols = st.sidebar.number_input("Number of columns:", min_value=2, max_value=10, value=3)
    if st.sidebar.button("Create Empty DataFrame"):
        cols = [f"Variable_{i+1}" for i in range(n_cols)]
        st.session_state.data = pd.DataFrame(np.zeros((n_rows, n_cols)), columns=cols)

elif data_source == "🌐 API Data":
    api_source = st.sidebar.selectbox(
        "Select API source:",
        ["FRED", "World Bank", "Yahoo Finance", "Custom API"]
    )
    st.sidebar.info("API integration feature - Coming soon!")

# Main content area
if model_category == "🏠 Home":
    # Welcome page
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>📊 Linear Models</h3>
        <p>OLS, Ridge, Lasso, Elastic Net, Polynomial Regression</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>⏰ Time Series</h3>
        <p>ARIMA, SARIMA, VAR, GARCH, Exponential Smoothing</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
        <h3>🧪 Statistical Tests</h3>
        <p>Unit Root, Cointegration, Diagnostic, Causality Tests</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start guide
    st.markdown('<div class="section-header">🚀 Quick Start Guide</div>', unsafe_allow_html=True)

    st.markdown("""
    1. **Select a Model Category** from the sidebar
    2. **Load or Generate Data** using the data input options
    3. **Configure Model Parameters** using interactive controls
    4. **Run Analysis** and view results with interactive visualizations
    5. **Export Results** for further analysis
    """)

    # Feature highlights
    st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🎯 Comprehensive Models**
        - Linear Regression (OLS, Ridge, Lasso)
        - Time Series (ARIMA, SARIMA, VAR)
        - Panel Data (Fixed/Random Effects)
        - Advanced Models (Logit, Probit, Tobit)

        **📊 Interactive Visualizations**
        - Real-time parameter adjustment
        - Animated time series plots
        - 3D surface plots for optimization
        - Bootstrap confidence intervals
        """)

    with col2:
        st.markdown("""
        **🧪 Statistical Testing**
        - Unit Root Tests (ADF, KPSS, PP)
        - Cointegration (Engle-Granger, Johansen)
        - Diagnostic Tests (DW, BP, JB)
        - Causality Tests (Granger, Toda-Yamamoto)

        **📁 Data Management**
        - Multiple import formats
        - API integration
        - Data preprocessing tools
        - Export capabilities
        """)

elif model_category == "📊 Linear Models":
    linear_module.run(st.session_state.data)

elif model_category == "⏰ Time Series Models":
    ts_module.run(st.session_state.data)

elif model_category == "📋 Panel Data Models":
    panel_module.run(st.session_state.data)

elif model_category == "🔬 Advanced Models":
    advanced_module.run(st.session_state.data)

elif model_category == "🧪 Statistical Tests":
    stats_module.run(st.session_state.data)

# Footer
st.markdown("""
<div class="footer">
    <h3>Created by HAMDI Boulanouar</h3>
    <p>Advanced Econometric Simulator | Version 1.0 | 2024</p>
    <p>📧 Contact | 🔗 LinkedIn | 📚 Documentation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [Documentation](https://github.com)
- [Tutorial Videos](https://youtube.com)
- [Sample Datasets](https://data.gov)
""")

st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This comprehensive econometric simulator provides professional tools for "
    "statistical analysis, regression modeling, and hypothesis testing. "
    "Built for researchers, students, and data scientists."
)
