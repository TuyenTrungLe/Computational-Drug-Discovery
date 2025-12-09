"""
Bio-ScreenNet: Multi-Stage Drug Discovery Pipeline
Main Streamlit Application Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Bio-ScreenNet | Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/TuyenTrungLe/Computational-Drug-Discovery',
        'Report a bug': 'https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues',
        'About': '''
        # Bio-ScreenNet
        AI-Powered Virtual Drug Screening System

        **Track C: Computational Drug Discovery**

        Team Members:
        - Lê Trung Tuyến
        - Bùi Hoàng Nhân
        '''
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #06A77D;
        --warning-color: #D00000;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }

    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    /* Warning/Info boxes */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed var(--primary-color);
        border-radius: 8px;
        padding: 1rem;
    }

    /* Pipeline stage indicators */
    .stage-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.25rem;
    }

    .stage-1 { background-color: #e3f2fd; color: #1976d2; }
    .stage-2 { background-color: #f3e5f5; color: #7b1fa2; }
    .stage-3 { background-color: #e8f5e9; color: #388e3c; }

    /* SMILES input box */
    .smiles-input {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
    }

    /* Results table styling */
    .results-table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
    }

    .results-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        text-align: left;
    }

    .results-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }

    .results-table tr:hover {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🧬 Bio-ScreenNet</h1>
        <p>AI-Powered Multi-Stage Drug Discovery Pipeline</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Navigation options
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "💊 Single Compound Screening",
        "📊 Batch Analysis",
        "🎯 ADMET Filter",
        "🔬 Model Comparison",
        "📖 About & Documentation"
    ],
    index=0
)

st.sidebar.markdown("---")

# Pipeline Status Indicator
st.sidebar.subheader("Pipeline Stages")
st.sidebar.markdown("""
<div class="stage-indicator stage-1">Stage 1: Bioactivity</div>
<div class="stage-indicator stage-2">Stage 2: ADMET Safety</div>
<div class="stage-indicator stage-3">Stage 3: XAI</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Project Info
st.sidebar.subheader("Project Info")
st.sidebar.info("""
**Track C: Computational Drug Discovery**

**Team:**
- Lê Trung Tuyến
- Bùi Hoàng Nhân

**Models:**
- Random Forest (Baseline)
- LSTM/GRU (Deep Learning)
- ADMET Classifiers
""")

# Main Content Area
if page == "🏠 Home":
    # Overview Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Welcome to Bio-ScreenNet")
        st.markdown("""
        ### Accelerating Drug Discovery with AI

        Drug discovery traditionally takes **10-15 years** and costs **billions of dollars**.
        Bio-ScreenNet leverages Machine Learning and Deep Learning to accelerate early-stage
        drug discovery by:

        - 🎯 **Predicting Bioactivity (pIC50)** from SMILES representations
        - 🛡️ **Filtering compounds** using ADMET safety models
        - 🔍 **Visualizing atom-level explanations** with XAI
        - 📈 **Comparing model performance** across multiple architectures
        """)

        st.markdown("""
        <div class="info-box">
        <strong>Target Protein:</strong> CHEMBL220 – Acetylcholinesterase<br>
        <strong>Disease Context:</strong> Alzheimer's Disease / Neurodegenerative Disorders<br>
        <strong>Dataset Source:</strong> ChEMBL Database + MoleculeNet (ADMET)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(str(project_root / "static" / "pipeline_architecture.png"),
                 caption="Pipeline Architecture", use_container_width=True)

    st.markdown("---")

    # Features Overview
    st.header("Key Features")

    feature_cols = st.columns(3)

    with feature_cols[0]:
        st.markdown("""
        <div class="info-card">
        <h3>💊 Single Compound</h3>
        <p>Input a single SMILES string to predict bioactivity and view XAI explanations</p>
        </div>
        """, unsafe_allow_html=True)

    with feature_cols[1]:
        st.markdown("""
        <div class="info-card">
        <h3>📊 Batch Analysis</h3>
        <p>Upload CSV files to screen thousands of compounds simultaneously</p>
        </div>
        """, unsafe_allow_html=True)

    with feature_cols[2]:
        st.markdown("""
        <div class="info-card">
        <h3>🎯 ADMET Filter</h3>
        <p>Apply safety filters to identify promising and safe drug candidates</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Start Guide
    st.header("Quick Start Guide")

    st.markdown("""
    ### How to Use Bio-ScreenNet:

    1. **Single Compound Screening**
       - Navigate to "💊 Single Compound Screening"
       - Enter a SMILES string (e.g., `CC(C)Cc1ccc(cc1)C(C)C(O)=O`)
       - Select prediction model (Random Forest or LSTM/GRU)
       - View pIC50 prediction and XAI visualization

    2. **Batch Analysis**
       - Navigate to "📊 Batch Analysis"
       - Upload CSV file with SMILES column
       - Run predictions on all compounds
       - Download filtered results

    3. **ADMET Safety Filtering**
       - Navigate to "🎯 ADMET Filter"
       - Apply toxicity, solubility, or BBBP filters
       - Export safe candidates for further analysis

    4. **Model Comparison**
       - Navigate to "🔬 Model Comparison"
       - Compare Random Forest vs LSTM/GRU performance
       - View evaluation metrics (R², RMSE, etc.)
    """)

    st.markdown("---")

    # Example SMILES
    st.header("Example SMILES for Testing")

    examples_df = {
        "Compound": ["Ibuprofen", "Aspirin", "Caffeine", "Dopamine", "Acetylcholine"],
        "SMILES": [
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "NCCc1ccc(O)c(O)c1",
            "CC(=O)OCC[N+](C)(C)C"
        ],
        "Expected Activity": ["Anti-inflammatory", "Anti-inflammatory", "Stimulant", "Neurotransmitter", "Neurotransmitter"]
    }

    st.table(examples_df)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Note:</strong> These are example compounds for testing purposes.
    Actual bioactivity predictions depend on the specific target protein and trained models.
    </div>
    """, unsafe_allow_html=True)

elif page == "💊 Single Compound Screening":
    from pages import single_compound_page
    single_compound_page.render()

elif page == "📊 Batch Analysis":
    from pages import batch_analysis_page
    batch_analysis_page.render()

elif page == "🎯 ADMET Filter":
    from pages import admet_filter_page
    admet_filter_page.render()

elif page == "🔬 Model Comparison":
    from pages import model_comparison_page
    model_comparison_page.render()

elif page == "📖 About & Documentation":
    from pages import about_page
    about_page.render()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🧬 Bio-ScreenNet | Data Analytics for Life Science - Capstone Project Track C</p>
        <p>Developed by: Lê Trung Tuyến & Bùi Hoàng Nhân | 2024-2025</p>
    </div>
""", unsafe_allow_html=True)
