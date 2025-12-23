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

# DO NOT import pages here - will import lazily when needed
# This prevents circular imports and duplicate widget issues

if __name__ == "__main__":
    # ============ PAGE CONFIGURATION (MUST BE FIRST) ============
    st.set_page_config(
        page_title="Bio-ScreenNet | Drug Discovery",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/TuyenTrungLe/Computational-Drug-Discovery',
            'Report a bug': 'https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues',
            'About': 'Bio-ScreenNet - AI-Powered Drug Discovery Pipeline'
        }
    )

    # ============ CUSTOM CSS ============
    st.markdown("""
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main header */
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
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Navigation Menu Styling */
        div[data-testid="stRadio"] > label {
            background-color: transparent;
            border: 1px solid transparent;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: all 0.3s ease;
            cursor: pointer;
            width: 100%;
        }

        div[data-testid="stRadio"] > label:hover {
            background-color: #e3f2fd;
            border-color: #bbdefb;
            transform: translateX(5px);
        }

        /* Active selection styling */
        div[data-testid="stRadio"] > label[data-checked="true"] {
             background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
             color: white !important;
             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Make text inside radio buttons stand out */
        div[data-testid="stRadio"] p {
            font-size: 1rem;
            font-weight: 500;
            margin: 0;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #b3e5fc;
            border-left: 4px solid #0277bd;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            border: 2px solid #0288d1;
            color: #01579b;
            font-weight: 500;
        }
        
        .warning-box {
            background-color: #ffe0b2;
            border-left: 4px solid #f57c00;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            border: 2px solid #ef6c00;
            color: #e65100;
            font-weight: 500;
        }
        
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        /* Stage indicators */
        .stage-indicator {
            display: block;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            text-align: center;
            border: 1px solid rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .stage-indicator:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stage-1 { background-color: #e3f2fd; color: #1565c0; border-left: 4px solid #1565c0; }
        .stage-2 { background-color: #f3e5f5; color: #7b1fa2; border-left: 4px solid #7b1fa2; }
        .stage-3 { background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32; }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # ============ MAIN HEADER ============
    st.markdown("""
        <div class="main-header">
            <h1>🧬 Bio-ScreenNet</h1>
            <p>AI-Powered Multi-Stage Drug Discovery Pipeline</p>
        </div>
    """, unsafe_allow_html=True)

    # ============ SIDEBAR NAVIGATION ============
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    # Use radio button for direct visibility
    page = st.sidebar.radio(
        "📍 Select Page",
        [
            "🏠 Home",
            "💊 Single Compound Screening",
            "📊 Batch Analysis",
            "🎯 ADMET Filter",
            "🔬 Model Comparison",
            "📖 About & Documentation"
        ],
        index=0,
        key="main_page_selector"
    )

    st.sidebar.markdown("---")

    # Pipeline Stages
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

    # ============ MAIN CONTENT ROUTER ============
    if page == "🏠 Home":
        # Home Page
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
            try:
                st.image(str(project_root / "static" / "pipeline_architecture.png"),
                        caption="Pipeline Architecture", width="stretch")
            except:
                st.info("Pipeline architecture image not found")
        
        st.markdown("---")
        
        # Gate Logic
        st.header("⚡ Gate Logic: Core of Track C Two-Stage Pipeline")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fb8c00 0%, #ef6c00 100%); 
                    padding: 1.2rem; border-radius: 8px; color: white; margin: 0.5rem 0; 
                    border: 3px solid #e65100; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
            <h4 style='color: white; margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>🚦 Gate Logic: Critical Decision Point</h4>
            <p style='font-size: 0.95rem; margin: 0.3rem 0; font-weight: 600;'><b>Why Two-Stage Filtering?</b> Only run ADMET on Active compounds to save resources.</p>
            <hr style='border-color: rgba(255,255,255,0.5); margin: 0.5rem 0;'>
            <p style='font-size: 0.9rem; margin: 0.3rem 0; font-weight: 600;'><b>Rule:</b></p>
            <ul style='font-size: 0.9rem; margin: 0.3rem 0 0.3rem 1.2rem; padding: 0;'>
                <li>Inactive → 🛑 STOP → REJECT</li>
                <li>Active → ✅ CONTINUE to ADMET → KEEP/REJECT based on safety</li>
            </ul>
            <p style='font-size: 1rem; font-weight: bold; margin: 0;'>KEEP = Active <b>AND</b> Non-Toxic</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features
        st.header("Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: #e3f2fd; padding: 1rem; border-radius: 6px; border: 3px solid #1976d2;'>
            <h3 style='color: #0d47a1; font-weight: 700;'>💊 Single Compound</h3>
            <p style='color: #1565c0;'>Input SMILES to predict bioactivity with XAI explanations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #f3e5f5; padding: 1rem; border-radius: 6px; border: 3px solid #7b1fa2;'>
            <h3 style='color: #4a148c; font-weight: 700;'>📊 Batch Analysis</h3>
            <p style='color: #6a1b9a;'>Screen thousands of compounds from CSV files</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: #e8f5e9; padding: 1rem; border-radius: 6px; border: 3px solid #388e3c;'>
            <h3 style='color: #1b5e20; font-weight: 700;'>🎯 ADMET Filter</h3>
            <p style='color: #2e7d32;'>Apply safety filters to identify drug candidates</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Start
        st.header("Quick Start Guide")
        
        st.markdown("""
        ### Recommended Workflow (Track C Two-Stage Pipeline):
        
        **Step 1: Single Compound Screening** (Validation)
        - Navigate to "💊 Single Compound Screening"
        - Test with a SMILES: `CC(C)Cc1ccc(cc1)C(C)C(O)=O` (Ibuprofen)
        - View Bioactivity + ADMET results
        
        **Step 2: Batch Analysis** (Mass Screening)
        - Navigate to "📊 Batch Analysis"
        - Upload CSV with SMILES column
        - Process all compounds and get predictions
        
        **Step 3: ADMET Filter** (Optimize)
        - Navigate to "🎯 ADMET Filter"
        - Adjust thresholds for pIC50 and toxicity
        
        **Step 4: Model Comparison** (Evaluation)
        - Navigate to "🔬 Model Comparison"
        - Compare model performance
        """)

    elif page == "💊 Single Compound Screening":
        # Lazy import to avoid circular dependencies
        from pages import single_compound_page
        single_compound_page.render()

    elif page == "📊 Batch Analysis":
        # Lazy import to avoid circular dependencies
        from pages import batch_analysis_page
        batch_analysis_page.render()

    elif page == "🎯 ADMET Filter":
        # Lazy import to avoid circular dependencies
        from pages import admet_filter_page
        admet_filter_page.render()

    elif page == "🔬 Model Comparison":
        # Lazy import to avoid circular dependencies
        from pages import model_comparison_page
        model_comparison_page.render()

    elif page == "📖 About & Documentation":
        # Lazy import to avoid circular dependencies
        from pages import about_page
        about_page.render()

    # ============ FOOTER ============
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>🧬 Bio-ScreenNet | Data Analytics for Life Science - Capstone Project Track C</p>
            <p>Developed by: Lê Trung Tuyến & Bùi Hoàng Nhân | 2024-2025</p>
        </div>
    """, unsafe_allow_html=True)
