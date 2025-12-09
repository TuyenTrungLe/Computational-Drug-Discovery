"""
About & Documentation Page
Project information, usage guide, and references
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def render():
    """Render the About & Documentation page"""

    st.title("📖 About & Documentation")
    st.markdown("Learn more about Bio-ScreenNet and how to use it effectively")

    st.markdown("---")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Project Overview",
        "🚀 Quick Start Guide",
        "📚 Technical Details",
        "📄 References",
        "👥 Team & Contact"
    ])

    with tab1:
        st.header("Project Overview")

        st.markdown("""
        ### 🧬 Bio-ScreenNet: Multi-Stage Drug Discovery Pipeline

        **Bio-ScreenNet** is an AI-powered virtual drug screening system designed to accelerate
        early-stage drug discovery by simultaneously optimizing **bioactivity** and **safety (ADMET)**
        for candidate compounds.

        ### 🎯 Project Goals:

        1. **Predict Bioactivity**: Estimate binding affinity (pIC50) from SMILES representations
        2. **Evaluate Safety**: Filter compounds using ADMET property predictions
        3. **Explain Predictions**: Visualize atom-level contributions with XAI
        4. **Accelerate Discovery**: Reduce time and cost of identifying drug candidates

        ### 🏆 Capstone Project - Track C

        This project is part of the **Data Analytics for Life Science** capstone program,
        focusing on **Computational Drug Discovery** (Track C).

        **Requirements Met:**
        - ✅ Multi-task pipeline (Bioactivity + ADMET)
        - ✅ Baseline model (Random Forest)
        - ✅ Advanced model (LSTM/GRU)
        - ✅ Transfer learning (ChemBERTa)
        - ✅ Rigorous evaluation (R², RMSE, MAE)
        - ✅ Explainable AI (XAI) with RDKit similarity maps
        - ✅ Streamlit web application
        """)

        st.markdown("---")

        st.subheader("🔬 Target Protein & Disease")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("""
            **Target Protein:**
            - CHEMBL220 – Acetylcholinesterase (AChE)
            - Enzyme that breaks down acetylcholine
            - Key target for neurodegenerative diseases

            **Biological Role:**
            - Regulates neurotransmission
            - Controls synaptic acetylcholine levels
            - Essential for cognitive function
            """)

        with info_col2:
            st.markdown("""
            **Disease Context:**
            - Alzheimer's Disease
            - Dementia
            - Myasthenia Gravis
            - Other neurodegenerative disorders

            **Therapeutic Strategy:**
            - AChE inhibitors increase acetylcholine
            - Improve cognitive symptoms
            - Currently approved drugs: Donepezil, Rivastigmine
            """)

        st.markdown("---")

        st.subheader("📊 Pipeline Architecture")

        st.image(str(project_root / "static" / "pipeline_architecture.png"),
                caption="Bio-ScreenNet Three-Stage Pipeline", use_container_width=True)

        st.markdown("""
        ### Three-Stage Pipeline:

        **Stage 1: Bioactivity Prediction**
        - Input: SMILES string
        - Models: Random Forest, LSTM/GRU, Transfer Learning
        - Output: Predicted pIC50 (binding affinity)
        - Threshold: pIC50 ≥ 6.0 for "Active"

        **Stage 2: ADMET Safety Filter**
        - Input: Active compounds from Stage 1
        - Models: Toxicity (Tox21), Solubility (ESOL), BBBP
        - Output: Safe compounds passing all filters
        - Filters: Non-toxic, good solubility, Lipinski-compliant

        **Stage 3: Explainable AI**
        - Input: Promising candidates
        - Method: RDKit Similarity Maps, Gradient-based attribution
        - Output: Atom contribution visualizations
        - Purpose: Identify pharmacophores for medicinal chemistry
        """)

    with tab2:
        st.header("Quick Start Guide")

        st.markdown("""
        ### 🚀 Getting Started with Bio-ScreenNet

        Follow these steps to start screening compounds:
        """)

        st.markdown("---")

        st.subheader("1️⃣ Single Compound Screening")

        with st.expander("Step-by-Step Instructions", expanded=True):
            st.markdown("""
            **A. Navigate to "💊 Single Compound Screening"**

            **B. Enter a SMILES string**
            ```
            Example: CC(C)Cc1ccc(cc1)C(C)C(O)=O
            ```

            **C. Select prediction model**
            - Random Forest (Fast)
            - LSTM/GRU (Accurate)
            - Both Models (Comparison)

            **D. Configure options**
            - Show molecular descriptors
            - Show fingerprint visualization
            - Set confidence threshold

            **E. Click "Predict Bioactivity"**

            **F. Review results**
            - Predicted pIC50 and IC50
            - Activity classification
            - Molecular structure
            - Lipinski's Rule compliance
            - XAI atom contributions
            - Pharmacophore highlights

            **G. Export results**
            - Download as CSV or JSON
            """)

        st.markdown("---")

        st.subheader("2️⃣ Batch Analysis")

        with st.expander("Batch Processing Workflow"):
            st.markdown("""
            **A. Prepare CSV file**

            Required format:
            ```csv
            SMILES,compound_id,name
            CC(C)Cc1ccc(cc1)C(C)C(O)=O,CHEM001,Ibuprofen
            CC(=O)Oc1ccccc1C(=O)O,CHEM002,Aspirin
            ```

            **B. Upload CSV file**
            - Navigate to "📊 Batch Analysis"
            - Use file uploader or download sample template

            **C. Configure settings**
            - Select prediction model
            - Set batch size
            - Enable ADMET filters (optional)

            **D. Run predictions**
            - Click "Run Batch Prediction"
            - Monitor progress bar

            **E. Filter and analyze results**
            - Filter by activity, pIC50 range, confidence
            - View distribution plots
            - Paginate through results

            **F. Export filtered candidates**
            - All results
            - Filtered results
            - Active compounds only
            """)

        st.markdown("---")

        st.subheader("3️⃣ ADMET Safety Filtering")

        with st.expander("Apply Safety Filters"):
            st.markdown("""
            **A. Load compounds**
            - Use previous batch results
            - Upload new CSV
            - Enter SMILES manually

            **B. Configure ADMET filters**

            **Toxicity:**
            - Tox21 (max probability threshold)
            - ClinTox (clinical toxicity)
            - Mutagenicity

            **Physicochemical:**
            - Solubility (LogS range)
            - BBBP requirement
            - Lipinski's Rule of Five

            **C. Run ADMET analysis**
            - Click "Run ADMET Analysis"
            - View pass rates

            **D. Review results**
            - All compounds tab
            - Passed filters tab
            - Failed compounds tab
            - Distribution visualizations

            **E. Export candidates**
            - Passed compounds (for further testing)
            - All ADMET results
            - Failed compounds (for analysis)
            """)

        st.markdown("---")

        st.subheader("4️⃣ Best Practices")

        st.info("""
        **Workflow Recommendations:**

        1. **Start with Single Compound** - Test the system with known drugs
        2. **Use Batch Analysis** - Screen your compound library
        3. **Apply ADMET Filters** - Prioritize safe candidates
        4. **Review XAI** - Understand why compounds are active
        5. **Iterate** - Modify structures based on pharmacophore insights

        **Tips:**
        - Validate SMILES before uploading large batches
        - Use Random Forest for initial screening (fast)
        - Use LSTM/GRU for final ranking (accurate)
        - Set appropriate confidence thresholds (0.7-0.8)
        - Check Lipinski's Rule violations
        - Export results regularly
        """)

    with tab3:
        st.header("Technical Details")

        st.markdown("---")

        tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs([
            "🗄️ Datasets",
            "🤖 Models",
            "📊 Evaluation",
            "🔍 XAI Methods"
        ])

        with tech_tab1:
            st.subheader("Datasets")

            st.markdown("""
            ### 1. Bioactivity Dataset (ChEMBL)

            **Source:** ChEMBL Database (https://www.ebi.ac.uk/chembl/)

            **Target:** CHEMBL220 (Acetylcholinesterase)

            **Data Collection:**
            - Query target using ChEMBL API
            - Filter for IC50 measurements
            - Remove duplicates and inconsistent data
            - Convert IC50 to pIC50: pIC50 = -log10(IC50 * 10^-9)

            **Dataset Statistics:**
            - Total compounds: ~5,000-10,000
            - Active (pIC50 ≥ 6.0): ~40%
            - Inactive (pIC50 < 6.0): ~60%
            - Split: 70% train, 15% validation, 15% test

            **Preprocessing:**
            - SMILES canonicalization
            - Salt stripping
            - Neutralization
            - Standardization with RDKit

            ---

            ### 2. ADMET Datasets (MoleculeNet - DeepChem)

            **A. Tox21 (Toxicity)**
            - 12 toxicity targets
            - ~8,000 compounds
            - Binary classification
            - Imbalanced classes

            **B. ESOL (Solubility)**
            - Aqueous solubility (LogS)
            - ~1,100 compounds
            - Regression task
            - Range: -10 to 0

            **C. BBBP (Blood-Brain Barrier)**
            - Binary permeability
            - ~2,000 compounds
            - Classification task

            **Molecular Descriptors:**
            - Lipinski descriptors (MW, LogP, HBD, HBA)
            - TPSA, rotatable bonds
            - Aromatic rings, formal charge
            - Morgan fingerprints (2048-bit, radius=2)
            - PubChem fingerprints
            """)

        with tech_tab2:
            st.subheader("Model Architectures")

            st.markdown("""
            ### 1. Random Forest (Baseline)

            **Architecture:**
            ```python
            RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            ```

            **Features:**
            - 2048-bit Morgan fingerprints (radius=2)
            - 200 molecular descriptors
            - Total: 2248 features

            **Training:**
            - Cross-validation: 5-fold
            - Hyperparameter tuning: GridSearchCV
            - Early stopping: Not applicable

            ---

            ### 2. LSTM/GRU (Deep Learning)

            **Architecture:**
            ```python
            Sequential([
                Embedding(vocab_size=50, embedding_dim=128),
                Bidirectional(LSTM(256, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(128)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1)  # Regression output
            ])
            ```

            **Input Processing:**
            - Character-level SMILES tokenization
            - Max sequence length: 100
            - Padding: post
            - Vocabulary: 50 unique characters

            **Training:**
            - Optimizer: Adam (lr=0.001)
            - Loss: MSE
            - Batch size: 32
            - Epochs: 100
            - Early stopping: patience=10
            - Learning rate scheduler: ReduceLROnPlateau

            **Data Augmentation:**
            - SMILES enumeration
            - Random shuffling of atoms

            ---

            ### 3. Transfer Learning (ChemBERTa)

            **Pre-trained Model:**
            - ChemBERTa-77M-MTR (from HuggingFace)
            - Pre-trained on 77M molecules
            - Multi-task regression objective

            **Fine-tuning:**
            - Freeze base layers
            - Add task-specific head
            - Train on target-specific data
            - Lower learning rate (1e-5)

            ---

            ### 4. ADMET Models

            **Toxicity (Tox21):**
            - XGBoost classifier
            - Handle class imbalance with SMOTE
            - Ensemble of 12 binary classifiers

            **Solubility (ESOL):**
            - Random Forest regressor
            - Features: descriptors + fingerprints

            **BBBP:**
            - SVM classifier
            - RBF kernel
            - Probability calibration
            """)

        with tech_tab3:
            st.subheader("Evaluation Metrics")

            st.markdown("""
            ### Regression Metrics (Bioactivity)

            **1. R² Score (Coefficient of Determination)**
            ```
            R² = 1 - (SS_res / SS_tot)
            ```
            - Range: -∞ to 1
            - Best: 1 (perfect prediction)
            - Interpretation: Proportion of variance explained

            **2. RMSE (Root Mean Squared Error)**
            ```
            RMSE = sqrt(mean((y_true - y_pred)²))
            ```
            - Units: Same as target (pIC50)
            - Lower is better
            - Penalizes large errors

            **3. MAE (Mean Absolute Error)**
            ```
            MAE = mean(|y_true - y_pred|)
            ```
            - Units: Same as target
            - Lower is better
            - Less sensitive to outliers than RMSE

            ---

            ### Classification Metrics (ADMET)

            **1. AUPRC (Area Under Precision-Recall Curve)**
            - Better for imbalanced datasets
            - Range: 0 to 1
            - Focuses on positive class

            **2. F1-Score**
            ```
            F1 = 2 × (Precision × Recall) / (Precision + Recall)
            ```
            - Harmonic mean of precision and recall
            - Good for imbalanced data

            **3. Confusion Matrix**
            - True Positives, False Positives
            - True Negatives, False Negatives

            ---

            ### Cross-Validation

            **Strategy:**
            - 5-fold cross-validation
            - Stratified splits for classification
            - Scaffold splits for molecular data
            - Prevent data leakage

            **Metrics Reported:**
            - Mean ± standard deviation across folds
            - Best fold performance
            - Worst fold performance
            """)

        with tech_tab4:
            st.subheader("Explainable AI (XAI) Methods")

            st.markdown("""
            ### 1. RDKit Similarity Maps

            **Method:**
            - Calculate fingerprint differences
            - Map contributions to atoms
            - Visualize with color gradient

            **Color Scheme:**
            - 🟢 Green: Positive contribution (increases activity)
            - 🔴 Red: Negative contribution (decreases activity)
            - ⚪ White: Neutral (no significant contribution)

            **Use Cases:**
            - Identify pharmacophores
            - Guide structure modification
            - Understand SAR (Structure-Activity Relationships)

            ---

            ### 2. Gradient-Based Attribution

            **For Deep Learning Models:**

            **A. Integrated Gradients**
            ```python
            attribution = ∫(∂f/∂x) dx
            ```
            - Compute gradients along interpolation path
            - Attribute prediction to input features
            - Satisfies axioms (sensitivity, implementation invariance)

            **B. Saliency Maps**
            - Gradient of output w.r.t. input
            - Highlights important atoms/bonds
            - Fast computation

            ---

            ### 3. SHAP (SHapley Additive exPlanations)

            **For Random Forest:**
            - TreeSHAP algorithm
            - Game-theoretic attribution
            - Shows feature importance
            - Additive and consistent

            **Visualizations:**
            - Force plots
            - Summary plots
            - Dependence plots

            ---

            ### 4. Pharmacophore Identification

            **Automated Detection:**
            - Common substructures in active compounds
            - Functional group analysis
            - 3D conformer-based pharmacophores

            **Output:**
            - Hydrophobic regions
            - H-bond donors/acceptors
            - Aromatic rings
            - Charged groups
            """)

    with tab4:
        st.header("References")

        st.markdown("""
        ### 📚 Key References

        #### Bioactivity Prediction:
        1. **Belaidi, A. et al. (2024)**. "Predicting pIC50 using Deep Learning on SMILES."
           *Journal of Cheminformatics*, 16(1), 42.

        2. **Gaulton, A. et al. (2017)**. "The ChEMBL database in 2017."
           *Nucleic Acids Research*, 45(D1), D945-D954.

        3. **Rogers, D. & Hahn, M. (2010)**. "Extended-Connectivity Fingerprints."
           *Journal of Chemical Information and Modeling*, 50(5), 742-754.

        #### ADMET Prediction:
        4. **Wu, Z. et al. (2018)**. "MoleculeNet: A benchmark for molecular machine learning."
           *Chemical Science*, 9(2), 513-530.

        5. **Daina, A. et al. (2017)**. "SwissADME: A free web tool to evaluate pharmacokinetics."
           *Scientific Reports*, 7, 42717.

        #### Deep Learning for Drug Discovery:
        6. **Stokes, J. M. et al. (2020)**. "A Deep Learning Approach to Antibiotic Discovery."
           *Cell*, 180(4), 688-702.

        7. **Chithrananda, S. et al. (2020)**. "ChemBERTa: Large-Scale Self-Supervised Pretraining."
           *arXiv preprint* arXiv:2010.09885.

        #### Explainable AI:
        8. **Riniker, S. & Landrum, G. A. (2013)**. "Similarity maps - a visualization strategy."
           *Journal of Cheminformatics*, 5(1), 43.

        9. **Sundararajan, M. et al. (2017)**. "Axiomatic Attribution for Deep Networks."
           *ICML*, 3319-3328.

        #### Drug Discovery Pipelines:
        10. **Walters, W. P. & Barzilay, R. (2021)**. "Applications of deep learning in molecule generation."
            *Accounts of Chemical Research*, 54(2), 263-270.

        ---

        ### 🔗 Useful Links

        **Databases:**
        - [ChEMBL](https://www.ebi.ac.uk/chembl/) - Bioactivity database
        - [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Chemical information
        - [MoleculeNet](http://moleculenet.ai/) - Benchmark datasets

        **Tools & Libraries:**
        - [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
        - [DeepChem](https://deepchem.io/) - Deep learning for chemistry
        - [Streamlit](https://streamlit.io/) - Web app framework

        **Educational Resources:**
        - [DeepChem Tutorials](https://deepchem.readthedocs.io/)
        - [RDKit Cookbook](https://www.rdkit.org/docs/Cookbook.html)
        - [Life Science AI Course Materials](https://github.com/)
        """)

    with tab5:
        st.header("Team & Contact")

        st.markdown("---")

        st.subheader("👥 Project Team")

        team_col1, team_col2 = st.columns(2)

        with team_col1:
            st.markdown("""
            <div class="info-card">
            <h3>Lê Trung Tuyến</h3>
            <p><strong>Role:</strong> Backend Development, Model Training</p>
            <p><strong>Responsibilities:</strong></p>
            <ul>
                <li>Data collection and preprocessing</li>
                <li>Model architecture design</li>
                <li>Training and evaluation</li>
                <li>XAI implementation</li>
            </ul>
            <p><strong>Email:</strong> letrungtuyen2002@gmail.com</p>
            </div>
            """, unsafe_allow_html=True)

        with team_col2:
            st.markdown("""
            <div class="info-card">
            <h3>Bùi Hoàng Nhân</h3>
            <p><strong>Role:</strong> Frontend Development, UI/UX</p>
            <p><strong>Responsibilities:</strong></p>
            <ul>
                <li>Streamlit application design</li>
                <li>User interface implementation</li>
                <li>Visualization components</li>
                <li>Documentation</li>
            </ul>
            <p><strong>Email:</strong> [Email]</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("📧 Contact & Support")

        st.markdown("""
        ### Get in Touch:

        **GitHub Repository:**
        🔗 [github.com/TuyenTrungLe/Computational-Drug-Discovery](https://github.com/TuyenTrungLe/Computational-Drug-Discovery)

        **Report Issues:**
        🐛 [GitHub Issues](https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues)

        **Project Documentation:**
        📖 [README.md](https://github.com/TuyenTrungLe/Computational-Drug-Discovery/blob/main/README.md)

        ---

        ### 🎓 Academic Context:

        **Course:** Data Analytics for Life Science

        **Project Track:** Track C - Computational Drug Discovery

        **Institution:** [Your University/Institution]

        **Academic Year:** 2024-2025

        **Supervisor:** [Supervisor Name]
        """)

        st.markdown("---")

        st.subheader("📜 License & Usage")

        st.info("""
        **MIT License**

        This project is open-source and available under the MIT License.

        You are free to:
        - Use the code for personal or commercial purposes
        - Modify and adapt the code
        - Distribute the code

        **Citation:**
        If you use this work in your research, please cite:
        ```
        Lê, T. T., & Bùi, H. N. (2024). Bio-ScreenNet: Multi-Stage Drug Discovery Pipeline.
        GitHub repository: https://github.com/TuyenTrungLe/Computational-Drug-Discovery
        ```
        """)

        st.markdown("---")

        st.subheader("🙏 Acknowledgments")

        st.markdown("""
        We would like to thank:

        - **Course instructors** for guidance and support
        - **ChEMBL team** for providing open bioactivity data
        - **DeepChem community** for ADMET datasets and tools
        - **RDKit developers** for cheminformatics toolkit
        - **Streamlit team** for the amazing web framework
        - **Open-source community** for inspiration and resources
        """)

        st.success("""
        **Thank you for using Bio-ScreenNet!**

        We hope this tool accelerates your drug discovery research.
        """)
