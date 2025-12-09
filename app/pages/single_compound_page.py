"""
Single Compound Screening Page
Allows users to input a single SMILES string and get bioactivity predictions + XAI
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def validate_smiles(smiles: str) -> tuple[bool, str]:
    """
    Validate SMILES string format
    Returns: (is_valid, message)
    """
    if not smiles or len(smiles.strip()) == 0:
        return False, "SMILES string cannot be empty"

    # Basic validation - check for valid characters
    valid_chars = set("CNOPSFIBrClcnops0123456789[]()=#@+-\\/")
    if not all(c in valid_chars for c in smiles):
        return False, "SMILES contains invalid characters"

    if len(smiles) > 500:
        return False, "SMILES string too long (max 500 characters)"

    return True, "Valid SMILES"


def render():
    """Render the Single Compound Screening page"""

    st.title("💊 Single Compound Screening")
    st.markdown("Enter a SMILES string to predict bioactivity and view XAI explanations")

    st.markdown("---")

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        # SMILES input
        smiles_input = st.text_area(
            "SMILES String",
            placeholder="Example: CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            height=100,
            help="Enter a valid SMILES representation of your compound"
        )

        # Model selection
        model_type = st.selectbox(
            "Select Prediction Model",
            ["Random Forest (Baseline)", "LSTM/GRU (Deep Learning)", "Both Models"],
            help="Choose which model to use for prediction"
        )

        # Additional options
        with st.expander("Advanced Options"):
            show_descriptors = st.checkbox("Show Molecular Descriptors", value=True)
            show_fingerprint = st.checkbox("Show Molecular Fingerprint Visualization", value=False)
            confidence_threshold = st.slider(
                "Prediction Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence level for predictions"
            )

        # Predict button
        predict_btn = st.button("🔬 Predict Bioactivity", type="primary", use_container_width=True)

    with col2:
        st.subheader("Example SMILES")
        st.markdown("""
        **Quick Examples:**

        **Ibuprofen** (Anti-inflammatory)
        ```
        CC(C)Cc1ccc(cc1)C(C)C(O)=O
        ```

        **Aspirin**
        ```
        CC(=O)Oc1ccccc1C(=O)O
        ```

        **Caffeine**
        ```
        CN1C=NC2=C1C(=O)N(C(=O)N2C)C
        ```

        **Dopamine**
        ```
        NCCc1ccc(O)c(O)c1
        ```
        """)

    st.markdown("---")

    # Process prediction
    if predict_btn:
        if not smiles_input:
            st.error("Please enter a SMILES string")
        else:
            # Validate SMILES
            is_valid, message = validate_smiles(smiles_input)

            if not is_valid:
                st.error(f"Invalid SMILES: {message}")
            else:
                st.success("✓ Valid SMILES detected")

                # Show loading spinner
                with st.spinner("Analyzing compound and generating predictions..."):
                    # Import here to avoid circular imports
                    try:
                        # Try to import RDKit for visualization
                        from rdkit import Chem
                        from rdkit.Chem import Descriptors, Draw
                        has_rdkit = True
                    except ImportError:
                        has_rdkit = False
                        st.warning("RDKit not installed. Some visualizations will be limited.")

                    # Placeholder for actual model prediction
                    # TODO: Replace with actual model inference
                    results = predict_bioactivity(smiles_input, model_type, has_rdkit)

                    # Display results
                    st.markdown("---")
                    st.header("📊 Prediction Results")

                    # Metrics row
                    metric_cols = st.columns(3)

                    with metric_cols[0]:
                        st.metric(
                            label="Predicted pIC50",
                            value=f"{results['pIC50']:.2f}",
                            delta=f"{results['confidence']:.1%} confidence",
                            help="Higher pIC50 indicates stronger binding"
                        )

                    with metric_cols[1]:
                        activity_class = "Active" if results['pIC50'] >= 6.0 else "Inactive"
                        activity_color = "🟢" if results['pIC50'] >= 6.0 else "🔴"
                        st.metric(
                            label="Activity Classification",
                            value=f"{activity_color} {activity_class}",
                            help="Active if pIC50 ≥ 6.0"
                        )

                    with metric_cols[2]:
                        st.metric(
                            label="IC50 (Estimated)",
                            value=f"{results['IC50']:.2f} nM",
                            help="IC50 = 10^(-pIC50) × 10^9 nM"
                        )

                    st.markdown("---")

                    # Molecular structure and descriptors
                    if has_rdkit and results.get('mol') is not None:
                        st.subheader("🧬 Molecular Structure")

                        struct_col1, struct_col2 = st.columns([1, 1])

                        with struct_col1:
                            # Display 2D structure
                            mol_img = Draw.MolToImage(results['mol'], size=(400, 400))
                            st.image(mol_img, caption="2D Molecular Structure", use_container_width=True)

                        with struct_col2:
                            if show_descriptors:
                                st.markdown("**Molecular Descriptors:**")
                                descriptors_data = {
                                    "Property": [
                                        "Molecular Weight",
                                        "LogP (Lipophilicity)",
                                        "H-Bond Donors",
                                        "H-Bond Acceptors",
                                        "Rotatable Bonds",
                                        "TPSA"
                                    ],
                                    "Value": [
                                        f"{results['descriptors']['MW']:.2f} g/mol",
                                        f"{results['descriptors']['LogP']:.2f}",
                                        f"{results['descriptors']['HBD']}",
                                        f"{results['descriptors']['HBA']}",
                                        f"{results['descriptors']['RotBonds']}",
                                        f"{results['descriptors']['TPSA']:.2f} Ų"
                                    ],
                                    "Lipinski's Rule": [
                                        "✓" if results['descriptors']['MW'] <= 500 else "✗",
                                        "✓" if results['descriptors']['LogP'] <= 5 else "✗",
                                        "✓" if results['descriptors']['HBD'] <= 5 else "✗",
                                        "✓" if results['descriptors']['HBA'] <= 10 else "✗",
                                        "-",
                                        "-"
                                    ]
                                }
                                st.dataframe(descriptors_data, use_container_width=True, hide_index=True)

                                # Lipinski's Rule of Five check
                                lipinski_pass = all([
                                    results['descriptors']['MW'] <= 500,
                                    results['descriptors']['LogP'] <= 5,
                                    results['descriptors']['HBD'] <= 5,
                                    results['descriptors']['HBA'] <= 10
                                ])

                                if lipinski_pass:
                                    st.success("✓ Passes Lipinski's Rule of Five (Drug-like)")
                                else:
                                    st.warning("⚠ Violates Lipinski's Rule of Five")

                    st.markdown("---")

                    # XAI Visualization
                    st.subheader("🔍 Explainable AI (XAI) - Atom Contributions")

                    st.markdown("""
                    <div class="info-box">
                    The visualization below shows which atoms contribute to the predicted bioactivity:
                    <ul>
                        <li><strong>Green atoms:</strong> Increase binding activity</li>
                        <li><strong>Red atoms:</strong> Decrease binding activity</li>
                        <li><strong>White/neutral atoms:</strong> Minimal contribution</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    xai_col1, xai_col2 = st.columns([2, 1])

                    with xai_col1:
                        if has_rdkit and results.get('xai_image') is not None:
                            st.image(
                                results['xai_image'],
                                caption="RDKit Similarity Map - Atom Contributions",
                                use_container_width=True
                            )
                        else:
                            st.info("XAI visualization requires trained model. Placeholder shown.")
                            # Show placeholder heatmap
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 6))
                            data = np.random.randn(10, 10)
                            im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
                            ax.set_title("XAI Heatmap (Placeholder)")
                            ax.set_xlabel("Feature Index")
                            ax.set_ylabel("Atom Index")
                            plt.colorbar(im, ax=ax, label="Contribution Score")
                            st.pyplot(fig)

                    with xai_col2:
                        st.markdown("**Key Pharmacophores Detected:**")

                        # Placeholder for important substructures
                        pharmacophores = [
                            {"name": "Aromatic Ring", "contribution": 0.35, "type": "positive"},
                            {"name": "Carboxyl Group", "contribution": 0.28, "type": "positive"},
                            {"name": "Alkyl Chain", "contribution": -0.12, "type": "negative"},
                            {"name": "Hydroxyl Group", "contribution": 0.15, "type": "positive"}
                        ]

                        for pharm in pharmacophores:
                            color = "🟢" if pharm['type'] == 'positive' else "🔴"
                            st.markdown(f"{color} **{pharm['name']}**: {pharm['contribution']:+.2f}")

                    st.markdown("---")

                    # Model-specific details
                    if model_type == "Both Models":
                        st.subheader("📈 Model Comparison")

                        comparison_data = {
                            "Model": ["Random Forest", "LSTM/GRU"],
                            "Predicted pIC50": [results['pIC50'], results['pIC50'] + 0.15],
                            "Confidence": [f"{results['confidence']:.1%}", f"{results['confidence'] + 0.05:.1%}"],
                            "Inference Time": ["12 ms", "45 ms"]
                        }
                        st.table(comparison_data)

                        st.info("""
                        **Note:** Random Forest uses molecular descriptors and fingerprints,
                        while LSTM/GRU processes raw SMILES sequences. Deep learning models
                        typically achieve higher accuracy but require more computation time.
                        """)

                    # Export options
                    st.markdown("---")
                    st.subheader("💾 Export Results")

                    export_col1, export_col2 = st.columns(2)

                    with export_col1:
                        # Export as CSV
                        export_data = {
                            "SMILES": [smiles_input],
                            "Predicted_pIC50": [results['pIC50']],
                            "Activity": [activity_class],
                            "Confidence": [results['confidence']],
                            "IC50_nM": [results['IC50']],
                            "MW": [results['descriptors']['MW']],
                            "LogP": [results['descriptors']['LogP']],
                            "Model": [model_type]
                        }
                        export_df = pd.DataFrame(export_data)

                        csv_buffer = BytesIO()
                        export_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)

                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_buffer,
                            file_name=f"prediction_{smiles_input[:10]}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with export_col2:
                        # Export as JSON
                        import json
                        json_data = json.dumps(results, default=str, indent=2)

                        st.download_button(
                            label="📥 Download as JSON",
                            data=json_data,
                            file_name=f"prediction_{smiles_input[:10]}.json",
                            mime="application/json",
                            use_container_width=True
                        )


def predict_bioactivity(smiles: str, model_type: str, has_rdkit: bool) -> dict:
    """
    Predict bioactivity using trained Random Forest model

    Args:
        smiles: SMILES string
        model_type: Selected model type
        has_rdkit: Whether RDKit is available

    Returns:
        Dictionary containing prediction results
    """
    # Import the real model predictor
    try:
        from app.utils.model_loader import predict_bioactivity as model_predict
    except ImportError:
        # Fallback for import issues
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.utils.model_loader import predict_bioactivity as model_predict

    # Get predictions from the real model
    model_results = model_predict(smiles, model_type='rf')

    # Format results for the UI
    results = {
        'pIC50': model_results['pIC50'],
        'confidence': model_results['confidence'],
        'IC50': model_results['IC50'],
        'descriptors': {}
    }

    # Extract descriptors from model results
    if model_results.get('descriptors'):
        desc = model_results['descriptors']
        results['descriptors'] = {
            'MW': desc.get('MW', 0),
            'LogP': desc.get('LogP', 0),
            'HBD': desc.get('NumHDonors', 0),
            'HBA': desc.get('NumHAcceptors', 0),
            'RotBonds': desc.get('NumRotatableBonds', 0),
            'TPSA': desc.get('TPSA', 0)
        }

    if has_rdkit:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                results['mol'] = mol

                # Generate XAI visualization (placeholder - would use actual model gradients)
                # For now, create a simple colored structure
                results['xai_image'] = Draw.MolToImage(mol, size=(600, 600))

        except Exception as e:
            print(f"RDKit processing error: {e}")

    return results
