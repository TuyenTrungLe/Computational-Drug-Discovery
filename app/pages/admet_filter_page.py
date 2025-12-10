"""
ADMET Safety Filter Page
Apply toxicity, solubility, and BBBP filters to compounds
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


def render():
    """Render the ADMET Filter page"""

    st.title("🎯 ADMET Safety Filter")
    st.markdown("Filter compounds based on ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties")

    st.markdown("---")

    # ADMET Overview
    with st.expander("ℹ️ About ADMET Properties", expanded=False):
        st.markdown("""
        ### ADMET Properties Explained:

        **ADMET** is crucial for drug development as it determines:

        - **Absorption**: How well the drug is absorbed into the bloodstream
        - **Distribution**: How the drug spreads throughout the body
        - **Metabolism**: How the drug is broken down
        - **Excretion**: How the drug is eliminated from the body
        - **Toxicity**: Potential harmful effects

        ### Properties We Evaluate:

        1. **Toxicity (Tox21)**: Binary classification of toxic vs non-toxic
        2. **Solubility (ESOL)**: Water solubility - critical for bioavailability
        3. **BBBP**: Blood-Brain Barrier Penetration - important for CNS drugs
        4. **Lipinski's Rule of Five**: Drug-likeness criteria
        5. **CYP450 Inhibition**: Drug-drug interaction potential

        ### Filtering Strategy:
        - **Predicted Active** (from bioactivity model)
        - **AND Predicted Non-Toxic** (from ADMET models)
        - = **Promising Drug Candidates**
        """)

    st.markdown("---")

    # Input section
    st.subheader("1️⃣ Input Compounds")

    input_method = st.radio(
        "Choose input method:",
        ["Use Previous Batch Results", "Upload New CSV", "Enter SMILES Manually"],
        horizontal=True
    )

    compounds_df = None

    if input_method == "Use Previous Batch Results":
        if 'batch_results' in st.session_state:
            compounds_df = st.session_state['batch_results'].copy()
            st.success(f"✓ Loaded {len(compounds_df)} compounds from previous batch analysis")
        else:
            st.warning("⚠️ No batch results found. Please run batch analysis first or use another input method.")

    elif input_method == "Upload New CSV":
        uploaded_file = st.file_uploader("Upload CSV with SMILES", type=['csv'])
        if uploaded_file:
            compounds_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(compounds_df)} compounds from CSV")

    else:  # Manual entry
        smiles_text = st.text_area(
            "Enter SMILES (one per line)",
            height=150,
            placeholder="CC(C)Cc1ccc(cc1)C(C)C(O)=O\nCC(=O)Oc1ccccc1C(=O)O"
        )
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
            compounds_df = pd.DataFrame({'SMILES': smiles_list})
            st.success(f"✓ Loaded {len(compounds_df)} compounds")

    if compounds_df is not None:
        st.dataframe(compounds_df.head(), use_container_width=True)

        st.markdown("---")

        # ADMET Filter Configuration
        st.subheader("2️⃣ Configure ADMET Filters")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            st.markdown("**Toxicity Filters**")

            enable_tox21 = st.checkbox("Filter by Tox21 (Toxicity)", value=True)
            if enable_tox21:
                tox21_threshold = st.slider(
                    "Max Toxicity Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Compounds with toxicity probability below this threshold will pass"
                )

            enable_clintox = st.checkbox("Filter by ClinTox (Clinical Toxicity)", value=False)
            if enable_clintox:
                clintox_threshold = st.slider(
                    "Max Clinical Toxicity Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )

            enable_mutagenicity = st.checkbox("Filter by Mutagenicity", value=False)

        with filter_col2:
            st.markdown("**Physicochemical Filters**")

            enable_solubility = st.checkbox("Filter by Solubility (ESOL)", value=True)
            if enable_solubility:
                solubility_range = st.slider(
                    "LogS Range (Solubility)",
                    min_value=-10.0,
                    max_value=0.0,
                    value=(-6.0, 0.0),
                    step=0.5,
                    help="Higher LogS = more soluble. Typical drugs: -6 to -2"
                )

            enable_bbbp = st.checkbox("Filter by BBBP (Blood-Brain Barrier)", value=False)
            if enable_bbbp:
                bbbp_requirement = st.radio(
                    "BBBP Requirement",
                    ["Must penetrate BBB", "Must NOT penetrate BBB", "No preference"],
                    help="For CNS drugs, choose 'Must penetrate'. For peripheral drugs, choose 'Must NOT'."
                )

            enable_lipinski = st.checkbox("Apply Lipinski's Rule of Five", value=True)

        st.markdown("---")

        # Run ADMET filtering
        st.subheader("3️⃣ Run ADMET Prediction")

        admet_col1, admet_col2 = st.columns([1, 3])

        with admet_col1:
            run_admet = st.button("🧪 Run ADMET Analysis", type="primary", use_container_width=True)

        with admet_col2:
            st.info("ADMET prediction will evaluate all selected properties and filter compounds accordingly")

        if run_admet:
            with st.spinner("Running ADMET predictions with trained models..."):
                # Use actual ADMET model predictions
                admet_results = predict_admet_properties(compounds_df)

                if admet_results is not None:
                    # Apply filters
                    filtered_results = apply_admet_filters(
                        admet_results,
                        enable_tox21=enable_tox21,
                        tox21_threshold=tox21_threshold if enable_tox21 else None,
                        enable_solubility=enable_solubility,
                        solubility_range=solubility_range if enable_solubility else None,
                        enable_bbbp=enable_bbbp,
                        bbbp_requirement=bbbp_requirement if enable_bbbp else None,
                        enable_lipinski=enable_lipinski
                    )

                    # Store in session state
                    st.session_state['admet_results'] = admet_results
                    st.session_state['filtered_admet'] = filtered_results

                    st.success(f"✓ ADMET analysis completed! Analyzed {len(admet_results)} compounds.")
                else:
                    st.error("Failed to run ADMET predictions. Please check the error messages above.")

        # Display results
        if 'admet_results' in st.session_state:
            admet_results = st.session_state['admet_results']
            filtered_results = st.session_state['filtered_admet']

            st.markdown("---")
            st.subheader("4️⃣ Results Summary")

            # Summary metrics
            metric_cols = st.columns(5)

            with metric_cols[0]:
                st.metric("Total Compounds", len(admet_results))

            with metric_cols[1]:
                non_toxic = len(admet_results[admet_results['tox21_pass'] == True])
                st.metric("Non-Toxic", non_toxic,
                         delta=f"{non_toxic/len(admet_results)*100:.1f}%")

            with metric_cols[2]:
                soluble = len(admet_results[admet_results['solubility_pass'] == True])
                st.metric("Good Solubility", soluble,
                         delta=f"{soluble/len(admet_results)*100:.1f}%")

            with metric_cols[3]:
                lipinski_pass = len(admet_results[admet_results['lipinski_pass'] == True])
                st.metric("Lipinski Pass", lipinski_pass,
                         delta=f"{lipinski_pass/len(admet_results)*100:.1f}%")

            with metric_cols[4]:
                final_pass = len(filtered_results)
                st.metric("Final Candidates", final_pass,
                         delta=f"{final_pass/len(admet_results)*100:.1f}%")

            st.markdown("---")

            # Detailed results
            st.subheader("📊 ADMET Predictions")

            results_tab1, results_tab2, results_tab3 = st.tabs([
                "All Compounds",
                "Passed Filters",
                "Failed Compounds"
            ])

            with results_tab1:
                st.dataframe(admet_results, use_container_width=True, height=400)

            with results_tab2:
                st.success(f"✓ {len(filtered_results)} compounds passed all filters")
                st.dataframe(filtered_results, use_container_width=True, height=400)

                if len(filtered_results) > 0:
                    # Show top candidates
                    st.markdown("**🏆 Top 10 Candidates (by overall score)**")
                    top_candidates = filtered_results.nlargest(min(10, len(filtered_results)), 'overall_score')
                    st.dataframe(top_candidates, use_container_width=True)

            with results_tab3:
                failed_df = admet_results[~admet_results['SMILES'].isin(filtered_results['SMILES'])]
                st.warning(f"⚠️ {len(failed_df)} compounds failed one or more filters")
                st.dataframe(failed_df, use_container_width=True, height=400)

            st.markdown("---")

            # Visualizations
            st.subheader("📈 ADMET Analysis Visualizations")

            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "Filter Pass Rates",
                "Property Distributions",
                "Risk Matrix"
            ])

            with viz_tab1:
                import matplotlib.pyplot as plt

                # Bar chart of pass rates
                pass_rates = {
                    'Tox21\n(Non-toxic)': admet_results['tox21_pass'].sum(),
                    'Solubility\n(Good)': admet_results['solubility_pass'].sum(),
                    'Lipinski\n(Drug-like)': admet_results['lipinski_pass'].sum(),
                    'BBBP\n(Pass)': admet_results['bbbp_pass'].sum() if 'bbbp_pass' in admet_results else 0,
                    'All Filters': len(filtered_results)
                }

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(pass_rates.keys(), pass_rates.values(), color=['#06A77D', '#2E86AB', '#A23B72', '#F18F01', '#D00000'])
                ax.set_ylabel('Number of Compounds')
                ax.set_title('ADMET Filter Pass Rates')
                ax.axhline(y=len(admet_results), color='gray', linestyle='--', alpha=0.5, label='Total')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}\n({height/len(admet_results)*100:.1f}%)',
                           ha='center', va='bottom')

                ax.legend()
                st.pyplot(fig)

            with viz_tab2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Toxicity distribution
                axes[0, 0].hist(admet_results['tox21_prob'], bins=20, color='#D00000', alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
                axes[0, 0].set_xlabel('Toxicity Probability')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Tox21 Distribution')
                axes[0, 0].legend()

                # Solubility distribution
                axes[0, 1].hist(admet_results['logS'], bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
                axes[0, 1].axvspan(-6, -2, alpha=0.2, color='green', label='Drug-like range')
                axes[0, 1].set_xlabel('LogS (Solubility)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Solubility Distribution')
                axes[0, 1].legend()

                # Molecular Weight distribution
                axes[1, 0].hist(admet_results['MW'], bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
                axes[1, 0].axvline(x=500, color='red', linestyle='--', label='Lipinski cutoff')
                axes[1, 0].set_xlabel('Molecular Weight (g/mol)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Molecular Weight Distribution')
                axes[1, 0].legend()

                # LogP distribution
                axes[1, 1].hist(admet_results['LogP'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(x=5, color='red', linestyle='--', label='Lipinski cutoff')
                axes[1, 1].set_xlabel('LogP (Lipophilicity)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('LogP Distribution')
                axes[1, 1].legend()

                plt.tight_layout()
                st.pyplot(fig)

            with viz_tab3:
                # Risk matrix: Toxicity vs Solubility
                fig, ax = plt.subplots(figsize=(10, 8))

                colors = ['green' if row['tox21_pass'] and row['solubility_pass'] else 'red'
                         for _, row in admet_results.iterrows()]

                scatter = ax.scatter(
                    admet_results['logS'],
                    admet_results['tox21_prob'],
                    c=colors,
                    alpha=0.6,
                    s=100,
                    edgecolors='black'
                )

                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axvspan(-6, -2, alpha=0.1, color='blue')

                ax.set_xlabel('LogS (Solubility)')
                ax.set_ylabel('Toxicity Probability')
                ax.set_title('Risk Matrix: Toxicity vs Solubility')

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', label='Pass both filters'),
                    Patch(facecolor='red', label='Fail one or more filters')
                ]
                ax.legend(handles=legend_elements)

                st.pyplot(fig)

            st.markdown("---")

            # Export section
            st.subheader("💾 Export Filtered Candidates")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # Export passed compounds
                csv_buffer = BytesIO()
                filtered_results.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="📥 Download Passed Compounds",
                    data=csv_buffer,
                    file_name="admet_passed_compounds.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with export_col2:
                # Export all ADMET results
                csv_buffer_all = BytesIO()
                admet_results.to_csv(csv_buffer_all, index=False)
                csv_buffer_all.seek(0)

                st.download_button(
                    label="📥 Download All ADMET Results",
                    data=csv_buffer_all,
                    file_name="admet_all_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with export_col3:
                # Export failed compounds
                failed_csv = BytesIO()
                failed_df.to_csv(failed_csv, index=False)
                failed_csv.seek(0)

                st.download_button(
                    label="📥 Download Failed Compounds",
                    data=failed_csv,
                    file_name="admet_failed_compounds.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    else:
        st.info("👆 Please select an input method and provide compounds to begin ADMET filtering")


def predict_admet_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict ADMET properties using trained models
    """
    try:
        # Import ADMET predictor
        from utils.admet_predictor import predict_batch_df, is_admet_available

        if not is_admet_available():
            st.error("⚠️ ADMET models not available. Please ensure models are trained and saved in models/admet_models/")
            return None

        # Use actual trained models
        results = predict_batch_df(df, smiles_col='SMILES')
        return results

    except Exception as e:
        st.error(f"Error during ADMET prediction: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def apply_admet_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Apply ADMET filters to dataframe"""
    filtered = df.copy()

    if filters.get('enable_tox21'):
        threshold = filters.get('tox21_threshold', 0.5)
        filtered = filtered[filtered['tox21_prob'] < threshold]

    if filters.get('enable_solubility'):
        sol_range = filters.get('solubility_range', (-6, 0))
        filtered = filtered[
            (filtered['logS'] >= sol_range[0]) &
            (filtered['logS'] <= sol_range[1])
        ]

    if filters.get('enable_lipinski'):
        filtered = filtered[filtered['lipinski_pass'] == True]

    return filtered
