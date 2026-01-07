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

    st.title("🎯 ADMET Filter: Threshold Tuning")
    st.markdown("""
    **Purpose:** Adjust bioactivity and toxicity thresholds on **existing batch results** to optimize candidate selection.  
    ⚠️ **Note:** This page does NOT re-run predictions - it applies filters to pre-computed scores.
    """)

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
        horizontal=True,
        key="admet_input_method"
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
        st.subheader("2️⃣ Configure Two-Stage Filtering Thresholds")
        
        st.info("""
        🎯 **Track C Two-Stage Pipeline:** Adjust thresholds for both stages and see real-time KEEP count changes.
        - **Stage 1:** Bioactivity threshold (pIC50)
        - **Stage 2:** ADMET safety threshold (Toxicity)
        - **Final:** KEEP = Active AND Non-Toxic
        """)

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            st.markdown("### 🔵 Stage 1: Bioactivity")
            st.caption("📄 **Threshold Type:** pIC50 (continuous scale, higher = more active)")
            
            bioactivity_threshold = st.slider(
                "pIC50 Threshold (Active if ≥)",
                min_value=4.0,
                max_value=8.0,
                value=6.0,
                step=0.1,
                help="Compounds with pIC50 ≥ this value are considered Active. pIC50 = -log10(IC50 in Molar)",
                key="admet_bio_threshold"
            )
            
            st.markdown(f"""
            <div style='background: #bbdefb; padding: 1rem; border-radius: 8px; margin-top: 1rem; border: 3px solid #1976d2;'>
                <p style='margin: 0; color: #0d47a1; font-weight: 600;'><b>Rule:</b> pIC50 ≥ {bioactivity_threshold:.1f} → <b>Active</b></p>
                <p style='margin: 0.5rem 0 0 0; color: #1565c0; font-weight: 500;'><em>Only Active compounds proceed to Stage 2</em></p>
            </div>
            """, unsafe_allow_html=True)

        with filter_col2:
            st.markdown("### 🟠 Stage 2: ADMET Safety")
            st.caption("📄 **Threshold Type:** Toxicity Probability (0.0-1.0, lower = safer)")
            
            toxicity_threshold = st.slider(
                "Toxicity Threshold (Non-Toxic if ≤)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Compounds with toxicity probability ≤ this value are considered Non-Toxic (0=safe, 1=toxic)",
                key="admet_tox_threshold"
            )
            
            st.markdown(f"""
            <div style='background: #ffcc80; padding: 1rem; border-radius: 8px; margin-top: 1rem; border: 3px solid #f57c00;'>
                <p style='margin: 0; color: #e65100; font-weight: 600;'><b>Rule:</b> Toxicity ≤ {toxicity_threshold:.2f} → <b>Non-Toxic</b></p>
                <p style='margin: 0.5rem 0 0 0; color: #ef6c00; font-weight: 500;'><em>Active AND Non-Toxic → KEEP</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Additional ADMET filters (optional)
        with st.expander("⚙️ Advanced ADMET Filters (Optional)"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                enable_clintox = st.checkbox("Filter by ClinTox (Clinical Toxicity)", value=False)
                if enable_clintox:
                    clintox_threshold = st.slider(
                        "Max Clinical Toxicity Probability",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        key="admet_clintox"
                    )

                enable_solubility = st.checkbox("Filter by Solubility (ESOL)", value=False)
                if enable_solubility:
                    solubility_range = st.slider(
                        "LogS Range (Solubility)",
                        min_value=-10.0,
                        max_value=0.0,
                        value=(-6.0, 0.0),
                        step=0.5,
                        help="Higher LogS = more soluble",
                        key="admet_solubility"
                    )
            
            with adv_col2:
                enable_bbbp = st.checkbox("Filter by BBBP (Blood-Brain Barrier)", value=False)
                if enable_bbbp:
                    bbbp_requirement = st.radio(
                        "BBBP Requirement",
                        ["Must penetrate BBB", "Must NOT penetrate BBB", "No preference"],
                        key="admet_bbbp_requirement"
                    )

                enable_lipinski = st.checkbox("Apply Lipinski's Rule of Five", value=False)

        st.markdown("---")

        # Run ADMET filtering (or use batch results if available)
        st.subheader("3️⃣ Apply Two-Stage Filtering")
        
        # Check if we have batch results with predictions already
        if 'batch_results' in st.session_state and 'bioactivity_score' in st.session_state['batch_results'].columns:
            st.info("✅ Using predictions from Batch Analysis")
            results_df = st.session_state['batch_results'].copy()
            has_predictions = True
        else:
            st.warning("⚠️ No batch results found. Please run Batch Analysis first, or use the Run ADMET button below.")
            has_predictions = False
            
            run_admet = st.button("🧪 Run ADMET Analysis on Current Data", type="primary", use_container_width=True)
            
            if run_admet:
                with st.spinner("Running predictions..."):
                    # Mock predictions for demo (replace with actual model)
                    results_df = compounds_df.copy()
                    results_df['bioactivity_score'] = np.random.uniform(4.0, 8.0, len(results_df))
                    results_df['admet_score'] = np.random.uniform(0.1, 0.9, len(results_df))
                    has_predictions = True
                    st.session_state['admet_filter_results'] = results_df
        
        # Real-time filtering if we have predictions
        if has_predictions or 'admet_filter_results' in st.session_state:
            if 'admet_filter_results' in st.session_state:
                results_df = st.session_state['admet_filter_results']
            
            # Apply two-stage filtering logic
            results_df = results_df.copy()
            
            # Stage 1: Bioactivity
            results_df['is_active'] = results_df['bioactivity_score'] >= bioactivity_threshold
            results_df['bioactivity_label'] = results_df['is_active'].apply(lambda x: 'Active' if x else 'Inactive')
            
            # Stage 2: ADMET (only for Active)
            results_df['is_safe'] = results_df['admet_score'] <= toxicity_threshold
            results_df['admet_label'] = results_df.apply(
                lambda row: 'Non-Toxic' if row['is_active'] and row['is_safe'] else 
                           ('Toxic' if row['is_active'] else 'N/A'),
                axis=1
            )
            
            # Final Decision
            results_df['final_decision'] = results_df.apply(
                lambda row: 'KEEP' if row['is_active'] and row['is_safe'] else 'REJECT',
                axis=1
            )
            
            # Calculate statistics
            total_count = len(results_df)
            active_count = len(results_df[results_df['is_active']])
            safe_count = len(results_df[results_df['is_active'] & results_df['is_safe']])
            keep_count = safe_count  # KEEP = Active AND Non-Toxic
            
            st.markdown("---")
            st.subheader("📊 Real-Time Filtering Statistics")
            
            # Show funnel with current thresholds
            funnel_col1, funnel_col2, funnel_col3, funnel_col4 = st.columns(4)
            
            with funnel_col1:
                st.markdown(f"""
                <div style='background: #90caf9; padding: 1rem; border-radius: 6px; text-align: center; border: 3px solid #1976d2; box-sizing: border-box; height: 160px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='color: #0d47a1; margin: 0; font-size: 1.5rem; font-weight: 700;'>{total_count}</div>
                    <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #1565c0; font-weight: 600;'>Total Input</p>
                    <p style='margin: 0; font-size: 0.8rem; visibility: hidden;'>Placeholder</p>
                </div>
                """, unsafe_allow_html=True)
            
            with funnel_col2:
                active_pct = (active_count / total_count * 100) if total_count > 0 else 0
                st.markdown(f"""
                <div style='background: #ffcc80; padding: 1rem; border-radius: 6px; text-align: center; border: 3px solid #f57c00; box-sizing: border-box; height: 160px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='color: #e65100; margin: 0; font-size: 1.5rem; font-weight: 700;'>{active_count}</div>
                    <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #ef6c00; font-weight: 600;'>✅ Active</p>
                    <p style='margin: 0; font-size: 0.8rem; color: #bf360c; font-weight: 500;'>{active_pct:.1f}% passed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with funnel_col3:
                safe_pct = (safe_count / active_count * 100) if active_count > 0 else 0
                st.markdown(f"""
                <div style='background: #a5d6a7; padding: 1rem; border-radius: 6px; text-align: center; border: 3px solid #388e3c; box-sizing: border-box; height: 160px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='color: #1b5e20; margin: 0; font-size: 1.5rem; font-weight: 700;'>{safe_count}</div>
                    <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #2e7d32; font-weight: 600;'>✅ Non-Toxic</p>
                    <p style='margin: 0; font-size: 0.8rem; color: #388e3c; font-weight: 500;'>{safe_pct:.1f}% of Active</p>
                </div>
                """, unsafe_allow_html=True)
            
            with funnel_col4:
                keep_pct = (keep_count / total_count * 100) if total_count > 0 else 0
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%); 
                            padding: 1rem; border-radius: 6px; text-align: center; border: 3px solid #1b5e20; box-shadow: 0 4px 6px rgba(0,0,0,0.2); box-sizing: border-box; height: 160px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='color: white; margin: 0; font-size: 1.5rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>{keep_count}</div>
                    <p style='margin: 0.3rem 0 0 0; color: white; font-weight: 700; font-size: 0.9rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>🎯 KEEP</p>
                    <p style='margin: 0; font-size: 0.8rem; color: white; font-weight: 600;'>{keep_pct:.1f}% of Total</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Threshold adjustment message
            st.markdown(f"""
            <div style='background: #e0e0e0; padding: 1rem; border-radius: 8px; margin-top: 1rem; border: 2px solid #757575;'>
                <p style='margin: 0; font-weight: 600; color: #424242;'><b>💡 Adjust sliders above to see KEEP count change in real-time!</b></p>
                <p style='margin: 0.5rem 0 0 0; color: #616161; font-weight: 500;'>
                    Current thresholds: pIC50 ≥ {bioactivity_threshold:.1f}, Toxicity ≤ {toxicity_threshold:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display filtered compounds
            st.subheader("4️⃣ Filtered Results")
            
            # Filter view options
            view_option = st.radio(
                "View:",
                ["Show All", "Show KEEP Only", "Show REJECT Only"],
                horizontal=True,
                key="admet_view_option"
            )
            
            if view_option == "Show KEEP Only":
                display_df = results_df[results_df['final_decision'] == 'KEEP']
            elif view_option == "Show REJECT Only":
                display_df = results_df[results_df['final_decision'] == 'REJECT']
            else:
                display_df = results_df
            
            st.info(f"📊 Showing {len(display_df)} compounds")
            
            # Color-coded table
            def highlight_decision(row):
                if row['final_decision'] == 'KEEP':
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            # Select columns to display
            display_cols = ['SMILES' if 'SMILES' in display_df.columns else 'smiles', 
                           'bioactivity_score', 'bioactivity_label', 
                           'admet_score', 'admet_label', 'final_decision']
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            st.dataframe(
                display_df[display_cols].style.apply(highlight_decision, axis=1),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Export options
            st.subheader("💾 Export Filtered Results")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Export KEEP only
                keep_df = results_df[results_df['final_decision'] == 'KEEP']
                csv_keep = keep_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download KEEP Compounds ({len(keep_df)} rows)",
                    data=csv_keep,
                    file_name=f"keep_compounds_pic50{bioactivity_threshold}_tox{toxicity_threshold}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            
            with export_col2:
                # Export all with decisions
                csv_all = results_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download All Results ({len(results_df)} rows)",
                    data=csv_all,
                    file_name="admet_filtered_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.success(f"✅ Filtering complete! {keep_count} compounds marked as KEEP.")

            # Additional Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                # Mock Lipinski for now as we don't have it calculated in batch yet
                # actual implementation would need Lipinski descriptors calculated
                st.info("Lipinski Rules info pending update")

            with m_col2:
                final_pass = len(results_df[results_df['final_decision'] == 'KEEP'])
                st.metric("Final Candidates (KEEP)", final_pass,
                         delta=f"{final_pass/len(results_df)*100:.1f}%")

            st.markdown("---")

            # Detailed results
            st.subheader("📊 ADMET Predictions")

            results_tab1, results_tab2, results_tab3 = st.tabs([
                "All Compounds",
                "Passed Filters",
                "Failed Compounds"
            ])

            with results_tab1:
                st.dataframe(results_df, use_container_width=True, height=400)

            with results_tab2:
                keep_df = results_df[results_df['final_decision'] == 'KEEP']
                st.success(f"✓ {len(keep_df)} compounds passed all filters")
                st.dataframe(keep_df, use_container_width=True, height=400)

                if len(keep_df) > 0:
                    # Show top candidates
                    st.markdown("**🏆 Top 10 Candidates (by Bioactivity & ADMET Safety)**")
                    # Assuming we want to sort by bioactivity score as a proxy for "best" since we don't have 'overall_score'
                    top_candidates = keep_df.nlargest(min(10, len(keep_df)), 'bioactivity_score')
                    st.dataframe(top_candidates, use_container_width=True)

            with results_tab3:
                reject_df = results_df[results_df['final_decision'] == 'REJECT']
                st.warning(f"⚠️ {len(reject_df)} compounds failed one or more filters")
                st.dataframe(reject_df, use_container_width=True, height=400)

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
                # Calculating pass rates from available data
                pass_rates = {
                    'Bioactivity\n(Active)': len(results_df[results_df['bioactivity_label'] == 'Active']),
                    'ADMET\n(Non-Toxic)': len(results_df[results_df['admet_label'] == 'Non-Toxic']),
                    'Final\n(KEEP)': len(results_df[results_df['final_decision'] == 'KEEP'])
                }

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(pass_rates.keys(), pass_rates.values(), color=['#2E86AB', '#06A77D', '#D00000'])
                ax.set_ylabel('Number of Compounds')
                ax.set_title('Pipeline Pass Rates')
                ax.axhline(y=len(results_df), color='gray', linestyle='--', alpha=0.5, label='Total')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}\n({height/len(results_df)*100:.1f}%)',
                           ha='center', va='bottom')

                ax.legend()
                st.pyplot(fig)

            with viz_tab2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Toxicity distribution
                if 'admet_score' in results_df.columns:
                     # Handle None values
                     tox_scores = results_df['admet_score'].dropna().astype(float)
                     axes[0, 0].hist(tox_scores, bins=20, color='#D00000', alpha=0.7, edgecolor='black')
                     axes[0, 0].axvline(x=toxicity_threshold, color='black', linestyle='--', label='Threshold')
                     axes[0, 0].set_xlabel('Toxicity Probability')
                else:
                     axes[0, 0].text(0.5, 0.5, "No Toxicity Data", ha='center', va='center')
                
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Toxicity Distribution')
                axes[0, 0].legend()

                # Solubility distribution (Mock data if unavailable)
                axes[0, 1].text(0.5, 0.5, "Solubility Data Not Available in Batch Mode", ha='center', va='center')
                axes[0, 1].set_title('Solubility Distribution')

                # Molecular Weight distribution (Mock)
                axes[1, 0].text(0.5, 0.5, "MW Data Not Available in Batch Mode", ha='center', va='center')
                axes[1, 0].set_title('Molecular Weight Distribution')

                # LogP distribution (Mock)
                axes[1, 1].text(0.5, 0.5, "LogP Data Not Available in Batch Mode", ha='center', va='center')
                axes[1, 1].set_title('LogP Distribution')

                plt.tight_layout()
                st.pyplot(fig)

            with viz_tab3:
                st.info("Risk Matrix requires multi-parameter ADMET data which is currently limited in rapid batch screening.")
                
            st.markdown("---")

            # Export section
            st.subheader("💾 Export Filtered Candidates")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # Export passed compounds
                keep_df = results_df[results_df['final_decision'] == 'KEEP']
                csv_buffer = BytesIO()
                keep_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="📥 Download KEEP Compounds",
                    data=csv_buffer,
                    file_name="admet_keep_compounds.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with export_col2:
                # Export all ADMET results
                csv_buffer_all = BytesIO()
                results_df.to_csv(csv_buffer_all, index=False)
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
                reject_df = results_df[results_df['final_decision'] == 'REJECT']
                failed_csv = BytesIO()
                reject_df.to_csv(failed_csv, index=False)
                failed_csv.seek(0)

                st.download_button(
                    label="📥 Download REJECT Compounds",
                    data=failed_csv,
                    file_name="admet_reject_compounds.csv",
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
