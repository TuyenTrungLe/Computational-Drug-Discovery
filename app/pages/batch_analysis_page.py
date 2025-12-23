"""
Batch Analysis Page
Upload and process multiple compounds from CSV files
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
    """Render the Batch Analysis page"""

    st.title("📊 Batch Analysis")
    st.markdown("Upload a CSV file containing multiple compounds for batch prediction")

    st.markdown("---")



    # Instructions
    with st.expander("📖 Instructions", expanded=True):
        st.markdown("""
        ### How to use Batch Analysis:

        1. **Prepare your CSV file** with at least a 'SMILES' column
        2. **Upload the file** using the file uploader below
        3. **Select prediction settings** (model type, filters, etc.)
        4. **Run predictions** on all compounds
        5. **Review results** with interactive filtering and sorting
        6. **Download filtered results** as CSV

        ### Required CSV Format:
        ```
        SMILES,compound_id,name
        CC(C)Cc1ccc(cc1)C(C)C(O)=O,CHEM001,Ibuprofen
        CC(=O)Oc1ccccc1C(=O)O,CHEM002,Aspirin
        ...
        ```

        **Minimum requirement:** A column named 'SMILES' or 'smiles'
        """)

    st.markdown("---")

    # File upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1️⃣ Upload Data")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing SMILES strings"
        )

        # Sample data download
        if st.button("📥 Download Sample CSV Template"):
            sample_data = {
                'SMILES': [
                    'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
                    'CC(=O)Oc1ccccc1C(=O)O',
                    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                    'NCCc1ccc(O)c(O)c1',
                    'CC(=O)OCC[N+](C)(C)C'
                ],
                'compound_id': ['CHEM001', 'CHEM002', 'CHEM003', 'CHEM004', 'CHEM005'],
                'name': ['Ibuprofen', 'Aspirin', 'Caffeine', 'Dopamine', 'Acetylcholine']
            }
            sample_df = pd.DataFrame(sample_data)

            csv_buffer = BytesIO()
            sample_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="Download",
                data=csv_buffer,
                file_name="sample_compounds.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("2️⃣ Settings")

        model_type = st.selectbox(
            "Prediction Model",
            ["Random Forest (Fast)", "LSTM/GRU (Accurate)", "Ensemble (Both)"],
            help="Choose prediction model"
        )

        batch_size = st.number_input(
            "Batch Size",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of compounds to process at once"
        )

    st.markdown("---")

    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✓ File uploaded successfully: {uploaded_file.name}")
            st.info(f"📊 Dataset contains {len(df)} compounds")

            # Detect SMILES column
            smiles_col = None
            for col in df.columns:
                if col.lower() in ['smiles', 'smile', 'canonical_smiles']:
                    smiles_col = col
                    break

            if smiles_col is None:
                st.error("❌ No SMILES column found. Please ensure your CSV has a 'SMILES' column.")
                return

            st.markdown("---")

            # Show preview
            st.subheader("3️⃣ Data Preview")

            preview_tab1, preview_tab2 = st.tabs(["📋 First 10 Rows", "📈 Statistics"])

            with preview_tab1:
                st.dataframe(df.head(10), use_container_width=True)

            with preview_tab2:
                col_stats1, col_stats2, col_stats3 = st.columns(3)

                with col_stats1:
                    st.metric("Total Compounds", len(df))

                with col_stats2:
                    unique_smiles = df[smiles_col].nunique()
                    st.metric("Unique SMILES", unique_smiles)

                with col_stats3:
                    duplicates = len(df) - unique_smiles
                    st.metric("Duplicates", duplicates)

                # Check for missing values
                missing_smiles = df[smiles_col].isna().sum()
                if missing_smiles > 0:
                    st.warning(f"⚠️ Found {missing_smiles} missing SMILES values")

            st.markdown("---")

            # Run prediction button
            st.subheader("4️⃣ Run Predictions")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

            with col_btn1:
                run_prediction = st.button(
                    "🚀 Run Batch Prediction",
                    type="primary",
                    use_container_width=True
                )

            with col_btn2:
                validate_only = st.button(
                    "✓ Validate SMILES Only",
                    use_container_width=True
                )

            with col_btn3:
                clear_results = st.button(
                    "🗑️ Clear Results",
                    use_container_width=True
                )

            # Clear results handler
            if clear_results:
                for key in ['batch_stage1_results', 'batch_results', 'admet_completed', 'active_with_admet']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

            # Process predictions (Stage 1)
            if run_prediction:
                st.markdown("---")
                st.subheader("5️⃣ Prediction Results")

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                results_list = []

                # Process in batches
                total_compounds = len(df)
                num_batches = (total_compounds + batch_size - 1) // batch_size

                # Import the real model predictor
                from app.utils.model_loader import predict_batch as batch_predict

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_compounds)

                    batch_df = df.iloc[start_idx:end_idx]

                    status_text.text(f"Processing batch {batch_idx + 1}/{num_batches}...")

                    # Get SMILES from batch
                    batch_smiles = batch_df[smiles_col].tolist()

                    # Get real predictions using the trained model
                    batch_results_df = batch_predict(batch_smiles, model_type='rf', batch_size=len(batch_smiles))

                    # Merge with original data
                    for idx, (orig_idx, row) in enumerate(batch_df.iterrows()):
                        result = batch_results_df.iloc[idx].to_dict()

                        # Add original columns (compound_id, name, etc.)
                        for col in df.columns:
                            if col != smiles_col:
                                result[col] = row[col]

                        results_list.append(result)

                    # Update progress
                    progress = (batch_idx + 1) / num_batches
                    progress_bar.progress(progress)

                status_text.text("✓ Stage 1 (Bioactivity) completed!")

                # Create results dataframe
                results_df = pd.DataFrame(results_list)
                
                # Add bioactivity classification (Track C format)
                bioactivity_threshold = 6.0
                results_df['bioactivity_score'] = results_df['predicted_pIC50']
                results_df['bioactivity_label'] = results_df['predicted_pIC50'].apply(
                    lambda x: 'Active' if x >= bioactivity_threshold else 'Inactive'
                )
                results_df['bioactivity_threshold'] = bioactivity_threshold
                
                # Save to session state
                st.session_state['batch_stage1_results'] = results_df
                
                # Clear previous downstream results if any
                if 'batch_results' in st.session_state:
                    del st.session_state['batch_results']
                if 'admet_completed' in st.session_state:
                    del st.session_state['admet_completed']
                
                st.success(f"✓ Stage 1 completed: {len(results_df)} compounds processed")
                st.rerun()

            # Render Results (from Session State)
            if 'batch_stage1_results' in st.session_state:
                results_df = st.session_state['batch_stage1_results']
                
                # ========== TWO-STAGE FILTERING ==========
                st.markdown("---")
                st.subheader("🔵 Stage 1: Bioactivity Results")
                
                active_df = results_df[results_df['bioactivity_label'] == 'Active']
                inactive_df = results_df[results_df['bioactivity_label'] == 'Inactive']
                
                col_stage1_1, col_stage1_2 = st.columns(2)
                
                with col_stage1_1:
                    st.metric(
                        "Total Compounds",
                        len(results_df),
                        help="Total compounds in batch"
                    )
                    st.metric(
                        "✅ Active (Passed Stage 1)",
                        len(active_df),
                        delta=f"{len(active_df)/len(results_df)*100:.1f}%",
                        help="Compounds with pIC50 ≥ 6.0"
                    )
                
                with col_stage1_2:
                    st.metric(
                        "❌ Inactive (Rejected)",
                        len(inactive_df),
                        delta=f"-{len(inactive_df)/len(results_df)*100:.1f}%",
                        delta_color="inverse",
                        help="Compounds with pIC50 < 6.0"
                    )
                    st.metric(
                        "Average pIC50",
                        f"{results_df['bioactivity_score'].mean():.2f}"
                    )
                
                # Gate Logic Visualization
                st.markdown("""
                <div style='background: #bbdefb; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #1976d2; margin: 1rem 0; border: 3px solid #1565c0;'>
                    <p style='margin: 0; color: #0d47a1; font-weight: 600;'><b>⚡ Gate Logic:</b> Only <b>{} Active compounds</b> will proceed to Stage 2 (ADMET).</p>
                    <p style='margin: 0.5rem 0 0 0; color: #1565c0; font-weight: 500;'><em>{} Inactive compounds are automatically marked as REJECT.</em></p>
                </div>
                """.format(len(active_df), len(inactive_df)), unsafe_allow_html=True)
                
                # ========== STAGE 2: ADMET (Only for Active) ==========
                if len(active_df) > 0:
                    st.markdown("---")
                    st.subheader("🟠 Stage 2: ADMET Prediction (Active Compounds Only)")
                    
                    run_admet = st.button(
                        f"🛡️ Run ADMET on {len(active_df)} Active Compounds",
                        type="primary",
                        use_container_width=True,
                        key="run_admet_btn"
                    )
                    
                    if run_admet or 'admet_completed' in st.session_state:
                        # If running for the first time or re-rendering
                        if run_admet:  # Force run if button clicked
                            with st.spinner(f"🔄 Running ADMET predictions on {len(active_df)} Active compounds..."):
                                try:
                                    from app.utils.admet_predictor import ADMETPredictor
                                    admet_predictor = ADMETPredictor()
                                    
                                    # Predict ADMET for active compounds
                                    admet_results = []
                                    error_details = []
                                    
                                    # Detect correct SMILES column
                                    actual_smiles_col = 'smiles'
                                    if 'SMILES' in active_df.columns:
                                        actual_smiles_col = 'SMILES'
                                    elif 'smiles' in active_df.columns:
                                        actual_smiles_col = 'smiles'
                                    else:
                                        # Fallback search
                                        for col in active_df.columns:
                                            if col.lower() == 'smiles':
                                                actual_smiles_col = col
                                                break
                                    
                                    for idx, row in active_df.iterrows():
                                        try:
                                            # Validate SMILES length
                                            val = row.get(actual_smiles_col, None)
                                            if val is None:
                                                 toxicity_prob = None
                                                 error_details.append(f"Missing column: {actual_smiles_col}")
                                            elif len(str(val)) > 1000:
                                                toxicity_prob = None  # Skip long molecules
                                                error_details.append("SMILES too long")
                                            else:
                                                admet_res = admet_predictor.predict(val)
                                                
                                                # Check if we got a valid probability
                                                if isinstance(admet_res, dict) and 'toxicity' in admet_res:
                                                    toxicity_prob = admet_res['toxicity'].get('probability')
                                                    error_details.append(None)
                                                else:
                                                    toxicity_prob = None
                                                    error_details.append(f"Invalid result format: {admet_res}")
                                        except Exception as e:
                                            error_msg = f"{type(e).__name__}: {str(e)}"
                                            # print(f"Error predicting for {row['smiles']}: {e}")
                                            toxicity_prob = None  # Error fallback
                                            error_details.append(error_msg)
                                            
                                        admet_results.append(toxicity_prob)
                                    
                                    # Add ADMET results
                                    active_df = active_df.copy()
                                    active_df['admet_score'] = admet_results
                                    active_df['error_details'] = error_details
                                    active_df['admet_threshold'] = 0.5
                                    
                                    # Update label logic to handle None
                                    def get_admet_label(row):
                                        if row['admet_score'] is None or pd.isna(row['admet_score']):
                                            return f"N/A ({row['error_details']})" if row['error_details'] else "N/A (Error)"
                                        return 'Non-Toxic' if row['admet_score'] <= 0.5 else 'Toxic'
                                        
                                    active_df['admet_label'] = active_df.apply(get_admet_label, axis=1)
                                    st.session_state['admet_completed'] = True
                                    st.session_state['active_with_admet'] = active_df
                                    st.rerun()

                                except Exception as e:
                                    st.warning(f"ADMET prediction unavailable: {e}. Please ensure 'rdkit' is installed.")
                                    # Fallback for entire batch failure
                                    active_df = active_df.copy()
                                    active_df['admet_score'] = None
                                    active_df['admet_threshold'] = 0.5
                                    active_df['admet_label'] = 'N/A (Error)'
                                    st.session_state['admet_completed'] = True
                                    st.session_state['active_with_admet'] = active_df
                                    st.rerun()
                        
                        # Display ADMET results (Persistence)
                        if 'active_with_admet' in st.session_state:
                            active_df = st.session_state['active_with_admet']
                        
                        keep_df = active_df[active_df['admet_label'] == 'Non-Toxic']
                        reject_toxic_df = active_df[active_df['admet_label'] == 'Toxic']
                        
                        col_stage2_1, col_stage2_2 = st.columns(2)
                        
                        with col_stage2_1:
                            st.metric(
                                "✅ Non-Toxic (Safe)",
                                len(keep_df),
                                delta=f"{len(keep_df)/len(active_df)*100:.1f}% of Active",
                                help="Active AND Non-Toxic"
                            )
                        
                        with col_stage2_2:
                            st.metric(
                                "❌ Toxic (Unsafe)",
                                len(reject_toxic_df),
                                delta=f"-{len(reject_toxic_df)/len(active_df)*100:.1f}% of Active",
                                delta_color="inverse",
                                help="Active BUT Toxic"
                            )
                        
                        # ========== FINAL DECISION ==========
                        st.markdown("---")
                        st.subheader("🎯 Final Decision & Statistics")
                        
                        # Apply final decision logic
                        active_df['final_decision'] = active_df['admet_label'].apply(
                            lambda x: 'KEEP' if x == 'Non-Toxic' else 'REJECT'
                        )
                        active_df['reason'] = active_df['admet_label'].apply(
                            lambda x: 'Active AND Non-Toxic' if x == 'Non-Toxic' 
                            else ('Active BUT Toxic' if x == 'Toxic' else 'ADMET Prediction Failed')
                        )
                        
                        # Add REJECT for inactive compounds
                        inactive_df = inactive_df.copy()
                        inactive_df['admet_score'] = None
                        inactive_df['admet_label'] = 'N/A (Not tested)'
                        inactive_df['admet_threshold'] = None
                        inactive_df['final_decision'] = 'REJECT'
                        inactive_df['reason'] = 'Inactive in bioactivity screening'
                        
                        # Combine all results
                        final_df = pd.concat([active_df, inactive_df], ignore_index=True)
                        st.session_state['batch_results'] = final_df
                        
                        # FUNNEL VISUALIZATION
                        st.markdown("### 📊 Two-Stage Filtering Funnel")
                        
                        funnel_col1, funnel_col2, funnel_col3, funnel_col4 = st.columns(4)
                        
                        with funnel_col1:
                            st.markdown("""
                            <div style='background: #90caf9; border-radius: 8px; border: 3px solid #1976d2; height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 35px; gap: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                <div style='color: #0d47a1; margin: 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2;'>{}</div>
                                <p style='margin: 0; font-size: 1rem; color: #1565c0; font-weight: 600; line-height: 1.2;'>Total Input</p>
                                <p style='margin: 0; font-size: 0.9rem; visibility: hidden; line-height: 1.2;'>Placeholder</p>
                            </div>
                            """.format(len(final_df)), unsafe_allow_html=True)
                        
                        with funnel_col2:
                            st.markdown("""
                            <div style='background: #ffcc80; border-radius: 8px; border: 3px solid #f57c00; height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 35px; gap: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                <div style='color: #e65100; margin: 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2;'>{}</div>
                                <p style='margin: 0; font-size: 1rem; color: #ef6c00; font-weight: 600; line-height: 1.2;'>✅ Active</p>
                                <p style='margin: 0; font-size: 0.9rem; color: #bf360c; font-weight: 500; line-height: 1.2;'>{:.1f}% passed</p>
                            </div>
                            """.format(len(active_df), len(active_df)/len(final_df)*100), unsafe_allow_html=True)
                        
                        with funnel_col3:
                            st.markdown("""
                            <div style='background: #a5d6a7; border-radius: 8px; border: 3px solid #388e3c; height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 35px; gap: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                <div style='color: #1b5e20; margin: 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2;'>{}</div>
                                <p style='margin: 0; font-size: 1rem; color: #2e7d32; font-weight: 600; line-height: 1.2;'>✅ Non-Toxic</p>
                                <p style='margin: 0; font-size: 0.9rem; color: #388e3c; font-weight: 500; line-height: 1.2;'>{:.1f}% of Active</p>
                            </div>
                            """.format(len(keep_df), len(keep_df)/len(active_df)*100 if len(active_df) > 0 else 0), unsafe_allow_html=True)
                        
                        with funnel_col4:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%); 
                                        border-radius: 8px; border: 3px solid #1b5e20; height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 35px; gap: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                                <div style='color: white; margin: 0; font-size: 1.8rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); line-height: 1.2;'>{}</div>
                                <p style='margin: 0; color: white; font-weight: 700; font-size: 1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); line-height: 1.2;'>🎯 KEEP</p>
                                <p style='margin: 0; font-size: 0.9rem; color: white; font-weight: 600; line-height: 1.2;'>{:.1f}% of Total</p>
                            </div>
                            """.format(len(keep_df), len(keep_df)/len(final_df)*100), unsafe_allow_html=True)
                        
                        st.success(f"""
                        ✅ **Filtering Complete!** 
                        - {len(final_df)} Total → {len(active_df)} Active → {len(keep_df)} KEEP ({len(keep_df)/len(final_df)*100:.1f}% retention rate)
                        """)
                else:
                    # No active compounds path logic
                    if 'batch_results' not in st.session_state:
                         # Automatically REJECT all inactive if they haven't been processed yet
                        inactive_df_all = results_df.copy()
                        inactive_df_all['admet_score'] = None
                        inactive_df_all['admet_label'] = 'N/A (Not tested)'
                        inactive_df_all['admet_threshold'] = None
                        inactive_df_all['final_decision'] = 'REJECT'
                        inactive_df_all['reason'] = 'Inactive in bioactivity screening'
                        st.session_state['batch_results'] = inactive_df_all
                    
                    st.warning("⚠️ No Active compounds found. All compounds are marked as REJECT.")

            # Display results (from batch_results session state which contains Finals)
            if 'batch_results' in st.session_state:
                results_df = st.session_state['batch_results']

                st.markdown("---")
                st.subheader("📋 Complete Results Table")

                st.markdown("---")

                # Interactive filtering
                st.subheader("🔍 Filter & View Results")

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    decision_filter = st.multiselect(
                        "Final Decision",
                        options=['KEEP', 'REJECT'],
                        default=['KEEP', 'REJECT'],
                        help="Filter by final decision"
                    )

                with filter_col2:
                    activity_filter = st.multiselect(
                        "Bioactivity Label",
                        options=['Active', 'Inactive'],
                        default=['Active', 'Inactive']
                    )

                with filter_col3:
                    pic50_range = st.slider(
                        "pIC50 Range",
                        min_value=float(results_df['bioactivity_score'].min()),
                        max_value=float(results_df['bioactivity_score'].max()),
                        value=(
                            float(results_df['bioactivity_score'].min()),
                            float(results_df['bioactivity_score'].max())
                        )
                    )

                # Apply filters
                filtered_df = results_df[
                    (results_df['final_decision'].isin(decision_filter)) &
                    (results_df['bioactivity_label'].isin(activity_filter)) &
                    (results_df['bioactivity_score'] >= pic50_range[0]) &
                    (results_df['bioactivity_score'] <= pic50_range[1])
                ]

                st.info(f"📊 Showing {len(filtered_df)} of {len(results_df)} compounds after filtering")

                # Results table with sorting
                st.subheader("📋 Results Table")

                # Color coding function
                def highlight_decision(row):
                    if row['final_decision'] == 'KEEP':
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)

                # Select important columns for display
                display_columns = [
                    'smiles', 'bioactivity_score', 'bioactivity_label',
                    'admet_score', 'admet_label', 'final_decision', 'reason'
                ]
                
                # Keep only columns that exist
                display_columns = [col for col in display_columns if col in filtered_df.columns]
                
                # Display with pagination
                page_size = 25
                total_pages = (len(filtered_df) + page_size - 1) // page_size

                if total_pages > 1:
                    page_num = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="batch_page_num"
                    )
                    start_idx = (page_num - 1) * page_size
                    end_idx = min(start_idx + page_size, len(filtered_df))
                    display_df = filtered_df[display_columns].iloc[start_idx:end_idx]
                    st.info(f"Page {page_num}/{total_pages} (Rows {start_idx+1}-{end_idx} of {len(filtered_df)})")
                else:
                    display_df = filtered_df[display_columns]

                st.dataframe(
                    display_df.style.apply(highlight_decision, axis=1),
                    use_container_width=True
                )

                st.markdown("---")

                # Export options
                st.subheader("💾 Export Results")
                
                st.info("""
                💡 **Track C Compliant Export:** Results include all required columns: 
                `smiles`, `bioactivity_score`, `bioactivity_label`, `bioactivity_threshold`, 
                `admet_score`, `admet_label`, `admet_threshold`, `final_decision`, `reason`
                """)

                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    # Export all results
                    csv_all = results_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download All Results ({len(results_df)} rows)",
                        data=csv_all,
                        file_name="batch_results_all.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with export_col2:
                    # Export KEEP only
                    keep_only_df = results_df[results_df['final_decision'] == 'KEEP']
                    csv_keep = keep_only_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download KEEP Only ({len(keep_only_df)} rows)",
                        data=csv_keep,
                        file_name="batch_results_keep.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )

                with export_col3:
                    # Export filtered results
                    csv_filtered = filtered_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Filtered ({len(filtered_df)} rows)",
                        data=csv_filtered,
                        file_name="batch_results_filtered.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.markdown("---")

                # Visualization
                st.subheader("📈 Distribution Analysis")

                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    "pIC50 Distribution",
                    "Activity Breakdown",
                    "Confidence Analysis"
                ])

                with viz_tab1:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(filtered_df['bioactivity_score'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                    ax.axvline(x=6.0, color='red', linestyle='--', label='Activity Threshold (pIC50=6.0)')
                    ax.set_xlabel('Predicted pIC50')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predicted pIC50 Values')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with viz_tab2:
                    activity_counts = filtered_df['bioactivity_label'].value_counts()

                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = ['#06A77D', '#D00000']
                    ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
                    ax.set_title('Activity Classification Breakdown')
                    st.pyplot(fig)

                with viz_tab3:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(
                        filtered_df['bioactivity_score'],
                        filtered_df['confidence'],
                        c=filtered_df['bioactivity_label'].map({'Active': '#06A77D', 'Inactive': '#D00000'}),
                        alpha=0.6,
                        s=50
                    )
                    ax.set_xlabel('Predicted pIC50')
                    ax.set_ylabel('Confidence')
                    ax.set_title('Prediction Confidence vs pIC50')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                st.markdown("---")

                # Export section
                st.subheader("💾 Export Results")

                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    # Export all results
                    csv_buffer = BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📥 Download All Results",
                        data=csv_buffer,
                        file_name="all_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with export_col2:
                    # Export filtered results
                    csv_buffer_filtered = BytesIO()
                    filtered_df.to_csv(csv_buffer_filtered, index=False)
                    csv_buffer_filtered.seek(0)

                    st.download_button(
                        label="📥 Download Filtered Results",
                        data=csv_buffer_filtered,
                        file_name="filtered_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with export_col3:
                    # Export active compounds only
                    active_df = results_df[results_df['bioactivity_label'] == 'Active']
                    csv_buffer_active = BytesIO()
                    active_df.to_csv(csv_buffer_active, index=False)
                    csv_buffer_active.seek(0)

                    st.download_button(
                        label="📥 Download Active Only",
                        data=csv_buffer_active,
                        file_name="active_compounds.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            elif validate_only:
                st.markdown("---")
                st.subheader("🧪 SMILES Validation Results")
                
                valid_count = 0
                invalid_count = 0
                validation_results = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Initializing validation...")
                
                try:
                    from rdkit import Chem
                    has_rdkit = True
                except ImportError:
                    has_rdkit = False
                    st.warning("⚠️ RDKit not installed. Performing basic string validation only.")
                
                total_rows = len(df)
                
                for idx, row in df.iterrows():
                    smiles = row[smiles_col]
                    is_valid = False
                    error_msg = "Unknown error"
                    
                    # Basic conversion to string
                    smiles_str = str(smiles).strip() if pd.notna(smiles) else ""
                    
                    if len(smiles_str) == 0:
                        is_valid = False
                        error_msg = "Empty value"
                    elif has_rdkit:
                        try:
                            # RDKit Validation
                            mol = Chem.MolFromSmiles(smiles_str)
                            if mol is not None:
                                is_valid = True
                                error_msg = "Valid"
                            else:
                                is_valid = False
                                error_msg = "RDKit parsing failed (Invalid syntax)"
                        except Exception as e:
                            is_valid = False
                            error_msg = f"Error: {str(e)}"
                    else:
                        # Fallback basic validation
                        valid_chars = set("CNOPSFIBrClcnops0123456789[]()=#@+-\\/")
                        if all(c in valid_chars for c in smiles_str):
                            is_valid = True
                            error_msg = "Pass (Basic structure check)"
                        else:
                            is_valid = False
                            error_msg = "Contains invalid characters"
                            
                    if is_valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        
                    validation_results.append({
                        'Row Index': idx + 1,
                        'Original Row ID': idx,
                        'SMILES': smiles_str,
                        'Status': 'Valid' if is_valid else 'Invalid',
                        'Message': error_msg
                    })
                    
                    # Update progress every 10%
                    if idx % max(1, total_rows // 10) == 0:
                         progress_bar.progress((idx + 1) / total_rows)
                
                progress_bar.progress(1.0)
                status_text.text("✓ Validation complete!")
                
                # Metrics
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    st.metric("Total Compounds", total_rows)
                with v_col2:
                    st.metric("✅ Valid SMILES", valid_count, delta=f"{valid_count/total_rows*100:.1f}%")
                with v_col3:
                    st.metric("❌ Invalid SMILES", invalid_count, delta=f"-{invalid_count}", delta_color="inverse")
                
                # Detail Table
                val_df = pd.DataFrame(validation_results)
                
                st.markdown("### 📋 Validation Details")
                
                # Define simple color styling
                def style_status(val):
                    color = '#c8e6c9' if val == 'Valid' else '#ffcdd2'
                    return f'background-color: {color}'

                # Display with styling
                st.dataframe(
                    val_df.style.map(style_status, subset=['Status']),
                    use_container_width=True
                )
                
                if invalid_count > 0:
                    st.warning("⚠️ Found invalid SMILES entries.")
                    col_fix1, col_fix2 = st.columns(2)
                    with col_fix1:
                        st.info("💡 **Tip:** Remove these rows from your CSV file and re-upload.")
                    with col_fix2:
                         # Export valid only
                         valid_rows_indices = [res['Original Row ID'] for res in validation_results if res['Status'] == 'Valid']
                         if valid_rows_indices:
                             clean_df = df.iloc[valid_rows_indices]
                             csv_clean = BytesIO()
                             clean_df.to_csv(csv_clean, index=False)
                             csv_clean.seek(0)
                             
                             st.download_button(
                                 label="📥 Download Cleaned CSV (Valid Only)",
                                 data=csv_clean,
                                 file_name="cleaned_compounds.csv",
                                 mime="text/csv",
                                 key="download_clean_csv"
                             )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Upload a CSV file to begin batch analysis")
