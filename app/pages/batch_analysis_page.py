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

        apply_filters = st.checkbox("Apply ADMET Filters", value=False)

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

            # Process predictions
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
                try:
                    from app.utils.model_loader import predict_batch as batch_predict
                except ImportError:
                    # Fallback for import issues
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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

                status_text.text("✓ Predictions completed!")

                # Create results dataframe
                results_df = pd.DataFrame(results_list)

                # Store in session state
                st.session_state['batch_results'] = results_df

                st.success(f"✓ Successfully processed {len(results_df)} compounds")

            # Display results
            if 'batch_results' in st.session_state:
                results_df = st.session_state['batch_results']

                st.markdown("---")
                st.subheader("📊 Results Summary")

                # Summary metrics
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    active_count = len(results_df[results_df['activity'] == 'Active'])
                    st.metric("Active Compounds", active_count,
                             delta=f"{active_count/len(results_df)*100:.1f}%")

                with metric_cols[1]:
                    avg_pic50 = results_df['predicted_pIC50'].mean()
                    st.metric("Avg pIC50", f"{avg_pic50:.2f}")

                with metric_cols[2]:
                    avg_confidence = results_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                with metric_cols[3]:
                    high_conf_active = len(results_df[
                        (results_df['activity'] == 'Active') &
                        (results_df['confidence'] >= 0.8)
                    ])
                    st.metric("High-Confidence Active", high_conf_active)

                st.markdown("---")

                # Interactive filtering
                st.subheader("🔍 Filter Results")

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    activity_filter = st.multiselect(
                        "Activity",
                        options=['Active', 'Inactive'],
                        default=['Active', 'Inactive']
                    )

                with filter_col2:
                    pic50_range = st.slider(
                        "pIC50 Range",
                        min_value=float(results_df['predicted_pIC50'].min()),
                        max_value=float(results_df['predicted_pIC50'].max()),
                        value=(
                            float(results_df['predicted_pIC50'].min()),
                            float(results_df['predicted_pIC50'].max())
                        )
                    )

                with filter_col3:
                    conf_threshold = st.slider(
                        "Min Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.05
                    )

                # Apply filters
                filtered_df = results_df[
                    (results_df['activity'].isin(activity_filter)) &
                    (results_df['predicted_pIC50'] >= pic50_range[0]) &
                    (results_df['predicted_pIC50'] <= pic50_range[1]) &
                    (results_df['confidence'] >= conf_threshold)
                ]

                st.info(f"Showing {len(filtered_df)} of {len(results_df)} compounds after filtering")

                # Results table with sorting
                st.subheader("📋 Filtered Results")

                # Add color coding
                def highlight_activity(row):
                    if row['activity'] == 'Active':
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)

                # Display with pagination
                page_size = 50
                total_pages = (len(filtered_df) + page_size - 1) // page_size

                if total_pages > 1:
                    page_num = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1
                    )
                    start_idx = (page_num - 1) * page_size
                    end_idx = min(start_idx + page_size, len(filtered_df))
                    display_df = filtered_df.iloc[start_idx:end_idx]
                else:
                    display_df = filtered_df

                st.dataframe(
                    display_df.style.apply(highlight_activity, axis=1),
                    use_container_width=True,
                    height=600
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
                    ax.hist(filtered_df['predicted_pIC50'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                    ax.axvline(x=6.0, color='red', linestyle='--', label='Activity Threshold (pIC50=6.0)')
                    ax.set_xlabel('Predicted pIC50')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predicted pIC50 Values')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with viz_tab2:
                    activity_counts = filtered_df['activity'].value_counts()

                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = ['#06A77D', '#D00000']
                    ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
                    ax.set_title('Activity Classification Breakdown')
                    st.pyplot(fig)

                with viz_tab3:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(
                        filtered_df['predicted_pIC50'],
                        filtered_df['confidence'],
                        c=filtered_df['activity'].map({'Active': '#06A77D', 'Inactive': '#D00000'}),
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
                    active_df = results_df[results_df['activity'] == 'Active']
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
                st.info("SMILES validation feature coming soon!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Upload a CSV file to begin batch analysis")
