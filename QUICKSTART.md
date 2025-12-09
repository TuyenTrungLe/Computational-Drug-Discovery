# 🚀 Quick Start Guide - Bio-ScreenNet

Get started with Bio-ScreenNet in 5 minutes!

## Prerequisites

- Python 3.8 - 3.11
- Git (for cloning repository)
- 4GB RAM minimum
- Internet connection (for first-time setup)

## Installation

### Option 1: Quick Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/TuyenTrungLe/Computational-Drug-Discovery.git
cd Computational-Drug-Discovery

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -e .

# 5. Run the application
streamlit run app/app.py
```

### Option 2: Using UV (Faster)

```bash
# 1. Install UV
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone https://github.com/TuyenTrungLe/Computational-Drug-Discovery.git
cd Computational-Drug-Discovery

# 3. Create venv and install
uv venv
.\.venv\Scripts\Activate.ps1  # Windows
uv pip install -e .

# 4. Run
streamlit run app/app.py
```

## First Run

After running `streamlit run app/app.py`, your browser will open to:

```
http://localhost:8501
```

You should see the Bio-ScreenNet home page!

## Quick Test

### Test 1: Single Compound Prediction

1. Click **"💊 Single Compound Screening"** in sidebar
2. Enter this SMILES:
   ```
   CC(C)Cc1ccc(cc1)C(C)C(O)=O
   ```
   (This is Ibuprofen)
3. Select **"Random Forest (Baseline)"**
4. Click **"🔬 Predict Bioactivity"**
5. View the predicted pIC50 and XAI visualization!

### Test 2: Batch Analysis

1. Click **"📊 Batch Analysis"** in sidebar
2. Click **"📥 Download Sample CSV Template"**
3. Upload the downloaded template
4. Click **"🚀 Run Batch Prediction"**
5. Explore the results and download filtered compounds!

## Common Issues & Solutions

### Issue: RDKit Import Error

```bash
# Solution: Install RDKit
conda install -c conda-forge rdkit
# Or using pip:
pip install rdkit-pypi
```

### Issue: Port Already in Use

```bash
# Solution: Use different port
streamlit run app/app.py --server.port 8502
```

### Issue: Module Not Found

```bash
# Solution: Make sure you're in the right directory and venv is activated
cd Computational-Drug-Discovery
.\.venv\Scripts\Activate.ps1  # Windows
pip install -e .
```

### Issue: Slow Performance

```bash
# Solution: Reduce batch size in settings
# In Batch Analysis page, set batch size to 50 instead of 100
```

## Project Structure Overview

```
Computational-Drug-Discovery/
├── app/                    # 🎨 Frontend (Streamlit)
│   ├── app.py             # Main application
│   └── pages/             # Individual pages
├── src/                    # 🔧 Backend (Models & Processing)
│   ├── data/              # Data processing
│   ├── models/            # ML/DL models
│   └── features/          # Feature engineering
├── data/                   # 💾 Datasets
├── models/                 # 🤖 Trained models
├── notebooks/              # 📓 Jupyter notebooks
└── static/                 # 🖼️ Images & assets
```

## Next Steps

1. **Explore the Interface:**
   - Try all pages in the sidebar
   - Test with different SMILES strings
   - Upload your own compound libraries

2. **Understand the Pipeline:**
   - Read the "📖 About & Documentation" page
   - View the pipeline architecture diagram
   - Check model comparison metrics

3. **Integrate with Your Data:**
   - Prepare your CSV files (SMILES column required)
   - Run batch predictions
   - Apply ADMET filters
   - Export results for further analysis

4. **Connect Backend Models:**
   - Train your models using notebooks in `notebooks/`
   - Save trained models to `models/` directory
   - Update `app/utils/model_loader.py` to load real models
   - Replace placeholder predictions with actual inference

## Example Workflows

### Workflow 1: Screen New Compounds

```
1. Prepare CSV with SMILES
2. Go to Batch Analysis
3. Upload CSV
4. Run predictions
5. Filter by pIC50 ≥ 6.0
6. Apply ADMET filters
7. Download top candidates
8. Send to experimental validation
```

### Workflow 2: Analyze Known Drugs

```
1. Go to Single Compound Screening
2. Enter drug SMILES (e.g., from PubChem)
3. Predict bioactivity
4. Review XAI explanations
5. Identify key pharmacophores
6. Design analogs with similar features
```

### Workflow 3: Compare Model Performance

```
1. Go to Model Comparison
2. Review metrics (R², RMSE, MAE)
3. Check training curves
4. View scatter plots
5. Decide which model to use for your task
```

## Keyboard Shortcuts (Streamlit)

- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show keyboard shortcuts

## Performance Tips

1. **For large datasets:**
   - Use batch size of 50-100
   - Enable pagination
   - Process overnight if needed

2. **For faster predictions:**
   - Use Random Forest model
   - Reduce visualization complexity
   - Cache results in session state

3. **For better accuracy:**
   - Use LSTM/GRU model
   - Increase confidence threshold
   - Apply multiple ADMET filters

## Getting Help

- 📖 **Documentation:** Check the "About & Documentation" page in the app
- 🐛 **Issues:** Report bugs on [GitHub Issues](https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues)
- 💬 **Questions:** Email letrungtuyen2002@gmail.com
- 📚 **Learn More:** Read the main [README.md](README.md)

## What's Next?

Now that you have the app running, you can:

1. ✅ Test with example compounds
2. ✅ Upload your own data
3. ✅ Explore all features
4. ✅ Train your own models (see notebooks/)
5. ✅ Deploy to production (see app/README.md)

**Happy Drug Discovery! 🧬💊**
