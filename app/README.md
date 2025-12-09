# Bio-ScreenNet Streamlit Application

This directory contains the Streamlit web application for Bio-ScreenNet drug discovery pipeline.

## Directory Structure

```
app/
├── app.py                      # Main application entry point
├── pages/                      # Page modules
│   ├── __init__.py
│   ├── single_compound_page.py # Single SMILES prediction
│   ├── batch_analysis_page.py  # Batch CSV processing
│   ├── admet_filter_page.py    # ADMET safety filtering
│   ├── model_comparison_page.py# Model performance comparison
│   └── about_page.py           # Documentation
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── smiles_utils.py         # SMILES validation
│   └── model_loader.py         # Model loading utilities
├── components/                 # Reusable UI components
└── README.md                   # This file
```

## Running the Application

### Local Development

```bash
# From project root directory
streamlit run app/app.py
```

### With Custom Port

```bash
streamlit run app/app.py --server.port 8501
```

### Production Deployment

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

## Features

### 1. Single Compound Screening
- Input individual SMILES strings
- Predict bioactivity (pIC50)
- View molecular structure and descriptors
- XAI visualization of atom contributions
- Export results

### 2. Batch Analysis
- Upload CSV files with multiple compounds
- Batch prediction with progress tracking
- Interactive filtering and sorting
- Distribution visualizations
- Export filtered results

### 3. ADMET Safety Filter
- Apply toxicity filters (Tox21, ClinTox)
- Evaluate solubility (ESOL)
- Check BBBP penetration
- Lipinski's Rule of Five compliance
- Risk matrix visualization

### 4. Model Comparison
- Compare Random Forest vs LSTM/GRU
- View performance metrics (R², RMSE, MAE)
- Training curves and scatter plots
- Feature importance analysis
- Model selection recommendations

### 5. About & Documentation
- Project overview
- Quick start guide
- Technical details
- References and resources
- Team information

## Configuration

The app uses Streamlit's configuration system. You can customize:

- Theme colors
- Page layout
- Sidebar behavior
- File upload limits

Create a `.streamlit/config.toml` file in the project root:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Integration with Backend

The app currently uses placeholder predictions. To integrate with trained models:

1. **Update `model_loader.py`:**
   - Implement actual model loading from `models/` directory
   - Add proper feature extraction for each model type
   - Handle preprocessing and postprocessing

2. **Update prediction functions:**
   - `single_compound_page.py`: Replace `predict_bioactivity()`
   - `batch_analysis_page.py`: Use model_loader for batch predictions
   - `admet_filter_page.py`: Replace `predict_admet_properties()`

3. **Add XAI integration:**
   - Implement RDKit similarity maps
   - Add gradient-based attribution for LSTM
   - Integrate SHAP for Random Forest

## Dependencies

Main dependencies (see `pyproject.toml` for complete list):

- `streamlit>=1.20.0` - Web framework
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting
- `rdkit>=2022.3.1` - Cheminformatics
- `scikit-learn>=1.0.0` - ML models
- `tensorflow>=2.8.0` - Deep learning (optional)

## Troubleshooting

### Common Issues

**1. RDKit import error:**
```bash
# Install RDKit
conda install -c conda-forge rdkit
# Or
pip install rdkit-pypi
```

**2. Port already in use:**
```bash
# Use different port
streamlit run app/app.py --server.port 8502
```

**3. File upload limit:**
- Edit `.streamlit/config.toml`
- Increase `maxUploadSize` value

**4. Memory issues with large batches:**
- Reduce batch size in settings
- Process in smaller chunks
- Use pagination for display

## Development Guidelines

### Adding New Pages

1. Create new file in `app/pages/`
2. Implement `render()` function
3. Add navigation option in `app.py`
4. Update imports

Example:
```python
# app/pages/my_new_page.py
import streamlit as st

def render():
    st.title("My New Page")
    st.write("Content here")
```

### Adding New Features

1. Keep UI components modular
2. Use session state for data persistence
3. Add progress indicators for long operations
4. Implement error handling
5. Add tooltips and help text

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions focused
- Comment complex logic

## Testing

### Manual Testing Checklist

- [ ] Single compound prediction works
- [ ] Batch upload and processing
- [ ] ADMET filters apply correctly
- [ ] Export functions download files
- [ ] Visualizations render properly
- [ ] Navigation between pages
- [ ] Mobile responsiveness
- [ ] Error messages display

### Example SMILES for Testing

```
# Ibuprofen (should be active)
CC(C)Cc1ccc(cc1)C(C)C(O)=O

# Aspirin (should be active)
CC(=O)Oc1ccccc1C(=O)O

# Caffeine (moderate activity)
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Configure secrets in dashboard
4. Deploy

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]
```

### Heroku

```bash
# Create Procfile
echo "web: streamlit run app/app.py --server.port=$PORT" > Procfile

# Deploy
heroku create bio-screennet
git push heroku main
```

## Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## License

MIT License - See main project README

## Support

For issues and questions:
- GitHub Issues: [Project Issues](https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues)
- Email: letrungtuyen2002@gmail.com
