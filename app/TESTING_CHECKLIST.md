# ✅ Testing Checklist - Bio-ScreenNet Frontend

Use this checklist to verify all features are working correctly.

## Pre-Testing Setup

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -e .`)
- [ ] App launches without errors (`streamlit run app/app.py`)
- [ ] Browser opens to correct URL (http://localhost:8501)

---

## 🏠 Home Page

### Visual Elements
- [ ] Header displays correctly with gradient
- [ ] Navigation sidebar visible
- [ ] Stage indicators shown (Stage 1, 2, 3)
- [ ] Project info sidebar visible
- [ ] Pipeline architecture image loads

### Content
- [ ] Overview section displays
- [ ] Key features cards visible
- [ ] Quick start guide readable
- [ ] Example SMILES table shows
- [ ] All text properly formatted

### Navigation
- [ ] Can click all navigation options
- [ ] Page changes when selecting different options
- [ ] Sidebar collapses/expands properly

---

## 💊 Single Compound Screening

### Input Section
- [ ] SMILES text area accepts input
- [ ] Model selection dropdown works
- [ ] Advanced options expander opens
- [ ] All checkboxes toggle
- [ ] Slider adjusts values
- [ ] Predict button visible and clickable

### Validation
- [ ] Empty SMILES shows error
- [ ] Invalid SMILES shows error message
- [ ] Valid SMILES shows success message

### Test Cases

#### Test 1: Ibuprofen
```
SMILES: CC(C)Cc1ccc(cc1)C(C)C(O)=O
Model: Random Forest
Expected: Should predict, show structure, pass Lipinski
```
- [ ] Prediction completes
- [ ] pIC50 value displays
- [ ] IC50 calculated
- [ ] Activity classification shown
- [ ] Molecular structure renders (if RDKit available)
- [ ] Descriptors table displays
- [ ] Lipinski check shows results
- [ ] XAI visualization appears

#### Test 2: Invalid SMILES
```
SMILES: XYZ123ABC
Expected: Validation error
```
- [ ] Error message displays
- [ ] No prediction attempted
- [ ] User can correct and retry

### Output Features
- [ ] Metrics display in columns
- [ ] Structure image renders
- [ ] Descriptors formatted correctly
- [ ] XAI section visible
- [ ] Pharmacophores listed
- [ ] Export buttons work

### Export Functions
- [ ] CSV download button creates file
- [ ] JSON download button creates file
- [ ] Downloaded files contain correct data
- [ ] Filenames are appropriate

---

## 📊 Batch Analysis

### Upload Section
- [ ] File uploader appears
- [ ] Sample template downloads
- [ ] Downloaded template opens in Excel/CSV viewer
- [ ] Settings dropdowns work
- [ ] Batch size input accepts numbers

### Test Cases

#### Test 1: Upload Sample Template
- [ ] Download sample template
- [ ] Upload the downloaded file
- [ ] File uploads successfully
- [ ] Preview shows data
- [ ] Statistics display (total, unique, duplicates)

#### Test 2: Run Batch Prediction
- [ ] Click "Run Batch Prediction"
- [ ] Progress bar appears and updates
- [ ] Status text updates
- [ ] Predictions complete
- [ ] Results summary displays

### Results Display
- [ ] Summary metrics show (4 cards)
- [ ] Filter controls work:
  - [ ] Activity multiselect
  - [ ] pIC50 slider
  - [ ] Confidence slider
- [ ] Results table displays
- [ ] Color coding applied (green=active, red=inactive)
- [ ] Pagination works (if >50 results)

### Visualizations
- [ ] pIC50 Distribution histogram renders
- [ ] Activity Breakdown pie chart renders
- [ ] Confidence Analysis scatter plot renders
- [ ] All charts have proper labels
- [ ] Charts are interactive (if using plotly)

### Export Functions
- [ ] Download All Results works
- [ ] Download Filtered Results works
- [ ] Download Active Only works
- [ ] All CSV files open correctly

---

## 🎯 ADMET Filter

### Input Section
- [ ] Three input methods available:
  - [ ] Previous batch results loads
  - [ ] New CSV upload works
  - [ ] Manual SMILES entry works

### Filter Configuration
- [ ] Toxicity filters section displays
- [ ] Tox21 checkbox and slider work
- [ ] ClinTox checkbox and slider work
- [ ] Mutagenicity checkbox works
- [ ] Solubility filter section displays
- [ ] ESOL checkbox and slider work
- [ ] BBBP checkbox and radio buttons work
- [ ] Lipinski checkbox works

### Test Cases

#### Test 1: Apply All Filters
```
Settings:
- Enable all filters
- Use default thresholds
- Run on previous batch results
```
- [ ] ADMET analysis runs
- [ ] Progress indicator shows
- [ ] Results generate

### Results Display
- [ ] 5 summary metrics show
- [ ] Three tabs display:
  - [ ] All Compounds
  - [ ] Passed Filters
  - [ ] Failed Compounds
- [ ] Top 10 candidates section visible
- [ ] Pass/fail clearly indicated

### Visualizations
- [ ] Filter Pass Rates bar chart displays
- [ ] Property Distributions (4 subplots) render
- [ ] Risk Matrix scatter plot displays
- [ ] All charts properly labeled
- [ ] Colors meaningful

### Export Functions
- [ ] Download Passed Compounds works
- [ ] Download All ADMET Results works
- [ ] Download Failed Compounds works

---

## 🔬 Model Comparison

### Overview Section
- [ ] 3 model cards display
- [ ] Information formatted correctly
- [ ] Icons and styling applied

### Metrics Table
- [ ] Performance table displays
- [ ] All metrics visible (R², RMSE, MAE, etc.)
- [ ] Highlighting applied correctly
- [ ] Values reasonable

### Analysis Tabs

#### Tab 1: Accuracy Comparison
- [ ] Bar charts render (3 charts)
- [ ] Values labeled on bars
- [ ] Summary text displays in columns
- [ ] Best/fastest model highlighted

#### Tab 2: Training Curves
- [ ] Loss curves plot renders
- [ ] R² curves plot renders
- [ ] Legends visible
- [ ] Grid and labels present
- [ ] Observations text displays

#### Tab 3: Prediction Scatter
- [ ] Two scatter plots render (RF and LSTM)
- [ ] Perfect prediction line shown
- [ ] Axes labeled correctly
- [ ] R² scores in titles
- [ ] Residual analysis section displays
- [ ] Residual plots render

#### Tab 4: Feature Importance
- [ ] Horizontal bar chart displays
- [ ] Features labeled
- [ ] Values shown
- [ ] Gradient colors applied
- [ ] Key insights text displays

### Recommendations
- [ ] Model selection boxes display
- [ ] Pros/cons listed
- [ ] Use cases described
- [ ] Ensemble approach explained

### Technical Details
- [ ] Expander opens
- [ ] Configuration details readable
- [ ] Code formatting applied

---

## 📖 About & Documentation

### Tab Navigation
- [ ] 5 tabs visible and clickable
- [ ] Content loads when switching tabs

### Tab 1: Project Overview
- [ ] Overview text displays
- [ ] Goals listed
- [ ] Requirements checklist shows
- [ ] Target protein info formatted
- [ ] Pipeline image loads
- [ ] Stage descriptions clear

### Tab 2: Quick Start Guide
- [ ] Instructions formatted
- [ ] Expandable sections work
- [ ] Code blocks formatted
- [ ] Steps numbered
- [ ] Tips box displays

### Tab 3: Technical Details
- [ ] Sub-tabs work (4 tabs)
- [ ] Dataset descriptions complete
- [ ] Model architectures shown
- [ ] Code blocks formatted
- [ ] Metrics explanations clear
- [ ] XAI methods described

### Tab 4: References
- [ ] References numbered
- [ ] Citations formatted
- [ ] Links clickable
- [ ] Sections organized

### Tab 5: Team & Contact
- [ ] Team cards display
- [ ] Contact info visible
- [ ] Links work
- [ ] License text shows
- [ ] Acknowledgments listed

---

## 🎨 UI/UX Elements

### Styling
- [ ] Custom CSS loaded
- [ ] Gradient backgrounds display
- [ ] Colors consistent throughout
- [ ] Buttons styled
- [ ] Cards have shadows
- [ ] Spacing appropriate

### Responsive Design
- [ ] Layout adjusts to window size
- [ ] Sidebar collapses on narrow screens
- [ ] Columns stack on mobile
- [ ] Text readable at all sizes

### Interactive Elements
- [ ] Buttons have hover effects
- [ ] Links underline on hover
- [ ] Expanders open/close smoothly
- [ ] Tabs switch instantly
- [ ] Forms submit correctly

### Feedback Messages
- [ ] Success messages (green) display
- [ ] Error messages (red) display
- [ ] Warning messages (yellow) display
- [ ] Info messages (blue) display
- [ ] Messages dismissible

---

## 🔧 Error Handling

### Input Validation
- [ ] Empty inputs rejected
- [ ] Invalid SMILES caught
- [ ] File format errors handled
- [ ] Missing columns detected

### Graceful Degradation
- [ ] App works without RDKit (shows warnings)
- [ ] Missing models handled
- [ ] Large files processed (with warnings)
- [ ] Network errors caught

---

## 🚀 Performance

### Loading Times
- [ ] Home page loads < 2 seconds
- [ ] Page switching < 1 second
- [ ] Single prediction < 3 seconds
- [ ] Batch 100 compounds < 30 seconds

### Memory Usage
- [ ] No memory leaks on repeated use
- [ ] Large batches don't crash
- [ ] Session state managed properly

---

## 📱 Cross-Browser Testing

### Browsers to Test
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if available)

### Features to Verify
- [ ] Layout consistent
- [ ] All features work
- [ ] File uploads work
- [ ] Downloads work
- [ ] Visualizations render

---

## 🐛 Known Issues

Document any issues found during testing:

| Issue | Page | Severity | Workaround |
|-------|------|----------|------------|
| Example: Slow on large batches | Batch Analysis | Medium | Use batch size < 100 |
|  |  |  |  |
|  |  |  |  |

---

## ✅ Sign-Off

**Testing Completed By:** _____________________

**Date:** _____________________

**Overall Status:** ⬜ Pass ⬜ Pass with Minor Issues ⬜ Fail

**Notes:**
_____________________________________________
_____________________________________________
_____________________________________________

---

## 🔄 Regression Testing

For future updates, re-run this checklist to ensure:
- [ ] Existing features still work
- [ ] New features don't break old ones
- [ ] Performance hasn't degraded
- [ ] UI remains consistent

---

## 📞 Reporting Issues

If you find bugs:

1. Document the issue in the table above
2. Create GitHub issue with:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots (if applicable)
3. Assign appropriate priority
4. Link to this checklist

**GitHub Issues:** https://github.com/TuyenTrungLe/Computational-Drug-Discovery/issues
