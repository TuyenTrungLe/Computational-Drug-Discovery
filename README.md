# 🧬 Bio-ScreenNet: Multi‑Stage Drug Discovery Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![AI](https://img.shields.io/badge/AI-TensorFlow%20%7C%20Scikit--Learn-orange)](https://tensorflow.org/)
[![App](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io/)
![Status](https://img.shields.io/badge/Project-Capstone%20Track%20C-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> **Capstone Project – Computational Drug Discovery (Track C)**  
> **Goal:** Build an AI‑powered *virtual drug screening* system to simultaneously optimize **bioactivity** and **safety (ADMET)** for candidate compounds targeting **[Protein Target]** related to **[Disease]**.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Pipeline Architecture](#-pipeline-architecture)
3. [Installation & Usage](#-installation--usage)
4. [Project Structure](#-project-structure)
5. [Datasets](#-datasets)
6. [Modeling Approach](#-modeling-approach)
7. [Results & Evaluation](#-results--evaluation)
8. [Explainable AI (XAI)](#-explainable-ai-xai)
9. [Demo Application](#-demo-application)
10. [References](#-references)
11. [Contributors](#-contributors)

---

## 🌐 Overview

Drug discovery traditionally takes *10–15 years* and billions of dollars.  
This project leverages **Machine Learning** and **Deep Learning** to accelerate early‑stage drug discovery via:

- Predicting bioactivity (pIC50) from SMILES  
- Filtering compounds using ADMET safety models  
- Visualizing atom‑level explanations with XAI  
- Providing a friendly Streamlit app for researchers

Target protein: **[Example: CHEMBL220 – Acetylcholinesterase]**  
Disease context: **[Example: Prostate Cancer]**

---

## 🔗 Pipeline Architecture

![Pipeline Architecture](static/pipeline_architecture.png)

### Pipeline Summary:
1. **Stage 1**: Predict bioactivity (pIC50) using Random Forest or LSTM/GRU
2. **Stage 2**: Filter compounds based on ADMET safety properties
3. **Stage 3**: Explain predictions with XAI visualizations

---

## 💻 Installation & Usage

### **Prerequisites**
- Python 3.8 - 3.11
- [UV package manager](https://docs.astral.sh/uv/) (recommended)

### **Quick Start with UV**

#### 1️⃣ Install UV
```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2️⃣ Clone Repository
```bash
git clone https://github.com/TuyenTrungLe/Computational-Drug-Discovery.git
cd Computational-Drug-Discovery
```

#### 3️⃣ Setup Environment
```powershell
# Create virtual environment
uv venv

# Activate environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# Install all dependencies
uv pip install -e .
```

#### 4️⃣ Run Project
```powershell
# Open Jupyter Notebook
jupyter notebook

# Or run Streamlit app
streamlit run app/app.py
```

### **Common UV Commands**
```powershell
# Install new package
uv pip install package-name

# List installed packages
uv pip list

# Update dependencies
uv pip install -e .
```

### **Alternative: Traditional pip (not recommended)**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

---

## 📁 Project Structure

```
Computational-Drug-Discovery/
├── 📂 src/                    # Source code
│   ├── data/                  # Data processing modules
│   ├── models/                # ML/DL models
│   ├── features/              # Feature engineering
│   ├── visualization/         # Plotting & XAI
│   └── utils/                 # Utilities
├── 📂 notebooks/              # Jupyter notebooks
├── 📂 data/                   # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── external/              # External datasets
├── 📂 models/                 # Saved models
├── 📂 app/                    # Streamlit application
├── 📂 configs/                # Configuration files
├── 📂 tests/                  # Unit tests
├── 📂 static/                 # Static assets
├── pyproject.toml             # Project dependencies
└── README.md                  # Documentation
```

---

## 💾 Datasets

### **1. Bioactivity Data (ChEMBL)**
- Source: [ChEMBL](https://www.ebi.ac.uk/chembl/)  
- Target protein: **[Protein Name]**
- Label: IC50 → transformed to **pIC50**
- Size: **[N samples] after preprocessing**

### **2. ADMET Dataset (MoleculeNet – DeepChem)**
- Safety attributes:
  - Toxicity classification (e.g., Tox21)
  - Solubility regression (e.g., ESOL)
- Purpose: Filter unsafe compounds before prioritization

---

## 🛠 Modeling Approach

### **4.1 Baseline: Random Forest**
- Input: Calculated molecular descriptors
  - Lipinski descriptors: MW, LogP, H‑donors, H‑acceptors  
  - Molecular fingerprints: Morgan/PubChem
- Pros: Fast, interpretable, strong baseline

### **4.2 Deep Learning: LSTM/GRU**
- Input: Raw SMILES sequence
- Steps:
  - Character tokenization  
  - Embedding layer  
  - LSTM or GRU  
  - Dense regression head  
- Inspired by: *Belaidi et al., 2024*

### **4.3 ADMET Safety Filter**
- Separate classifier/regressor  
- Methods: SVM / RF  
- Output: Toxic / Non‑toxic or numeric ADMET scores

---

## 📊 Results & Evaluation

| Metric | Random Forest | LSTM/GRU | Notes |
|-------|---------------|----------|-------|
| R² | 0.XX | 0.YY | DL typically higher |
| RMSE | 0.XX | 0.YY | Lower = better |
| Training Time | X min | Y min | DL slower but stronger |

Scatter plots, training curves, and confusion matrices are available in the Jupyter Notebook.

---

## 🔍 Explainable AI (XAI)

To address the *black box* problem, we provide:

### **RDKit Similarity Maps**
- Shows atom contributions  
- Green = increases activity  
- Red = decreases activity  
- Supports medicinal chemistry reasoning

---

## 📱 Demo Application

A user‑friendly Streamlit application:

### Features:
- Input SMILES or upload CSV  
- Predict pIC50 using both models  
- Run ADMET safety filtering  
- Export final candidate list  
- View XAI heatmaps  

---

## 📚 References
- EnriqueSPR — Drug Discovery Project (Random Forest baseline)  
- Belaidi, A. et al. (2024). *Predicting pIC50 using Deep Learning*  
- Gaulton, A. et al. (2012). *ChEMBL: A large-scale bioactivity database*  

---

## 👥 Contributors
| Name | Email |
|------|-------|
| **Lê Trung Tuyến** | letrungtuyen2002@gmail.com |
| **Bùi Hoàng Nhân** | [Email] |

---

*Capstone Project - Computational Drug Discovery Track C*
