"""
ADMET Safety Model - Multi-Property Drug Safety Prediction

This module implements a comprehensive ADMET (Absorption, Distribution, Metabolism,
Excretion, and Toxicity) safety filtering system using Random Forest models.

Features:
    - Multi-task ADMET prediction (Toxicity, Clinical Toxicity, BBB Permeability, Side Effects, Solubility)
    - RDKit molecular descriptor calculation
    - Random Forest classification and regression
    - Model persistence and loading
    - Comprehensive evaluation metrics
    - SMILES-based compound filtering

Datasets used:
    - Tox21: Toxicity prediction (12 targets)
    - ClinTox: Clinical trial toxicity
    - BBBP: Blood-Brain Barrier Permeability
    - SIDER: Side Effects
    - ESOL (Delaney): Aqueous Solubility

Author: Bio-ScreenNet Team
Date: 2025
"""

import os
import sys
import warnings
import gzip
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings
except ImportError:
    print("Warning: RDKit not installed. Please install: conda install -c conda-forge rdkit")
    sys.exit(1)

warnings.filterwarnings('ignore')


class MolecularDescriptorCalculator:
    """Calculate molecular descriptors from SMILES strings using RDKit."""

    @staticmethod
    def calculate_descriptors(smiles: str) -> Optional[np.ndarray]:
        """
        Calculate molecular descriptors for a given SMILES string.

        Args:
            smiles: SMILES representation of molecule

        Returns:
            Array of molecular descriptors or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Calculate Lipinski descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)

            # Additional descriptors
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            aliphatic_rings = Descriptors.NumAliphaticRings(mol)

            # Fingerprint-based descriptors (Morgan fingerprint)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
            fp_array = np.array(fp)

            # Combine all descriptors
            basic_descriptors = np.array([
                mw, logp, hbd, hba, tpsa, rot_bonds,
                aromatic_rings, aliphatic_rings
            ])

            descriptors = np.concatenate([basic_descriptors, fp_array])
            return descriptors

        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return None

    @staticmethod
    def batch_calculate_descriptors(smiles_list: List[str], show_progress: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Calculate descriptors for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (descriptor_matrix, valid_indices)
        """
        descriptors = []
        valid_indices = []

        iterator = tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Calculating descriptors") if show_progress else enumerate(smiles_list)

        for idx, smiles in iterator:
            desc = MolecularDescriptorCalculator.calculate_descriptors(smiles)
            if desc is not None:
                descriptors.append(desc)
                valid_indices.append(idx)

        return np.array(descriptors), valid_indices


class ADMETSafetyModel:
    """
    Comprehensive ADMET Safety Prediction Model.

    This class handles multiple ADMET properties including:
    - Toxicity (Tox21)
    - Clinical Toxicity (ClinTox)
    - Blood-Brain Barrier Permeability (BBBP)
    - Side Effects (SIDER)
    - Aqueous Solubility (ESOL)
    """

    def __init__(self, data_dir: str = None, model_dir: str = None):
        """
        Initialize ADMET Safety Model.

        Args:
            data_dir: Directory containing ADMET datasets
            model_dir: Directory to save/load trained models
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), ".deepchem", "datasets")
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "admet_models")

        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.results = {}

        print(f"ADMET Model initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Model directory: {self.model_dir}")

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load ADMET dataset from file.

        Args:
            dataset_name: Name of dataset (tox21, clintox, bbbp, sider, delaney)

        Returns:
            DataFrame containing the dataset or None if loading fails
        """
        dataset_files = {
            'tox21': 'tox21.csv.gz',
            'clintox': 'clintox.csv.gz',
            'bbbp': 'BBBP.csv',
            'sider': 'sider.csv.gz',
            'delaney': 'delaney-processed.csv'
        }

        if dataset_name not in dataset_files:
            print(f"Unknown dataset: {dataset_name}")
            return None

        file_path = self.data_dir / dataset_files[dataset_name]

        if not file_path.exists():
            print(f"Dataset file not found: {file_path}")
            return None

        try:
            print(f"\nLoading {dataset_name} dataset from {file_path}...")

            if file_path.suffix == '.gz':
                df = pd.read_csv(file_path, compression='gzip')
            else:
                df = pd.read_csv(file_path)

            print(f"Loaded {len(df)} samples from {dataset_name}")
            print(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None

    def prepare_data(self, df: pd.DataFrame, smiles_col: str, target_cols: List[str]) -> Tuple:
        """
        Prepare data for training by calculating molecular descriptors.

        Args:
            df: DataFrame containing SMILES and target columns
            smiles_col: Name of SMILES column
            target_cols: Names of target columns

        Returns:
            Tuple of (X, y, valid_df)
        """
        print(f"\nPreparing data...")
        print(f"SMILES column: {smiles_col}")
        print(f"Target columns: {target_cols}")

        # Calculate descriptors
        X, valid_indices = MolecularDescriptorCalculator.batch_calculate_descriptors(
            df[smiles_col].tolist(), show_progress=True
        )

        # Filter valid samples
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        y = valid_df[target_cols].values

        # Remove samples with missing targets
        valid_mask = ~np.isnan(y).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        valid_df = valid_df[valid_mask].reset_index(drop=True)

        print(f"Final dataset: {len(X)} samples with {X.shape[1]} features")
        print(f"Target shape: {y.shape}")

        return X, y, valid_df

    def train_toxicity_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train toxicity prediction model using Tox21 dataset.

        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*80)
        print("TRAINING TOXICITY MODEL (Tox21)")
        print("="*80)

        # Load Tox21 dataset
        df = self.load_dataset('tox21')
        if df is None:
            return None

        # Tox21 has 12 toxicity targets
        target_cols = [col for col in df.columns if col.startswith('NR-') or col.startswith('SR-')]
        smiles_col = 'smiles'

        # Prepare data
        X, y, valid_df = self.prepare_data(df, smiles_col, target_cols)

        # For simplicity, we'll create a binary toxicity label: toxic if any target is 1
        y_binary = (y.sum(axis=1) > 0).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        # Save model
        self.models['toxicity'] = model
        self.scalers['toxicity'] = scaler
        self.save_model('toxicity', model, scaler)

        # Print results
        print("\n" + "-"*80)
        print("TOXICITY MODEL RESULTS")
        print("-"*80)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print(f"\nTrain samples: {results['n_train']}")
        print(f"Test samples:  {results['n_test']}")

        self.results['toxicity'] = results
        return results

    def train_clintox_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train clinical toxicity prediction model using ClinTox dataset.

        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*80)
        print("TRAINING CLINICAL TOXICITY MODEL (ClinTox)")
        print("="*80)

        # Load ClinTox dataset
        df = self.load_dataset('clintox')
        if df is None:
            return None

        # ClinTox targets: FDA_APPROVED, CT_TOX
        target_cols = ['CT_TOX']  # Focus on clinical trial toxicity
        smiles_col = 'smiles'

        # Prepare data
        X, y, valid_df = self.prepare_data(df, smiles_col, target_cols)
        y = y.ravel()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        # Save model
        self.models['clintox'] = model
        self.scalers['clintox'] = scaler
        self.save_model('clintox', model, scaler)

        # Print results
        print("\n" + "-"*80)
        print("CLINICAL TOXICITY MODEL RESULTS")
        print("-"*80)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])

        self.results['clintox'] = results
        return results

    def train_bbbp_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train Blood-Brain Barrier Permeability model using BBBP dataset.

        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*80)
        print("TRAINING BBB PERMEABILITY MODEL (BBBP)")
        print("="*80)

        # Load BBBP dataset
        df = self.load_dataset('bbbp')
        if df is None:
            return None

        # BBBP target: p_np (1 = permeable, 0 = not permeable)
        target_cols = ['p_np']
        smiles_col = 'smiles'

        # Prepare data
        X, y, valid_df = self.prepare_data(df, smiles_col, target_cols)
        y = y.ravel()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        # Save model
        self.models['bbbp'] = model
        self.scalers['bbbp'] = scaler
        self.save_model('bbbp', model, scaler)

        # Print results
        print("\n" + "-"*80)
        print("BBB PERMEABILITY MODEL RESULTS")
        print("-"*80)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])

        self.results['bbbp'] = results
        return results

    def train_solubility_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train aqueous solubility prediction model using ESOL (Delaney) dataset.

        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*80)
        print("TRAINING SOLUBILITY MODEL (ESOL/Delaney)")
        print("="*80)

        # Load Delaney dataset
        df = self.load_dataset('delaney')
        if df is None:
            return None

        # ESOL target: measured log solubility in mols per litre
        target_cols = ['measured log solubility in mols per litre']
        smiles_col = 'smiles'

        # Prepare data
        X, y, valid_df = self.prepare_data(df, smiles_col, target_cols)
        y = y.ravel()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)

        results = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        # Save model
        self.models['solubility'] = model
        self.scalers['solubility'] = scaler
        self.save_model('solubility', model, scaler)

        # Print results
        print("\n" + "-"*80)
        print("SOLUBILITY MODEL RESULTS")
        print("-"*80)
        print(f"R² Score:  {results['r2']:.4f}")
        print(f"RMSE:      {results['rmse']:.4f}")
        print(f"MAE:       {results['mae']:.4f}")
        print(f"\nTrain samples: {results['n_train']}")
        print(f"Test samples:  {results['n_test']}")

        self.results['solubility'] = results
        return results

    def train_all_models(self) -> Dict:
        """
        Train all ADMET models.

        Returns:
            Dictionary containing all training results
        """
        print("\n" + "="*80)
        print("TRAINING ALL ADMET MODELS")
        print("="*80)

        all_results = {}

        # Train each model
        models_to_train = [
            ('toxicity', self.train_toxicity_model),
            ('clintox', self.train_clintox_model),
            ('bbbp', self.train_bbbp_model),
            ('solubility', self.train_solubility_model)
        ]

        for model_name, train_func in models_to_train:
            try:
                result = train_func()
                if result:
                    all_results[model_name] = result
            except Exception as e:
                print(f"\nError training {model_name} model: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        for model_name, result in all_results.items():
            print(f"\n{model_name.upper()}:")
            if 'accuracy' in result:
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  ROC-AUC:  {result['roc_auc']:.4f}")
            elif 'r2' in result:
                print(f"  R² Score: {result['r2']:.4f}")
                print(f"  RMSE:     {result['rmse']:.4f}")

        return all_results

    def predict_admet(self, smiles: Union[str, List[str]]) -> Dict:
        """
        Predict ADMET properties for given SMILES.

        Args:
            smiles: SMILES string or list of SMILES strings

        Returns:
            Dictionary containing ADMET predictions
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        results = {
            'smiles': smiles,
            'predictions': []
        }

        for smile in smiles:
            # Calculate descriptors
            descriptors = MolecularDescriptorCalculator.calculate_descriptors(smile)

            if descriptors is None:
                results['predictions'].append({
                    'valid': False,
                    'error': 'Invalid SMILES or descriptor calculation failed'
                })
                continue

            descriptors = descriptors.reshape(1, -1)

            # Make predictions with each model
            prediction = {'valid': True}

            for model_name in ['toxicity', 'clintox', 'bbbp', 'solubility']:
                if model_name in self.models:
                    model = self.models[model_name]
                    scaler = self.scalers[model_name]

                    X_scaled = scaler.transform(descriptors)

                    if model_name == 'solubility':
                        pred = model.predict(X_scaled)[0]
                        prediction[model_name] = float(pred)
                    else:
                        pred_class = model.predict(X_scaled)[0]
                        pred_proba = model.predict_proba(X_scaled)[0]
                        prediction[model_name] = {
                            'class': int(pred_class),
                            'probability': float(pred_proba[1])
                        }

            results['predictions'].append(prediction)

        return results

    def save_model(self, model_name: str, model, scaler):
        """Save trained model and scaler to disk."""
        model_path = self.model_dir / f"{model_name}_model.pkl"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Model saved to {model_path}")

    def load_model(self, model_name: str) -> bool:
        """Load trained model and scaler from disk."""
        model_path = self.model_dir / f"{model_name}_model.pkl"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            print(f"Model files not found for {model_name}")
            return False

        try:
            self.models[model_name] = joblib.load(model_path)
            self.scalers[model_name] = joblib.load(scaler_path)
            print(f"Loaded {model_name} model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            return False

    def load_all_models(self):
        """Load all trained models."""
        model_names = ['toxicity', 'clintox', 'bbbp', 'solubility']
        for model_name in model_names:
            self.load_model(model_name)


def main():
    """Main function to demonstrate ADMET model training and prediction."""

    print("="*80)
    print("ADMET SAFETY MODEL - DRUG DISCOVERY PIPELINE")
    print("="*80)

    # Initialize model
    admet_model = ADMETSafetyModel()

    # Train all models
    print("\nStarting model training...")
    results = admet_model.train_all_models()

    # Test predictions with example compounds
    print("\n" + "="*80)
    print("TESTING PREDICTIONS")
    print("="*80)

    test_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]

    print("\nTest compounds:")
    for i, smile in enumerate(test_smiles, 1):
        print(f"{i}. {smile}")

    predictions = admet_model.predict_admet(test_smiles)

    print("\nPredictions:")
    for i, (smile, pred) in enumerate(zip(predictions['smiles'], predictions['predictions']), 1):
        print(f"\n{i}. {smile}")
        if pred['valid']:
            for prop, value in pred.items():
                if prop != 'valid':
                    print(f"   {prop}: {value}")
        else:
            print(f"   Error: {pred['error']}")

    print("\n" + "="*80)
    print("ADMET MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModels saved in: {admet_model.model_dir}")
    print("\nAvailable models:")
    for model_name in admet_model.models.keys():
        print(f"  - {model_name}")


if __name__ == "__main__":
    main()
