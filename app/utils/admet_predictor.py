"""
ADMET Property Prediction using trained models
Handles loading and prediction for multiple ADMET properties
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import sys
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ADMET model
try:
    from src.models.admet_safety_model import MolecularDescriptorCalculator, ADMETSafetyModel
except ImportError:
    print("[WARNING] Could not import ADMET model from src.models")
    MolecularDescriptorCalculator = None
    ADMETSafetyModel = None


class ADMETPredictor:
    """
    ADMET property prediction using trained Random Forest models

    Models:
        - Toxicity (Tox21): General toxicity prediction
        - Clinical Toxicity (ClinTox): Clinical trial toxicity
        - BBB Permeability (BBBP): Blood-Brain Barrier penetration
        - Solubility (ESOL): Aqueous solubility
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize ADMET predictor

        Args:
            models_dir: Directory containing trained ADMET models
        """
        if models_dir is None:
            models_dir = project_root / "models" / "admet_models"

        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.descriptor_calculator = MolecularDescriptorCalculator if MolecularDescriptorCalculator else None

        # Load models
        self._load_models()

    def _load_models(self):
        """Load trained ADMET models from disk"""
        model_names = ['toxicity', 'clintox', 'bbbp', 'solubility']

        for model_name in model_names:
            try:
                model_path = self.models_dir / f"{model_name}_model.pkl"
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"

                if model_path.exists() and scaler_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    print(f"[OK] Loaded {model_name} model from {model_path}")
                else:
                    print(f"[WARNING] Model files not found: {model_path}")

            except Exception as e:
                print(f"[WARNING] Error loading {model_name} model: {e}")

    def _calculate_descriptors(self, smiles: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Calculate molecular descriptors for SMILES

        Args:
            smiles: SMILES string or list of SMILES

        Returns:
            Array of molecular descriptors (n_samples, 520)
        """
        if self.descriptor_calculator is None:
            print("[ERROR] Descriptor calculator not available")
            return None

        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = smiles

        descriptors = []
        valid_indices = []

        for idx, smile in enumerate(smiles_list):
            desc = self.descriptor_calculator.calculate_descriptors(smile)
            if desc is not None:
                descriptors.append(desc)
                valid_indices.append(idx)

        if len(descriptors) == 0:
            return None, []

        return np.array(descriptors), valid_indices

    def predict_admet(self, smiles: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Predict ADMET properties for given SMILES

        Args:
            smiles: SMILES string or list of SMILES

        Returns:
            Dictionary with ADMET predictions
        """
        single_molecule = isinstance(smiles, str)
        if single_molecule:
            smiles_list = [smiles]
        else:
            smiles_list = smiles

        # Calculate molecular descriptors
        descriptors, valid_indices = self._calculate_descriptors(smiles_list)

        if descriptors is None:
            # Return None predictions for invalid molecules
            return self._get_fallback_predictions(smiles_list)

        results = {
            'smiles': smiles_list,
            'predictions': []
        }

        # Make predictions for each compound
        for i, smile in enumerate(smiles_list):
            if i not in valid_indices:
                # Invalid SMILES
                results['predictions'].append({
                    'valid': False,
                    'error': 'Invalid SMILES or descriptor calculation failed'
                })
                continue

            desc_idx = valid_indices.index(i)
            descriptors_single = descriptors[desc_idx:desc_idx+1]

            prediction = {'valid': True}

            # Toxicity prediction
            if 'toxicity' in self.models:
                X_scaled = self.scalers['toxicity'].transform(descriptors_single)
                pred_class = self.models['toxicity'].predict(X_scaled)[0]
                pred_proba = self.models['toxicity'].predict_proba(X_scaled)[0]
                prediction['toxicity'] = {
                    'class': int(pred_class),
                    'probability': float(pred_proba[1]),
                    'label': 'Toxic' if pred_class == 1 else 'Non-toxic'
                }

            # Clinical toxicity prediction
            if 'clintox' in self.models:
                X_scaled = self.scalers['clintox'].transform(descriptors_single)
                pred_class = self.models['clintox'].predict(X_scaled)[0]
                pred_proba = self.models['clintox'].predict_proba(X_scaled)[0]
                prediction['clintox'] = {
                    'class': int(pred_class),
                    'probability': float(pred_proba[1]),
                    'label': 'Toxic' if pred_class == 1 else 'Non-toxic'
                }

            # BBB permeability prediction
            if 'bbbp' in self.models:
                X_scaled = self.scalers['bbbp'].transform(descriptors_single)
                pred_class = self.models['bbbp'].predict(X_scaled)[0]
                pred_proba = self.models['bbbp'].predict_proba(X_scaled)[0]
                prediction['bbbp'] = {
                    'class': int(pred_class),
                    'probability': float(pred_proba[1]),
                    'label': 'Permeable' if pred_class == 1 else 'Not Permeable'
                }

            # Solubility prediction
            if 'solubility' in self.models:
                X_scaled = self.scalers['solubility'].transform(descriptors_single)
                pred = self.models['solubility'].predict(X_scaled)[0]
                prediction['solubility'] = {
                    'logS': float(pred),
                    'label': self._classify_solubility(pred)
                }

            results['predictions'].append(prediction)

        if single_molecule:
            return results['predictions'][0]
        else:
            return results

    def predict_batch_df(self, df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
        """
        Predict ADMET properties for a DataFrame of compounds

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with ADMET predictions added
        """
        results = df.copy()

        smiles_list = df[smiles_col].tolist()
        predictions = self.predict_admet(smiles_list)

        if 'predictions' in predictions:
            preds = predictions['predictions']
        else:
            preds = [predictions]  # Single molecule

        # Extract predictions into DataFrame columns
        tox21_prob = []
        tox21_class = []
        clintox_prob = []
        clintox_class = []
        bbbp_prob = []
        bbbp_class = []
        solubility_logS = []

        for pred in preds:
            if pred.get('valid', True):
                # Toxicity
                if 'toxicity' in pred:
                    tox21_prob.append(pred['toxicity']['probability'])
                    tox21_class.append(pred['toxicity']['label'])
                else:
                    tox21_prob.append(np.nan)
                    tox21_class.append('Unknown')

                # Clinical toxicity
                if 'clintox' in pred:
                    clintox_prob.append(pred['clintox']['probability'])
                    clintox_class.append(pred['clintox']['label'])
                else:
                    clintox_prob.append(np.nan)
                    clintox_class.append('Unknown')

                # BBBP
                if 'bbbp' in pred:
                    bbbp_prob.append(pred['bbbp']['probability'])
                    bbbp_class.append(pred['bbbp']['label'])
                else:
                    bbbp_prob.append(np.nan)
                    bbbp_class.append('Unknown')

                # Solubility
                if 'solubility' in pred:
                    solubility_logS.append(pred['solubility']['logS'])
                else:
                    solubility_logS.append(np.nan)
            else:
                # Invalid SMILES
                tox21_prob.append(np.nan)
                tox21_class.append('Invalid')
                clintox_prob.append(np.nan)
                clintox_class.append('Invalid')
                bbbp_prob.append(np.nan)
                bbbp_class.append('Invalid')
                solubility_logS.append(np.nan)

        # Add to DataFrame
        results['tox21_prob'] = tox21_prob
        results['tox21_class'] = tox21_class
        results['clintox_prob'] = clintox_prob
        results['clintox_class'] = clintox_class
        results['bbbp_prob'] = bbbp_prob
        results['bbbp_class'] = bbbp_class
        results['logS'] = solubility_logS

        # Calculate Lipinski properties
        results = self._add_lipinski_properties(results, smiles_col)

        # Add pass/fail flags
        results['tox21_pass'] = results['tox21_prob'] < 0.5
        results['clintox_pass'] = results['clintox_prob'] < 0.5
        results['solubility_pass'] = (results['logS'] >= -6) & (results['logS'] <= 0)
        results['bbbp_pass'] = True  # Can be adjusted based on requirements

        # Overall score
        results['overall_score'] = (
            results['tox21_pass'].astype(int) * 0.3 +
            results['clintox_pass'].astype(int) * 0.2 +
            results['solubility_pass'].astype(int) * 0.3 +
            results['lipinski_pass'].astype(int) * 0.2
        )

        return results

    def _add_lipinski_properties(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Add Lipinski Rule of Five properties"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        MW = []
        LogP = []
        HBD = []
        HBA = []

        for smiles in df[smiles_col]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    MW.append(Descriptors.MolWt(mol))
                    LogP.append(Descriptors.MolLogP(mol))
                    HBD.append(Descriptors.NumHDonors(mol))
                    HBA.append(Descriptors.NumHAcceptors(mol))
                else:
                    MW.append(np.nan)
                    LogP.append(np.nan)
                    HBD.append(np.nan)
                    HBA.append(np.nan)
            except:
                MW.append(np.nan)
                LogP.append(np.nan)
                HBD.append(np.nan)
                HBA.append(np.nan)

        df['MW'] = MW
        df['LogP'] = LogP
        df['HBD'] = HBD
        df['HBA'] = HBA

        # Lipinski pass/fail
        df['lipinski_pass'] = (
            (df['MW'] <= 500) &
            (df['LogP'] <= 5) &
            (df['HBD'] <= 5) &
            (df['HBA'] <= 10)
        )

        return df

    def _classify_solubility(self, logS: float) -> str:
        """Classify solubility based on logS value"""
        if logS > -2:
            return 'Highly soluble'
        elif logS > -4:
            return 'Soluble'
        elif logS > -6:
            return 'Moderately soluble'
        else:
            return 'Poorly soluble'

    def _get_fallback_predictions(self, smiles_list: List[str]) -> Dict:
        """Return fallback predictions for invalid SMILES"""
        return {
            'smiles': smiles_list,
            'predictions': [{
                'valid': False,
                'error': 'Invalid SMILES'
            }] * len(smiles_list)
        }

    def is_models_loaded(self) -> bool:
        """Check if models are loaded"""
        return len(self.models) > 0


# Global predictor instance
try:
    _admet_predictor = ADMETPredictor()
    print("[OK] ADMET Predictor initialized successfully")
except Exception as e:
    print(f"[WARNING] Could not initialize ADMET predictor: {e}")
    _admet_predictor = None


def predict_admet(smiles: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Convenience function for ADMET prediction

    Args:
        smiles: SMILES string or list

    Returns:
        ADMET predictions
    """
    if _admet_predictor is None:
        raise RuntimeError("ADMET Predictor not initialized")

    return _admet_predictor.predict_admet(smiles)


def predict_batch_df(df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
    """
    Convenience function for batch ADMET prediction

    Args:
        df: DataFrame with SMILES
        smiles_col: SMILES column name

    Returns:
        DataFrame with predictions
    """
    if _admet_predictor is None:
        raise RuntimeError("ADMET Predictor not initialized")

    return _admet_predictor.predict_batch_df(df, smiles_col)


def is_admet_available() -> bool:
    """Check if ADMET models are available"""
    return _admet_predictor is not None and _admet_predictor.is_models_loaded()
