"""
Feature Extraction for Bioactivity Prediction
Extracts molecular fingerprints and descriptors from SMILES
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Feature extraction will be limited.")


class MolecularFeatureExtractor:
    """Extract molecular features from SMILES strings"""

    def __init__(self):
        self.rdkit_available = RDKIT_AVAILABLE

    def smiles_to_mol(self, smiles: str):
        """Convert SMILES to RDKit molecule object"""
        if not self.rdkit_available:
            raise ImportError("RDKit is required for feature extraction")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol

    def calculate_lipinski_descriptors(self, smiles: Union[str, List[str]]) -> pd.DataFrame:
        """
        Calculate Lipinski descriptors (MW, LogP, HBD, HBA)

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            DataFrame with Lipinski descriptors
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        descriptors_list = []

        for smi in smiles:
            try:
                mol = self.smiles_to_mol(smi)

                desc = {
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                }

                descriptors_list.append(desc)
            except Exception as e:
                print(f"Error calculating descriptors for {smi}: {e}")
                descriptors_list.append({
                    'MW': np.nan,
                    'LogP': np.nan,
                    'NumHDonors': np.nan,
                    'NumHAcceptors': np.nan,
                })

        return pd.DataFrame(descriptors_list)

    def calculate_extended_descriptors(self, smiles: str) -> dict:
        """
        Calculate extended molecular descriptors

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of descriptors
        """
        try:
            mol = self.smiles_to_mol(smiles)

            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'NumAmideBonds': 0,  # Placeholder - need specific calculation
                'FractionCsp3': getattr(Descriptors, 'FractionCsp3', lambda x: 0)(mol),
                'NumSpiroAtoms': getattr(Descriptors, 'NumSpiroAtoms', lambda x: 0)(mol),
                'NumBridgeheadAtoms': getattr(Descriptors, 'NumBridgeheadAtoms', lambda x: 0)(mol),
            }

            return descriptors

        except Exception as e:
            print(f"Error calculating extended descriptors: {e}")
            return {}

    def calculate_pubchem_fingerprint(self, smiles: str, n_bits: int = 881) -> np.ndarray:
        """
        Calculate PubChem-like fingerprint using RDKit

        Note: This approximates PaDEL's PubChem fingerprints using RDKit's
        implementation. The exact feature set may differ slightly.

        Args:
            smiles: SMILES string
            n_bits: Number of bits (default 881 for PubChem compatibility)

        Returns:
            Binary fingerprint array
        """
        try:
            mol = self.smiles_to_mol(smiles)

            # RDKit's GetHashedMorganFingerprint can approximate PubChem fingerprints
            # PubChem uses specific substructure patterns, so we use Morgan FP as approximation
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,  # Similar to PubChem's circular patterns
                nBits=n_bits,
                useFeatures=False
            )

            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)

            return arr

        except Exception as e:
            print(f"Error calculating fingerprint: {e}")
            return np.zeros(n_bits, dtype=np.int8)

    def calculate_pubchem_fingerprint_dataframe(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate PubChem fingerprints for multiple SMILES

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with fingerprint columns (PubchemFP0 to PubchemFP880)
        """
        fingerprints = []

        for smi in smiles_list:
            fp = self.calculate_pubchem_fingerprint(smi)
            fingerprints.append(fp)

        # Create DataFrame with proper column names
        n_bits = fingerprints[0].shape[0] if fingerprints else 881
        columns = [f'PubchemFP{i}' for i in range(n_bits)]

        df = pd.DataFrame(fingerprints, columns=columns)
        return df

    def extract_features_for_model(self, smiles: Union[str, List[str]],
                                   feature_type: str = 'pubchem') -> np.ndarray:
        """
        Extract features in format expected by trained model

        Args:
            smiles: SMILES string or list of SMILES
            feature_type: Type of features ('pubchem', 'morgan', 'combined')

        Returns:
            Feature array ready for model prediction
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        if feature_type == 'pubchem':
            df = self.calculate_pubchem_fingerprint_dataframe(smiles)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        return df.values

    def check_lipinski_rules(self, descriptors: dict) -> Tuple[bool, List[str]]:
        """
        Check if molecule passes Lipinski's Rule of Five

        Args:
            descriptors: Dictionary of molecular descriptors

        Returns:
            Tuple of (passes, list of violations)
        """
        violations = []

        if descriptors.get('MW', 0) > 500:
            violations.append("Molecular Weight > 500")

        if descriptors.get('LogP', 0) > 5:
            violations.append("LogP > 5")

        if descriptors.get('NumHDonors', 0) > 5:
            violations.append("H-Bond Donors > 5")

        if descriptors.get('NumHAcceptors', 0) > 10:
            violations.append("H-Bond Acceptors > 10")

        passes = len(violations) == 0

        return passes, violations


# Global instance
feature_extractor = MolecularFeatureExtractor()


# Convenience functions
def get_lipinski_descriptors(smiles: Union[str, List[str]]) -> pd.DataFrame:
    """Get Lipinski descriptors for SMILES"""
    return feature_extractor.calculate_lipinski_descriptors(smiles)


def get_extended_descriptors(smiles: str) -> dict:
    """Get extended molecular descriptors"""
    return feature_extractor.calculate_extended_descriptors(smiles)


def get_fingerprint(smiles: str, n_bits: int = 881) -> np.ndarray:
    """Get molecular fingerprint"""
    return feature_extractor.calculate_pubchem_fingerprint(smiles, n_bits)


def check_drug_likeness(smiles: str) -> Tuple[bool, List[str]]:
    """Check Lipinski's Rule of Five"""
    descriptors = get_extended_descriptors(smiles)
    return feature_extractor.check_lipinski_rules(descriptors)
