"""
SMILES utilities for validation and processing
"""

import re
from typing import Tuple, List


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate SMILES string format

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not smiles or len(smiles.strip()) == 0:
        return False, "SMILES string cannot be empty"

    # Basic validation - check for valid characters
    valid_chars = set("CNOPSFIBrClcnops0123456789[]()=#@+-\\/")
    if not all(c in valid_chars for c in smiles):
        invalid_chars = set(smiles) - valid_chars
        return False, f"SMILES contains invalid characters: {invalid_chars}"

    if len(smiles) > 500:
        return False, "SMILES string too long (max 500 characters)"

    # Check for balanced brackets
    if smiles.count('[') != smiles.count(']'):
        return False, "Unbalanced square brackets"

    if smiles.count('(') != smiles.count(')'):
        return False, "Unbalanced parentheses"

    return True, "Valid SMILES"


def validate_smiles_rdkit(smiles: str) -> Tuple[bool, str]:
    """
    Validate SMILES using RDKit (if available)

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES: RDKit could not parse"
        return True, "Valid SMILES (RDKit)"
    except ImportError:
        return validate_smiles(smiles)
    except Exception as e:
        return False, f"RDKit error: {str(e)}"


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize SMILES string using RDKit

    Args:
        smiles: SMILES string

    Returns:
        Canonical SMILES string
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return smiles


def batch_validate_smiles(smiles_list: List[str]) -> List[Tuple[str, bool, str]]:
    """
    Validate a batch of SMILES strings

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of tuples (smiles, is_valid, message)
    """
    results = []
    for smiles in smiles_list:
        is_valid, message = validate_smiles_rdkit(smiles)
        results.append((smiles, is_valid, message))
    return results
