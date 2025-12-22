"""
Centralized validation logic following Single Responsibility Principle
"""

from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """Abstract base validator"""
    
    @abstractmethod
    def validate(self, value: any) -> Tuple[bool, str]:
        """
        Validate value
        
        Returns:
            Tuple of (is_valid, message)
        """
        pass


class SMILESValidator(BaseValidator):
    """Validator for SMILES strings"""
    
    def __init__(self, max_length: int = 500, use_rdkit: bool = True):
        """
        Initialize SMILES validator
        
        Args:
            max_length: Maximum SMILES length
            use_rdkit: Whether to use RDKit for validation
        """
        self.max_length = max_length
        self.use_rdkit = use_rdkit
        self._rdkit_available = self._check_rdkit()
    
    def _check_rdkit(self) -> bool:
        """Check if RDKit is available"""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False
    
    def validate(self, smiles: str) -> Tuple[bool, str]:
        """Validate SMILES string"""
        # Basic checks
        if not smiles or len(smiles.strip()) == 0:
            return False, "SMILES string cannot be empty"
        
        if len(smiles) > self.max_length:
            return False, f"SMILES string too long (max {self.max_length} characters)"
        
        # Character validation
        valid_chars = set("CNOPSFIBrClcnops0123456789[]()=#@+-\\/")
        invalid_chars = set(smiles) - valid_chars
        if invalid_chars:
            return False, f"SMILES contains invalid characters: {invalid_chars}"
        
        # Bracket balance check
        if smiles.count('[') != smiles.count(']'):
            return False, "Unbalanced square brackets"
        
        if smiles.count('(') != smiles.count(')'):
            return False, "Unbalanced parentheses"
        
        # RDKit validation
        if self.use_rdkit and self._rdkit_available:
            is_valid, message = self._validate_with_rdkit(smiles)
            if not is_valid:
                return is_valid, message
        
        return True, "Valid SMILES"
    
    def _validate_with_rdkit(self, smiles: str) -> Tuple[bool, str]:
        """Validate using RDKit"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES: RDKit could not parse"
            return True, "Valid SMILES (RDKit verified)"
        except Exception as e:
            return False, f"RDKit error: {str(e)}"
    
    def canonicalize(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form"""
        if not self._rdkit_available:
            return smiles
        
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def batch_validate(self, smiles_list: List[str]) -> List[Tuple[str, bool, str]]:
        """
        Validate batch of SMILES
        
        Returns:
            List of tuples (smiles, is_valid, message)
        """
        results = []
        for smiles in smiles_list:
            is_valid, message = self.validate(smiles)
            results.append((smiles, is_valid, message))
        return results


class DataFrameValidator(BaseValidator):
    """Validator for DataFrame inputs"""
    
    def __init__(self, required_columns: List[str]):
        """
        Initialize DataFrame validator
        
        Args:
            required_columns: List of required column names
        """
        self.required_columns = required_columns
    
    def validate(self, df) -> Tuple[bool, str]:
        """Validate DataFrame"""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        return True, "Valid DataFrame"
    
    def find_smiles_column(self, df) -> Optional[str]:
        """Find SMILES column in DataFrame"""
        smiles_variants = ['smiles', 'smile', 'SMILES', 'SMILE', 'canonical_smiles']
        
        for col in df.columns:
            if col in smiles_variants or col.lower() in [v.lower() for v in smiles_variants]:
                return col
        
        return None


class ConfigValidator(BaseValidator):
    """Validator for configuration dictionaries"""
    
    def __init__(self, required_keys: List[str], optional_keys: Optional[List[str]] = None):
        """
        Initialize config validator
        
        Args:
            required_keys: Required configuration keys
            optional_keys: Optional configuration keys
        """
        self.required_keys = required_keys
        self.optional_keys = optional_keys or []
    
    def validate(self, config: dict) -> Tuple[bool, str]:
        """Validate configuration dictionary"""
        if not isinstance(config, dict):
            return False, "Config must be a dictionary"
        
        # Check required keys
        missing_keys = set(self.required_keys) - set(config.keys())
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Check for unexpected keys
        all_valid_keys = set(self.required_keys + self.optional_keys)
        unexpected_keys = set(config.keys()) - all_valid_keys
        if unexpected_keys:
            return False, f"Unexpected keys: {unexpected_keys}"
        
        return True, "Valid configuration"


# Singleton validator instances for reuse
_smiles_validator = None
_df_validator = None


def get_smiles_validator(**kwargs) -> SMILESValidator:
    """Get singleton SMILES validator"""
    global _smiles_validator
    if _smiles_validator is None:
        _smiles_validator = SMILESValidator(**kwargs)
    return _smiles_validator


def get_dataframe_validator(required_columns: List[str]) -> DataFrameValidator:
    """Get DataFrame validator"""
    return DataFrameValidator(required_columns)
