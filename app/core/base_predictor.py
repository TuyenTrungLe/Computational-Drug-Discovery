"""
Base Predictor interface for all prediction models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional
import numpy as np
from pathlib import Path


class BasePredictor(ABC):
    """
    Abstract base class for all predictors
    Defines common interface for model loading and prediction
    """
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize predictor
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.is_loaded = False
        
    @abstractmethod
    def _load_models(self):
        """Load trained models from disk - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict(self, smiles: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Make predictions for given SMILES
        
        Args:
            smiles: SMILES string or list of SMILES
            
        Returns:
            Dictionary with predictions and metadata
        """
        pass
    
    def is_available(self) -> bool:
        """Check if predictor is available and loaded"""
        return self.is_loaded and len(self.models) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'loaded': self.is_loaded,
            'models_dir': str(self.models_dir),
            'num_models': len(self.models),
            'model_names': list(self.models.keys())
        }
    
    def _validate_smiles_input(self, smiles: Union[str, List[str]]) -> tuple[List[str], bool]:
        """
        Validate and normalize SMILES input
        
        Args:
            smiles: SMILES string or list
            
        Returns:
            Tuple of (smiles_list, is_single_molecule)
        """
        is_single = isinstance(smiles, str)
        smiles_list = [smiles] if is_single else smiles
        
        if not smiles_list:
            raise ValueError("SMILES list cannot be empty")
        
        return smiles_list, is_single
    
    def _get_fallback_prediction(self, smiles: str) -> Dict[str, Any]:
        """
        Get fallback prediction when model fails
        
        Args:
            smiles: SMILES string
            
        Returns:
            Fallback prediction dictionary
        """
        return {
            'smiles': smiles,
            'valid': False,
            'error': 'Prediction failed - model not available'
        }


class PredictorFactory:
    """
    Factory for creating predictors
    Implements factory pattern for predictor instantiation
    """
    
    _predictors = {}
    
    @classmethod
    def register_predictor(cls, name: str, predictor_class: type):
        """Register a predictor class"""
        cls._predictors[name] = predictor_class
    
    @classmethod
    def create_predictor(cls, name: str, **kwargs) -> BasePredictor:
        """
        Create predictor instance
        
        Args:
            name: Predictor name
            **kwargs: Arguments for predictor constructor
            
        Returns:
            Predictor instance
        """
        if name not in cls._predictors:
            raise ValueError(f"Unknown predictor: {name}")
        
        predictor_class = cls._predictors[name]
        return predictor_class(**kwargs)
    
    @classmethod
    def list_predictors(cls) -> List[str]:
        """List all registered predictors"""
        return list(cls._predictors.keys())
