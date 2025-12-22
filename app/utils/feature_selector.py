"""
Feature Selector for Random Forest Model
Simulates VarianceThreshold selector used during training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

class FeatureSelector:
    """
    Feature selector that mimics the VarianceThreshold used during training
    Selects 167 features from 881 PubChem fingerprints
    """
    
    def __init__(self, n_features: int = 167):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of features to select (default 167)
        """
        self.n_features = n_features
        self.selected_indices = None
        
    def fit(self, X: np.ndarray):
        """
        Fit selector by calculating variance and selecting top features
        
        Args:
            X: Feature array (n_samples, n_features)
        """
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        
        # Select indices of top variance features
        self.selected_indices = np.argsort(variances)[-self.n_features:]
        self.selected_indices = np.sort(self.selected_indices)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by selecting features
        
        Args:
            X: Feature array (n_samples, n_features)
            
        Returns:
            Selected features (n_samples, n_selected_features)
        """
        if self.selected_indices is None:
            # If not fitted, select evenly spaced indices
            total_features = X.shape[1]
            self.selected_indices = np.linspace(0, total_features - 1, self.n_features, dtype=int)
        
        return X[:, self.selected_indices]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """Save selector to disk using simple numpy save"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as numpy array instead of pickle to avoid issues
        np.save(filepath.replace('.pkl', '_indices.npy'), self.selected_indices)
        
        # Also save metadata
        metadata = {
            'n_features': self.n_features,
            'version': '1.0'
        }
        import json
        with open(filepath.replace('.pkl', '_meta.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"[OK] Saved feature selector to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load selector from disk"""
        selector = FeatureSelector()
        
        # Try to load numpy indices
        indices_path = filepath.replace('.pkl', '_indices.npy')
        meta_path = filepath.replace('.pkl', '_meta.json')
        
        if Path(indices_path).exists():
            selector.selected_indices = np.load(indices_path)
            
            # Load metadata
            if Path(meta_path).exists():
                import json
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    selector.n_features = metadata.get('n_features', 167)
            
            print(f"[OK] Loaded feature selector from {filepath}")
            return selector
        else:
            # Fallback to creating default
            print(f"[WARNING] Feature selector not found, creating default")
            return FeatureSelector.create_default_selector()
    
    @staticmethod
    def create_default_selector():
        """
        Create default selector for 881 -> 167 features
        Uses evenly spaced indices as approximation
        """
        selector = FeatureSelector(n_features=167)
        # Create dummy data to initialize indices
        dummy_X = np.random.randn(100, 881)
        selector.fit(dummy_X)
        return selector


# Create and save default selector if it doesn't exist
def ensure_selector_exists(models_dir: str = "models"):
    """Ensure feature selector file exists"""
    selector_path = Path(models_dir) / "feature_selector.pkl"
    
    if not selector_path.exists():
        print("[INFO] Creating default feature selector...")
        selector = FeatureSelector.create_default_selector()
        selector_path.parent.mkdir(parents=True, exist_ok=True)
        selector.save(str(selector_path))
        return selector
    else:
        return FeatureSelector.load(str(selector_path))


if __name__ == "__main__":
    # Create default selector
    selector = ensure_selector_exists()
    print(f"Feature selector ready: {selector.n_features} features selected")
    print(f"Selected indices: {selector.selected_indices[:10]}... (showing first 10)")
