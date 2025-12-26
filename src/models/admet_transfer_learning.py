"""
ADMET Transfer Learning Module
Uses pre-trained bioactivity LSTM model embeddings to enhance ADMET predictions

This module implements transfer learning by:
1. Loading the pre-trained LSTM model (trained on bioactivity data)
2. Extracting learned embeddings from the LSTM encoder
3. Combining these embeddings with RDKit molecular descriptors
4. Training enhanced ADMET models with the combined features

This leverages knowledge from the bioactivity task to improve ADMET predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import joblib

warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("[WARNING] TensorFlow not available. Transfer learning features will be limited.")
    TF_AVAILABLE = False

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    print("[WARNING] RDKit not available.")
    RDKIT_AVAILABLE = False

# Import existing ADMET model
try:
    from src.models.admet_safety_model import MolecularDescriptorCalculator
except ImportError:
    print("[WARNING] Could not import MolecularDescriptorCalculator")
    MolecularDescriptorCalculator = None

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
)


class LSTMEmbeddingExtractor:
    """
    Extracts molecular embeddings from pre-trained LSTM model
    """

    def __init__(self, lstm_model_path: Union[str, Path], tokenizer_path: Union[str, Path]):
        """
        Initialize embedding extractor

        Args:
            lstm_model_path: Path to trained LSTM model (.keras or .h5)
            tokenizer_path: Path to tokenizer (.pkl)
        """
        self.lstm_model_path = Path(lstm_model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.lstm_model = None
        self.tokenizer = None
        self.embedding_model = None
        self.embedding_dim = None

        self._load_model()

    def _load_model(self):
        """Load LSTM model and tokenizer"""
        if not TF_AVAILABLE:
            print("[ERROR] TensorFlow not available")
            return False

        try:
            # Load tokenizer
            if self.tokenizer_path.exists():
                self.tokenizer = joblib.load(self.tokenizer_path)
                print(f"[OK] Loaded tokenizer from {self.tokenizer_path}")
            else:
                print(f"[ERROR] Tokenizer not found: {self.tokenizer_path}")
                return False

            # Load LSTM model
            if self.lstm_model_path.exists():
                self.lstm_model = keras.models.load_model(self.lstm_model_path)
                print(f"[OK] Loaded LSTM model from {self.lstm_model_path}")

                # Create embedding extraction model
                # Extract embeddings from the last LSTM layer before the dense output
                self._create_embedding_model()
                return True
            else:
                print(f"[ERROR] LSTM model not found: {self.lstm_model_path}")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_embedding_model(self):
        """Create a model that extracts embeddings from LSTM"""
        try:
            # Build the model first by calling it with dummy data
            dummy_input = np.zeros((1, 100), dtype=np.int32)  # (batch_size=1, maxlen=100)
            _ = self.lstm_model(dummy_input)  # This builds the model

            # Find the last LSTM or GRU layer index
            embedding_layer_index = None

            for i, layer in enumerate(self.lstm_model.layers):
                if isinstance(layer, (keras.layers.LSTM, keras.layers.GRU, keras.layers.Bidirectional)):
                    embedding_layer_index = i

            if embedding_layer_index is None:
                print("[WARNING] No LSTM/GRU layer found, using last Dense layer before output")
                # Find last Dense layer before output
                for i in range(len(self.lstm_model.layers) - 2, -1, -1):
                    layer = self.lstm_model.layers[i]
                    if isinstance(layer, keras.layers.Dense):
                        embedding_layer_index = i
                        break

            if embedding_layer_index is not None:
                # Store the layer index for later use
                self.embedding_layer_index = embedding_layer_index

                # Test to get embedding dimension by extracting embeddings from dummy input
                test_output = self._extract_embeddings_from_layers(dummy_input)
                if test_output is not None:
                    self.embedding_dim = test_output.shape[-1]
                    # Mark that we have a working embedding extractor
                    self.embedding_model = True  # Just a flag
                    print(f"[OK] Created embedding extractor (dim={self.embedding_dim})")
                else:
                    print("[ERROR] Failed to extract test embeddings")
            else:
                print("[ERROR] Could not find suitable embedding layer")

        except Exception as e:
            print(f"[ERROR] Failed to create embedding model: {e}")
            import traceback
            traceback.print_exc()

    def _extract_embeddings_from_layers(self, sequences):
        """Extract embeddings by manually passing through layers"""
        try:
            # Pass input through layers up to embedding layer
            x = sequences
            for i, layer in enumerate(self.lstm_model.layers):
                x = layer(x)
                if i == self.embedding_layer_index:
                    return x.numpy()
            return None
        except Exception as e:
            print(f"[ERROR] Failed to extract embeddings: {e}")
            return None

    def smiles_to_sequence(self, smiles: str, maxlen: int = 100) -> Optional[np.ndarray]:
        """
        Convert SMILES to tokenized sequence

        Args:
            smiles: SMILES string
            maxlen: Maximum sequence length

        Returns:
            Tokenized sequence or None
        """
        if self.tokenizer is None:
            return None

        try:
            # Use Keras Tokenizer API
            # For character-level tokenization, Keras Tokenizer expects the string directly
            sequences = self.tokenizer.texts_to_sequences([smiles])

            if not sequences or len(sequences[0]) == 0:
                return None

            sequence = sequences[0]

            # Pad or truncate to maxlen
            if len(sequence) < maxlen:
                sequence = sequence + [0] * (maxlen - len(sequence))
            else:
                sequence = sequence[:maxlen]

            return np.array(sequence)
        except Exception as e:
            print(f"[ERROR] Failed to tokenize SMILES: {e}")
            return None

    def extract_embeddings(self, smiles: Union[str, List[str]], maxlen: int = 100) -> Optional[np.ndarray]:
        """
        Extract embeddings from SMILES

        Args:
            smiles: SMILES string or list of SMILES
            maxlen: Maximum sequence length

        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        if not hasattr(self, 'embedding_layer_index') or self.embedding_layer_index is None:
            print("[ERROR] Embedding model not initialized")
            return None, []

        # Handle single SMILES
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = smiles

        # Convert SMILES to sequences
        sequences = []
        valid_indices = []

        for idx, smi in enumerate(smiles_list):
            seq = self.smiles_to_sequence(smi, maxlen)
            if seq is not None:
                sequences.append(seq)
                valid_indices.append(idx)

        if len(sequences) == 0:
            return None, []

        # Convert to array
        sequences_array = np.array(sequences)

        # Extract embeddings using layer-by-layer approach
        try:
            embeddings = self._extract_embeddings_from_layers(sequences_array)
            if embeddings is not None:
                return embeddings, valid_indices
            else:
                return None, []
        except Exception as e:
            print(f"[ERROR] Failed to extract embeddings: {e}")
            return None, []


class ADMETTransferLearningModel:
    """
    ADMET prediction using transfer learning from bioactivity LSTM model

    Combines:
    - LSTM embeddings (learned from bioactivity task)
    - RDKit molecular descriptors (520 features)

    To create enhanced feature representations for ADMET prediction
    """

    def __init__(self,
                 lstm_model_path: Union[str, Path] = None,
                 tokenizer_path: Union[str, Path] = None,
                 models_dir: Union[str, Path] = None):
        """
        Initialize transfer learning model

        Args:
            lstm_model_path: Path to LSTM model
            tokenizer_path: Path to tokenizer
            models_dir: Directory to save/load models
        """
        # Set default paths
        if lstm_model_path is None:
            lstm_model_path = Path("models") / "lstm_pIC50_model.keras"
        if tokenizer_path is None:
            tokenizer_path = Path("models") / "tokenizer.pkl"
        if models_dir is None:
            models_dir = Path("models") / "admet_models"

        self.lstm_model_path = Path(lstm_model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedding_extractor = None
        self.descriptor_calculator = MolecularDescriptorCalculator() if MolecularDescriptorCalculator else None
        self.models = {}
        self.scalers = {}
        self.results = {}

        print(f"[INIT] ADMET Transfer Learning Model")
        print(f"  LSTM model: {self.lstm_model_path}")
        print(f"  Tokenizer: {self.tokenizer_path}")
        print(f"  Models dir: {self.models_dir}")

        # Load LSTM embedding extractor
        if TF_AVAILABLE and self.lstm_model_path.exists() and self.tokenizer_path.exists():
            self.embedding_extractor = LSTMEmbeddingExtractor(
                self.lstm_model_path,
                self.tokenizer_path
            )
        else:
            print("[WARNING] LSTM model or tokenizer not found. Will use RDKit descriptors only.")

    def extract_combined_features(self, smiles: Union[str, List[str]]) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Extract combined features: LSTM embeddings + RDKit descriptors

        Args:
            smiles: SMILES string or list

        Returns:
            Tuple of (combined_features, valid_indices)
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = smiles

        # Extract LSTM embeddings
        if self.embedding_extractor is not None:
            lstm_embeddings, lstm_valid_indices = self.embedding_extractor.extract_embeddings(smiles_list)
        else:
            lstm_embeddings = None
            lstm_valid_indices = list(range(len(smiles_list)))

        # Extract RDKit descriptors
        rdkit_descriptors = []
        rdkit_valid_indices = []

        if self.descriptor_calculator is not None:
            for idx, smi in enumerate(smiles_list):
                desc = self.descriptor_calculator.calculate_descriptors(smi)
                if desc is not None:
                    rdkit_descriptors.append(desc)
                    rdkit_valid_indices.append(idx)

        # Find common valid indices
        if lstm_embeddings is not None:
            valid_indices = list(set(lstm_valid_indices) & set(rdkit_valid_indices))
        else:
            valid_indices = rdkit_valid_indices

        if len(valid_indices) == 0:
            return None, []

        # Combine features
        combined_features = []

        for idx in valid_indices:
            features = []

            # Add LSTM embedding
            if lstm_embeddings is not None:
                lstm_idx = lstm_valid_indices.index(idx)
                features.append(lstm_embeddings[lstm_idx])

            # Add RDKit descriptors
            rdkit_idx = rdkit_valid_indices.index(idx)
            features.append(rdkit_descriptors[rdkit_idx])

            # Concatenate
            combined_features.append(np.concatenate(features))

        return np.array(combined_features), valid_indices

    def train_model(self,
                   smiles_list: List[str],
                   targets: np.ndarray,
                   model_name: str,
                   model_type: str = 'classifier',
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train ADMET model with transfer learning

        Args:
            smiles_list: List of SMILES strings
            targets: Target values
            model_name: Name for the model (e.g., 'toxicity_tl')
            model_type: 'classifier' or 'regressor'
            test_size: Test set size
            random_state: Random seed

        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()} MODEL WITH TRANSFER LEARNING")
        print(f"{'='*80}")

        # Extract combined features
        print("\nExtracting combined features (LSTM embeddings + RDKit descriptors)...")
        X, valid_indices = self.extract_combined_features(smiles_list)

        if X is None or len(valid_indices) == 0:
            print("[ERROR] No valid features extracted")
            return None

        # Filter targets to match valid indices
        y = targets[valid_indices]

        print(f"Feature shape: {X.shape}")
        print(f"  - Combined features: {X.shape[1]} dimensions")
        if self.embedding_extractor:
            print(f"    * LSTM embeddings: {self.embedding_extractor.embedding_dim} dims")
            print(f"    * RDKit descriptors: {X.shape[1] - self.embedding_extractor.embedding_dim} dims")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print(f"\nTraining Random Forest {model_type.capitalize()}...")

        if model_type == 'classifier':
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
        else:  # regressor
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = model.predict(X_test_scaled)

        results = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1]
        }

        if model_type == 'classifier':
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            results.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            })
        else:
            results.update({
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            })

        # Save model
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.save_model(model_name, model, scaler)

        # Print results
        print(f"\n{'-'*80}")
        print(f"{model_name.upper()} RESULTS (TRANSFER LEARNING)")
        print(f"{'-'*80}")
        if model_type == 'classifier':
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1-Score:  {results['f1']:.4f}")
            print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        else:
            print(f"R² Score:  {results['r2']:.4f}")
            print(f"RMSE:      {results['rmse']:.4f}")
            print(f"MAE:       {results['mae']:.4f}")
        print(f"\nTrain samples: {results['n_train']}")
        print(f"Test samples:  {results['n_test']}")
        print(f"Total features: {results['n_features']}")

        self.results[model_name] = results
        return results

    def save_model(self, model_name: str, model, scaler):
        """Save trained model and scaler"""
        model_path = self.models_dir / f"{model_name}_model.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"[OK] Model saved to {model_path}")

    def load_model(self, model_name: str) -> bool:
        """Load trained model and scaler"""
        model_path = self.models_dir / f"{model_name}_model.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            print(f"[WARNING] Model files not found for {model_name}")
            return False

        try:
            self.models[model_name] = joblib.load(model_path)
            self.scalers[model_name] = joblib.load(scaler_path)
            print(f"[OK] Loaded {model_name} model from {model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            return False

    def predict(self, smiles: Union[str, List[str]], model_name: str) -> Dict:
        """
        Predict ADMET properties using transfer learning model

        Args:
            smiles: SMILES string or list
            model_name: Name of the model to use

        Returns:
            Dictionary with predictions
        """
        if model_name not in self.models:
            print(f"[ERROR] Model {model_name} not loaded")
            return None

        # Extract features
        X, valid_indices = self.extract_combined_features(smiles)

        if X is None or len(valid_indices) == 0:
            return {'error': 'Failed to extract features'}

        # Scale features
        X_scaled = self.scalers[model_name].transform(X)

        # Predict
        model = self.models[model_name]
        predictions = model.predict(X_scaled)

        # Get probabilities for classifiers
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
        else:
            probabilities = None

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'valid_indices': valid_indices
        }


# Convenience function
def create_transfer_learning_model(**kwargs):
    """Create and return an ADMET Transfer Learning model"""
    return ADMETTransferLearningModel(**kwargs)
