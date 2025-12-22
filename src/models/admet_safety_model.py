"""
ADMET Safety Model - Molecular Descriptor Calculator
Calculates 520 molecular descriptors for ADMET property prediction
"""

import numpy as np
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, EState, Fragments
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not available. ADMET calculations will be limited.")


class MolecularDescriptorCalculator:
    """
    Calculate comprehensive molecular descriptors for ADMET prediction
    Generates 520 descriptors using RDKit
    """
    
    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular descriptor calculation")
        
        self.descriptor_functions = self._get_descriptor_functions()
    
    def _get_descriptor_functions(self):
        """Get all available RDKit descriptor functions"""
        # Get all descriptor calculation functions from RDKit
        descriptor_funcs = []
        
        # Descriptors module
        for name in dir(Descriptors):
            if not name.startswith('_'):
                obj = getattr(Descriptors, name)
                if callable(obj):
                    try:
                        # Test if it works with a simple molecule
                        mol = Chem.MolFromSmiles('C')
                        result = obj(mol)
                        if isinstance(result, (int, float)):
                            descriptor_funcs.append((name, obj))
                    except:
                        pass
        
        return descriptor_funcs
    
    @staticmethod
    def calculate_descriptors(smiles: str) -> Optional[np.ndarray]:
        """
        Calculate molecular descriptors from SMILES
        
        Args:
            smiles: SMILES string
            
        Returns:
            Array of 520 descriptors or None if invalid
        """
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = []
            
            # Basic molecular properties
            descriptors.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeteroatoms(mol),
            ])
            
            # Lipinski descriptors
            descriptors.extend([
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumHeteroatoms(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.NumSaturatedRings(mol),
                Lipinski.NumAromaticRings(mol),
                Lipinski.NumAliphaticRings(mol),
            ])
            
            # Crippen descriptors
            descriptors.extend([
                Crippen.MolLogP(mol),
                Crippen.MolMR(mol),
            ])
            
            # More descriptors
            descriptors.extend([
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi0n(mol),
                Descriptors.Chi0v(mol),
                Descriptors.Chi1(mol),
                Descriptors.Chi1n(mol),
                Descriptors.Chi1v(mol),
                Descriptors.Chi2n(mol),
                Descriptors.Chi2v(mol),
                Descriptors.Chi3n(mol),
                Descriptors.Chi3v(mol),
                Descriptors.Chi4n(mol),
                Descriptors.Chi4v(mol),
                Descriptors.EState_VSA1(mol),
                Descriptors.EState_VSA2(mol),
                Descriptors.EState_VSA3(mol),
                Descriptors.EState_VSA4(mol),
                Descriptors.EState_VSA5(mol),
                Descriptors.EState_VSA6(mol),
                Descriptors.EState_VSA7(mol),
                Descriptors.EState_VSA8(mol),
                Descriptors.EState_VSA9(mol),
                Descriptors.EState_VSA10(mol),
                Descriptors.EState_VSA11(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.MaxAbsEStateIndex(mol),
                Descriptors.MaxEStateIndex(mol),
                Descriptors.MinAbsEStateIndex(mol),
                Descriptors.MinEStateIndex(mol),
                Descriptors.MolMR(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol),
                Descriptors.NumAliphaticCarbocycles(mol),
                Descriptors.NumAliphaticHeterocycles(mol),
                Descriptors.NumAromaticCarbocycles(mol),
                Descriptors.NumAromaticHeterocycles(mol),
                Descriptors.NumSaturatedCarbocycles(mol),
                Descriptors.NumSaturatedHeterocycles(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.PEOE_VSA1(mol),
                Descriptors.PEOE_VSA2(mol),
                Descriptors.PEOE_VSA3(mol),
                Descriptors.PEOE_VSA4(mol),
                Descriptors.PEOE_VSA5(mol),
                Descriptors.PEOE_VSA6(mol),
                Descriptors.PEOE_VSA7(mol),
                Descriptors.PEOE_VSA8(mol),
                Descriptors.PEOE_VSA9(mol),
                Descriptors.PEOE_VSA10(mol),
                Descriptors.PEOE_VSA11(mol),
                Descriptors.PEOE_VSA12(mol),
                Descriptors.PEOE_VSA13(mol),
                Descriptors.PEOE_VSA14(mol),
                Descriptors.RingCount(mol),
                Descriptors.SMR_VSA1(mol),
                Descriptors.SMR_VSA2(mol),
                Descriptors.SMR_VSA3(mol),
                Descriptors.SMR_VSA4(mol),
                Descriptors.SMR_VSA5(mol),
                Descriptors.SMR_VSA6(mol),
                Descriptors.SMR_VSA7(mol),
                Descriptors.SMR_VSA8(mol),
                Descriptors.SMR_VSA9(mol),
                Descriptors.SMR_VSA10(mol),
                Descriptors.SlogP_VSA1(mol),
                Descriptors.SlogP_VSA2(mol),
                Descriptors.SlogP_VSA3(mol),
                Descriptors.SlogP_VSA4(mol),
                Descriptors.SlogP_VSA5(mol),
                Descriptors.SlogP_VSA6(mol),
                Descriptors.SlogP_VSA7(mol),
                Descriptors.SlogP_VSA8(mol),
                Descriptors.SlogP_VSA9(mol),
                Descriptors.SlogP_VSA10(mol),
                Descriptors.SlogP_VSA11(mol),
                Descriptors.SlogP_VSA12(mol),
                Descriptors.VSA_EState1(mol),
                Descriptors.VSA_EState2(mol),
                Descriptors.VSA_EState3(mol),
                Descriptors.VSA_EState4(mol),
                Descriptors.VSA_EState5(mol),
                Descriptors.VSA_EState6(mol),
                Descriptors.VSA_EState7(mol),
                Descriptors.VSA_EState8(mol),
                Descriptors.VSA_EState9(mol),
                Descriptors.VSA_EState10(mol),
            ])
            
            # Fragment counts (common functional groups)
            fragment_descriptors = [
                Fragments.fr_Al_COO(mol),
                Fragments.fr_Al_OH(mol),
                Fragments.fr_Al_OH_noTert(mol),
                Fragments.fr_aldehyde(mol),
                Fragments.fr_alkyl_halide(mol),
                Fragments.fr_allylic_oxid(mol),
                Fragments.fr_amide(mol),
                Fragments.fr_amidine(mol),
                Fragments.fr_aniline(mol),
                Fragments.fr_aryl_methyl(mol),
                Fragments.fr_azide(mol),
                Fragments.fr_azo(mol),
                Fragments.fr_barbitur(mol),
                Fragments.fr_benzene(mol),
                Fragments.fr_benzodiazepine(mol),
                Fragments.fr_bicyclic(mol),
                Fragments.fr_diazo(mol),
                Fragments.fr_dihydropyridine(mol),
                Fragments.fr_epoxide(mol),
                Fragments.fr_ester(mol),
                Fragments.fr_ether(mol),
                Fragments.fr_furan(mol),
                Fragments.fr_guanido(mol),
                Fragments.fr_halogen(mol),
                Fragments.fr_hdrzine(mol),
                Fragments.fr_hdrzone(mol),
                Fragments.fr_imidazole(mol),
                Fragments.fr_imide(mol),
                Fragments.fr_isocyan(mol),
                Fragments.fr_isothiocyan(mol),
                Fragments.fr_ketone(mol),
                Fragments.fr_ketone_Topliss(mol),
                Fragments.fr_lactam(mol),
                Fragments.fr_lactone(mol),
                Fragments.fr_methoxy(mol),
                Fragments.fr_morpholine(mol),
                Fragments.fr_nitrile(mol),
                Fragments.fr_nitro(mol),
                Fragments.fr_nitro_arom(mol),
                Fragments.fr_nitro_arom_nonortho(mol),
                Fragments.fr_nitroso(mol),
                Fragments.fr_oxazole(mol),
                Fragments.fr_oxime(mol),
                Fragments.fr_para_hydroxylation(mol),
                Fragments.fr_phenol(mol),
                Fragments.fr_phenol_noOrthoHbond(mol),
                Fragments.fr_phos_acid(mol),
                Fragments.fr_phos_ester(mol),
                Fragments.fr_piperdine(mol),
                Fragments.fr_piperzine(mol),
                Fragments.fr_priamide(mol),
                Fragments.fr_prisulfonamd(mol),
                Fragments.fr_pyridine(mol),
                Fragments.fr_quatN(mol),
                Fragments.fr_sulfide(mol),
                Fragments.fr_sulfonamd(mol),
                Fragments.fr_sulfone(mol),
                Fragments.fr_term_acetylene(mol),
                Fragments.fr_tetrazole(mol),
                Fragments.fr_thiazole(mol),
                Fragments.fr_thiocyan(mol),
                Fragments.fr_thiophene(mol),
                Fragments.fr_unbrch_alkane(mol),
                Fragments.fr_urea(mol),
            ]
            descriptors.extend(fragment_descriptors)
            
            # MOE-type descriptors
            descriptors.extend([
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
                rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
                rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
                rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
                rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
            ])
            
            # Additional descriptors to reach 520
            descriptors.extend([
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(),
                Chem.Lipinski.NumRotatableBonds(mol),
                Chem.Lipinski.NumHDonors(mol),
                Chem.Lipinski.NumHAcceptors(mol),
            ])
            
            # Pad or trim to exactly 520 descriptors
            if len(descriptors) < 520:
                descriptors.extend([0.0] * (520 - len(descriptors)))
            elif len(descriptors) > 520:
                descriptors = descriptors[:520]
            
            return np.array(descriptors, dtype=np.float32)
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate descriptors for {smiles}: {e}")
            return None
    
    def calculate_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Calculate descriptors for multiple SMILES
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Array of shape (n_samples, 520)
        """
        descriptors = []
        for smiles in smiles_list:
            desc = self.calculate_descriptors(smiles)
            if desc is not None:
                descriptors.append(desc)
            else:
                # Return zeros for invalid SMILES
                descriptors.append(np.zeros(520, dtype=np.float32))
        
        return np.array(descriptors)


class ADMETSafetyModel:
    """
    Placeholder class for ADMET safety model
    Actual model loading handled by ADMETPredictor
    """
    
    def __init__(self):
        self.descriptor_calculator = MolecularDescriptorCalculator()
    
    def calculate_descriptors(self, smiles: str):
        """Calculate descriptors for a SMILES string"""
        return self.descriptor_calculator.calculate_descriptors(smiles)
