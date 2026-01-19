"""
Molecular metrics for evaluating generated molecules.
Includes validity, uniqueness, diversity, novelty, and chemical properties.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski, AllChem, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from collections import Counter


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit molecule object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def is_valid(smiles: str) -> bool:
    """Check if SMILES string represents a valid molecule."""
    mol = smiles_to_mol(smiles)
    return mol is not None


def validity_rate(smiles_list: List[str]) -> float:
    """Calculate the fraction of valid molecules."""
    if not smiles_list:
        return 0.0
    valid_count = sum(1 for s in smiles_list if is_valid(s))
    return valid_count / len(smiles_list)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def uniqueness_rate(smiles_list: List[str]) -> Tuple[float, List[str]]:
    """
    Calculate the fraction of unique molecules among valid ones.
    Returns (uniqueness_rate, unique_smiles_list)
    """
    # Canonicalize all valid SMILES
    canonical = []
    for s in smiles_list:
        c = canonicalize_smiles(s)
        if c is not None:
            canonical.append(c)

    if not canonical:
        return 0.0, []

    unique = list(set(canonical))
    return len(unique) / len(canonical), unique


def diversity_rate(smiles_list: List[str], n_samples: int = 1000) -> float:
    """
    Calculate diversity as average pairwise Tanimoto distance.
    Uses Morgan fingerprints. Samples pairs if list is large.
    """
    # Get valid molecules
    mols = [smiles_to_mol(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return 0.0

    # Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    # Sample pairs if too many molecules
    if len(fps) > 100:
        np.random.seed(42)
        indices = np.random.choice(len(fps), size=min(n_samples, len(fps)), replace=False)
        sampled_fps = [fps[i] for i in indices]
    else:
        sampled_fps = fps

    # Calculate pairwise Tanimoto distances
    distances = []
    for i in range(len(sampled_fps)):
        for j in range(i + 1, len(sampled_fps)):
            similarity = DataStructs.TanimotoSimilarity(sampled_fps[i], sampled_fps[j])
            distances.append(1 - similarity)  # Distance = 1 - similarity

    if not distances:
        return 0.0

    return float(np.mean(distances))


def novelty_rate(generated_smiles: List[str], train_smiles: List[str]) -> float:
    """
    Calculate the fraction of generated molecules not in training set.
    """
    # Canonicalize generated SMILES
    gen_canonical = set()
    for s in generated_smiles:
        c = canonicalize_smiles(s)
        if c is not None:
            gen_canonical.add(c)

    if not gen_canonical:
        return 0.0

    # Canonicalize training SMILES
    train_canonical = set()
    for s in train_smiles:
        c = canonicalize_smiles(s)
        if c is not None:
            train_canonical.add(c)

    # Count novel molecules
    novel_count = sum(1 for s in gen_canonical if s not in train_canonical)
    return novel_count / len(gen_canonical)


# Chemical property calculations

def calculate_qed(smiles: str) -> Optional[float]:
    """Calculate Quantitative Estimate of Drug-likeness."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return QED.qed(mol)


def calculate_sa(smiles: str) -> Optional[float]:
    """
    Calculate Synthetic Accessibility score.
    Note: Requires rdkit.Chem.SA_Score module (not always available).
    Returns None if not available.
    """
    try:
        from rdkit.Chem import SA_Score
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
        return SA_Score.sascorer.calculateScore(mol)
    except ImportError:
        # Fallback: estimate based on complexity
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
        # Simple heuristic: (num_rings + num_heteroatoms) / num_heavy_atoms
        num_rings = Descriptors.RingCount(mol)
        num_heteroatoms = Lipinski.NumHeteroatoms(mol)
        num_heavy = Lipinski.HeavyAtomCount(mol)
        if num_heavy == 0:
            return None
        return min(10.0, (num_rings + num_heteroatoms) / num_heavy * 10)


def calculate_logp(smiles: str) -> Optional[float]:
    """Calculate LogP (partition coefficient)."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)


def calculate_mw(smiles: str) -> Optional[float]:
    """Calculate molecular weight."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def calculate_tpsa(smiles: str) -> Optional[float]:
    """Calculate Topological Polar Surface Area."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Descriptors.TPSA(mol)


def calculate_num_rotatable_bonds(smiles: str) -> Optional[int]:
    """Calculate number of rotatable bonds."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Lipinski.NumRotatableBonds(mol)


def calculate_num_h_donors(smiles: str) -> Optional[int]:
    """Calculate number of hydrogen bond donors."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Lipinski.NumHDonors(mol)


def calculate_num_h_acceptors(smiles: str) -> Optional[int]:
    """Calculate number of hydrogen bond acceptors."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Lipinski.NumHAcceptors(mol)


# Similarity metrics

def tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Calculate Tanimoto similarity between two molecules using Morgan fingerprints."""
    mol1 = smiles_to_mol(smiles1)
    mol2 = smiles_to_mol(smiles2)

    if mol1 is None or mol2 is None:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def tanimoto_distance(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Calculate Tanimoto distance (1 - similarity) between two molecules."""
    sim = tanimoto_similarity(smiles1, smiles2, radius, n_bits)
    if sim is None:
        return None
    return 1.0 - sim


# Molecular filters

def lipinski_filter(smiles: str) -> bool:
    """
    Check if molecule passes Lipinski's Rule of Five.
    MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


def pains_filter(smiles: str) -> bool:
    """
    Check if molecule passes PAINS filter (not a Pan-Assay Interference compound).
    Returns True if molecule is NOT a PAINS compound (i.e., passes filter).
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return False

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    return not catalog.HasMatch(mol)


def drug_like_filter(smiles: str) -> bool:
    """
    Check if molecule is drug-like (passes both Lipinski and PAINS filters).
    """
    return lipinski_filter(smiles) and pains_filter(smiles)


# Statistical functions

def compute_property_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics for a list of property values.
    Returns mean, std, median, min, max, and percentiles.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def compute_all_properties(smiles_list: List[str]) -> Dict[str, List[float]]:
    """
    Compute all chemical properties for a list of SMILES.
    Returns dict with property names as keys and lists of values.
    """
    properties = {
        "qed": [],
        "sa": [],
        "logp": [],
        "mw": [],
        "tpsa": [],
        "num_rotatable_bonds": [],
        "num_h_donors": [],
        "num_h_acceptors": [],
    }

    for smiles in smiles_list:
        if not is_valid(smiles):
            continue

        qed = calculate_qed(smiles)
        sa = calculate_sa(smiles)
        logp = calculate_logp(smiles)
        mw = calculate_mw(smiles)
        tpsa = calculate_tpsa(smiles)
        rot_bonds = calculate_num_rotatable_bonds(smiles)
        h_donors = calculate_num_h_donors(smiles)
        h_acceptors = calculate_num_h_acceptors(smiles)

        if qed is not None:
            properties["qed"].append(qed)
        if sa is not None:
            properties["sa"].append(sa)
        if logp is not None:
            properties["logp"].append(logp)
        if mw is not None:
            properties["mw"].append(mw)
        if tpsa is not None:
            properties["tpsa"].append(tpsa)
        if rot_bonds is not None:
            properties["num_rotatable_bonds"].append(float(rot_bonds))
        if h_donors is not None:
            properties["num_h_donors"].append(float(h_donors))
        if h_acceptors is not None:
            properties["num_h_acceptors"].append(float(h_acceptors))

    return properties
