from contextlib import suppress

import datamol as dm
import safe as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def smiles_to_safe(smiles: str, ignore_stereo: bool = True) -> str:
    return sf.SAFEConverter(ignore_stereo=ignore_stereo).encoder(smiles, allow_empty=True)


def safe_to_smiles(safe_str: str, fix: bool = True) -> str:
    if fix:
        safe_str = ".".join(
            [frag for frag in safe_str.split(".") if sf.decode(frag, ignore_errors=True) is not None]
        )
    return sf.decode(safe_str, canonical=True, ignore_errors=True)


def filter_by_substructure(sequences, substruct):
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(Chem.MolFromSmarts(substruct), Chem.MolFromSmiles("*"))
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


def mix_sequences(prefix_sequences, suffix_sequences, prefix, suffix, num_samples=1):
    mol_linker_slicer = sf.utils.MolSlicer(require_ring_system=False)

    prefix_linkers = []
    suffix_linkers = []
    prefix_query = dm.from_smarts(prefix)
    suffix_query = dm.from_smarts(suffix)

    for x in prefix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, prefix_query)
            prefix_linkers.append(out[1])

    for x in suffix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, suffix_query)
            suffix_linkers.append(out[1])

    linked = []
    linkers = [x for x in (prefix_linkers + suffix_linkers) if x is not None]
    for n_linked, linker in enumerate(linkers):
        linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
        if n_linked > num_samples:
            break
        linked = [x for x in linked if x]
    return linked[:num_samples]
