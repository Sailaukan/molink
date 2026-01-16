import logging
import warnings
from typing import Iterable, List, Optional, Sequence

import datamol as dm
import safe as sf
import torch
from rdkit import RDLogger

from molink.model import MolinkRWKV
from molink.utils.bracket_safe import BracketSAFEConverter, bracketsafe2safe
from molink.utils.safe_utils import filter_by_substructure, safe_to_smiles

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


class MolinkSampler:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            self.model = MolinkRWKV.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            raise

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Using device: {self.device}")

        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        self.use_bracket_safe = self.model.config.training.get("use_bracket_safe", False)
        self.safe_encoder = (
            BracketSAFEConverter(ignore_stereo=True)
            if self.use_bracket_safe
            else sf.SAFEConverter(ignore_stereo=True)
        )
        logger.info(f"Sampler initialized (bracket_safe: {self.use_bracket_safe})")

    def _encode_prefix(self, text: str, is_safe: bool = False) -> str:
        """Encode text to SAFE format."""
        if is_safe:
            return text
        try:
            return self.safe_encoder.encoder(text, allow_empty=True)
        except Exception as e:
            logger.warning(f"Failed to encode prefix '{text}': {e}")
            return ""

    def _decode_safe(self, safe_str: str) -> Optional[str]:
        """Decode SAFE string to SMILES."""
        if not safe_str:
            return None
        try:
            if self.use_bracket_safe:
                safe_str = bracketsafe2safe(safe_str)
            return safe_to_smiles(safe_str, fix=True)
        except Exception as e:
            logger.warning(f"Failed to decode SAFE string: {e}")
            return None

    def _generate_safe(
        self,
        prefix_safe: str,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[str]:
        """Generate SAFE strings from prefix."""
        try:
            batch = self.tokenizer(
                [prefix_safe] * num_samples,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.model.max_seq_len,
            )
            input_ids = batch["input_ids"].to(self.device)

            # Remove trailing EOS token if present and sequence is not empty
            if input_ids.shape[1] > 1 and input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                input_ids = input_ids[:, :-1]

            # Generate new tokens
            with torch.no_grad():
                output_ids = self.model.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []

    def de_novo_generation(
        self,
        num_samples: int = 8,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> List[str]:
        """Generate molecules de novo (from scratch)."""
        logger.info(f"De novo generation: {num_samples} samples, max_tokens={max_new_tokens}, "
                   f"temp={temperature}, top_p={top_p}, top_k={top_k}")
        try:
            bos = torch.full((num_samples, 1), self.tokenizer.bos_token_id, device=self.device)
            with torch.no_grad():
                output_ids = self.model.model.generate(
                    bos,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            safe_strings = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            smiles = [self._decode_safe(s) for s in safe_strings if s]
            smiles = [s for s in smiles if s is not None]
            logger.info(f"Generated {len(smiles)}/{num_samples} valid molecules")
            return smiles
        except Exception as e:
            logger.error(f"De novo generation failed: {e}")
            return []

    def scaffold_decoration(
        self,
        scaffold: str,
        num_samples: int = 8,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
    ) -> List[str]:
        """Decorate a scaffold with additional chemical groups."""
        logger.info(f"Scaffold decoration: scaffold='{scaffold}', {num_samples} samples")

        prefix_safe = self._encode_prefix(scaffold)
        if not prefix_safe:
            logger.error("Failed to encode scaffold")
            return []

        if not prefix_safe.endswith("."):
            prefix_safe = prefix_safe + "."

        safe_strings = self._generate_safe(
            prefix_safe,
            num_samples,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )

        smiles = [self._decode_safe(s) for s in safe_strings if s]
        smiles = [s for s in smiles if s is not None]

        if apply_filter and smiles:
            try:
                smiles = filter_by_substructure(smiles, scaffold)
                logger.info(f"After filtering: {len(smiles)} molecules contain scaffold")
            except Exception as e:
                logger.warning(f"Substructure filtering failed: {e}")

        logger.info(f"Generated {len(smiles)}/{num_samples} valid molecules")
        return smiles

    def motif_extension(self, motif: str, **kwargs) -> List[str]:
        return self.scaffold_decoration(motif, **kwargs)

    def linker_design(
        self,
        fragment_a: str,
        fragment_b: str,
        num_samples: int = 8,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
    ) -> List[str]:
        """Design linkers between two fragments."""
        logger.info(f"Linker design: fragment_a='{fragment_a}', fragment_b='{fragment_b}', {num_samples} samples")

        safe_a = self._encode_prefix(fragment_a)
        safe_b = self._encode_prefix(fragment_b)

        if not safe_a or not safe_b:
            logger.error("Failed to encode one or both fragments")
            return []

        prefix_safe = f"{safe_a}.{safe_b}."
        safe_strings = self._generate_safe(
            prefix_safe,
            num_samples,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )

        smiles = [self._decode_safe(s) for s in safe_strings if s]
        smiles = [s for s in smiles if s is not None]

        if apply_filter and smiles:
            try:
                initial_count = len(smiles)
                smiles = filter_by_substructure(smiles, fragment_a)
                smiles = filter_by_substructure(smiles, fragment_b)
                logger.info(f"After filtering: {len(smiles)}/{initial_count} molecules contain both fragments")
            except Exception as e:
                logger.warning(f"Substructure filtering failed: {e}")

        logger.info(f"Generated {len(smiles)}/{num_samples} valid molecules")
        return smiles

    def scaffold_morphing(
        self,
        side_chains: Optional[Sequence[str]] = None,
        mol: Optional[str] = None,
        core: Optional[str] = None,
        num_samples: int = 8,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> List[str]:
        """Morph scaffold while keeping side chains (scaffold hopping)."""
        logger.info(f"Scaffold morphing: {num_samples} samples")

        if side_chains is None:
            if mol is None or core is None:
                raise ValueError("Provide side_chains or (mol and core)")
            try:
                side_chains = sf.utils.compute_side_chains(dm.to_mol(mol), dm.to_mol(core))
            except Exception as e:
                logger.error(f"Failed to compute side chains: {e}")
                return []

        if isinstance(side_chains, str):
            side_chains = [side_chains]

        try:
            side_chains = [dm.to_smiles(dm.to_mol(sc)) for sc in side_chains]
        except Exception as e:
            logger.error(f"Failed to process side chains: {e}")
            return []

        side_chains_str = ".".join(side_chains)
        if "*" not in side_chains_str:
            warnings.warn("Side chains do not contain dummy atoms; morphing may fail.")
            logger.warning("Side chains do not contain dummy atoms")

        prefix_safe = self._encode_prefix(side_chains_str)
        if not prefix_safe:
            logger.error("Failed to encode side chains")
            return []

        if not prefix_safe.endswith("."):
            prefix_safe = prefix_safe + "."

        safe_strings = self._generate_safe(
            prefix_safe,
            num_samples,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )

        smiles = [self._decode_safe(s) for s in safe_strings if s]
        smiles = [s for s in smiles if s is not None]

        if smiles:
            try:
                smiles = filter_by_substructure(smiles, side_chains_str)
                logger.info(f"After filtering: {len(smiles)} molecules contain side chains")
            except Exception as e:
                logger.warning(f"Substructure filtering failed: {e}")

        logger.info(f"Generated {len(smiles)}/{num_samples} valid molecules")
        return smiles
