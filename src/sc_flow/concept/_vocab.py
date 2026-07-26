"""Gene vocabulary + Ensembl alignment for the contrastive cell encoder (:mod:`sc_flow.concept`).

The contrastive encoder consumes *tokens*, not raw expression: every gene a cell expresses is looked
up as an integer token and the transformer sees those tokens (their order encodes expression rank — see
:mod:`sc_flow.concept._tokenize`). A :class:`GeneVocab` is the fixed, ordered gene panel for one species
that *defines* that token space.

Reserved tokens come first so ``padding_idx=0`` composes with :class:`torch.nn.Embedding`: ``PAD=0``,
``CLS=1``; real genes are numbered from :data:`NUM_SPECIAL` in vocabulary order. (scConcept uses the
opposite ``cls=0, pad=1`` layout; we never load its weights, so we take the padding-friendly convention
and record the difference here.)

Alignment maps an :class:`~anndata.AnnData`'s ``var`` (Ensembl gene ids) onto these tokens, dropping
genes absent from the vocabulary — the mechanism that lets a heterogeneous CellxGene corpus (many
panels, different gene sets per dataset) speak a single token space.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["GeneVocab", "PAD_TOKEN", "CLS_TOKEN", "NUM_SPECIAL"]

PAD_TOKEN = 0
CLS_TOKEN = 1
NUM_SPECIAL = 2  # tokens [0, NUM_SPECIAL) are reserved; real genes are numbered from NUM_SPECIAL


def _normalize_ensembl(gene_id: str) -> str:
    """Uppercase and strip a trailing Ensembl version suffix (``ENSG…​.15`` -> ``ENSG…``)."""
    g = gene_id.strip().upper()
    dot = g.find(".")
    return g[:dot] if dot != -1 else g


class GeneVocab:
    """A fixed, ordered Ensembl gene panel for one species mapped to stable integer tokens.

    Parameters
    ----------
    gene_ids
        Ensembl gene ids in the order that defines their token numbering. Duplicates (after
        normalization) are dropped, keeping the first occurrence.
    species
        Species tag (e.g. ``"hsapiens"``) — carried for multi-species routing, not used for lookup.
    strip_version
        Strip trailing Ensembl version suffixes (``.N``) on both the vocabulary and query ids, so
        ``ENSG…​.15`` matches ``ENSG…``. On by default (CellxGene ids often carry versions).
    """

    def __init__(self, gene_ids: Iterable[str], *, species: str = "hsapiens", strip_version: bool = True) -> None:
        self.species = species
        self._strip = strip_version
        # Dedupe preserving first-seen order — the order *is* the token numbering, so it must be stable.
        seen: dict[str, int] = {}
        for g in gene_ids:
            n = self._norm(g)
            if n not in seen:
                seen[n] = len(seen)
        self._gene_ids: tuple[str, ...] = tuple(seen)
        self._id_to_token: dict[str, int] = {g: i + NUM_SPECIAL for i, g in enumerate(self._gene_ids)}

    def _norm(self, gene_id: str) -> str:
        return _normalize_ensembl(gene_id) if self._strip else gene_id.strip().upper()

    @property
    def gene_ids(self) -> tuple[str, ...]:
        """The normalized Ensembl ids in token order (index ``i`` -> token ``i + NUM_SPECIAL``)."""
        return self._gene_ids

    @property
    def n_genes(self) -> int:
        return len(self._gene_ids)

    @property
    def n_tokens(self) -> int:
        """Size of the embedding table: reserved specials + genes."""
        return NUM_SPECIAL + len(self._gene_ids)

    def token_of(self, gene_id: str) -> int:
        """Token id for a single Ensembl id, or ``-1`` if the gene is not in the vocabulary."""
        return self._id_to_token.get(self._norm(gene_id), -1)

    def align(self, var_ids: Sequence[str]) -> np.ndarray:
        """Map an AnnData ``var`` (Ensembl ids, in column order) to tokens; ``-1`` where unmapped.

        Returns an ``int64`` array of length ``len(var_ids)``: entry ``j`` is the token for column ``j``
        of the count matrix, or ``-1`` if that gene is absent from the vocabulary (the caller drops it).
        Computed once per data source and reused for every cell.
        """
        get = self._id_to_token.get
        norm = self._norm
        out = np.full(len(var_ids), -1, dtype=np.int64)
        for j, g in enumerate(var_ids):
            out[j] = get(norm(g), -1)
        return out

    @classmethod
    def from_csv(
        cls, path: str, *, gene_id_column: str = "gene_id", species: str = "hsapiens", strip_version: bool = True
    ) -> GeneVocab:
        """Load a vocabulary from a CSV whose ``gene_id_column`` holds Ensembl ids (row order = token order)."""
        import csv

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        return cls((r[gene_id_column] for r in rows), species=species, strip_version=strip_version)

    def __len__(self) -> int:
        return self.n_tokens

    def __repr__(self) -> str:
        return f"GeneVocab(species={self.species!r}, n_genes={self.n_genes}, n_tokens={self.n_tokens})"
