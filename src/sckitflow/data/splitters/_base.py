from __future__ import annotations

import pandas as pd
from anndata import AnnData

from sckitflow.data._utils import with_derived_obs

__all__ = ["Splitter"]


class Splitter:
    """Base class for splitters that annotate observations with a split label.

    A splitter is deliberately decoupled from :class:`~sckitflow.data.DataManager`: its only
    job is to write a categorical ``split`` column into ``adata.obs``. The data manager then
    consumes that column via ``split_by=<column>`` and builds one data loader per split value,
    without knowing *how* the split was produced. This keeps splitting policy (which cells go
    where) separate from data configuration (how cells are read and batched).

    Subclasses override :meth:`assign` to return a per-observation label series; :meth:`split`
    writes it into ``adata.obs``.
    """

    def __init__(self, *, split_key: str = "split") -> None:
        """Initializes the splitter.

        :param split_key: Name of the ``adata.obs`` column the split label is written to. This
            is the value later passed to :class:`~sckitflow.data.DataManager` as ``split_by``.
            Defaults to ``"split"``.
        :type split_key: class: `str`
        """
        self._split_key = split_key

    @property
    def split_key(self) -> str:
        """The ``adata.obs`` column the split label is written to."""
        return self._split_key

    def assign(self, adata: AnnData) -> pd.Series:
        """Computes a per-observation split label. Must be overridden by subclasses.

        :param adata: The annotated data object to split.
        :type adata: class: `AnnData`

        :returns: A series of split labels aligned to ``adata.obs_names``.
        :rtype: class: `pandas.Series`
        """
        raise NotImplementedError

    def split(self, adata: AnnData, *, copy: bool = False) -> AnnData:
        """Writes the split label into ``adata.obs[self.split_key]``.

        Refuses to overwrite an existing ``split_key`` column: a previous split is someone's decision about
        which cells are held out, and silently replacing it makes every downstream loader disagree with the
        run that produced it. Drop the column (or pass a different ``split_key``) to re-split deliberately.

        :param adata: The annotated data object to annotate.
        :type adata: class: `AnnData`

        :param copy: If ``True``, write to a *shallow* copy and return that, leaving the input's ``.obs``
            untouched -- ``X`` / ``obsm`` / ``uns`` stay shared, so no cell data is copied. Defaults to
            ``False`` (annotate in place and return the same object).
        :type copy: class: `bool`

        :returns: The annotated annotated data object (a shallow copy when ``copy=True``).
        :rtype: class: `AnnData`
        """
        if self._split_key in adata.obs.columns:
            raise ValueError(
                f"adata.obs already has a {self._split_key!r} column; refusing to overwrite an existing split. "
                f"Drop it (`del adata.obs[{self._split_key!r}]`) to re-split, or construct this splitter with "
                "another `split_key`."
            )
        labels = pd.Categorical(self.assign(adata).reindex(adata.obs_names).to_numpy())
        if copy:
            return with_derived_obs(adata, **{self._split_key: labels})
        adata.obs[self._split_key] = labels
        return adata

    def __call__(self, adata: AnnData, *, copy: bool = False) -> AnnData:
        """Alias for :meth:`split`."""
        return self.split(adata, copy=copy)
