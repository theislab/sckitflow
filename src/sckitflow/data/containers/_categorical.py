from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from sckitflow._types import TargetCovariatesEncoderCls
from sckitflow.data._mixins import BatchMixin, MappedArray
from sckitflow.data._utils import convert_to_categorical_in_place, get_one_hot_encoder
from sckitflow.data.containers._base import BaseData

__all__ = ["CategoricalData"]


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """Container class for categorical data.

    Any categorical data is defined over a set of column, stored in a :class: `pandas.DataFrame`.
    There are two possible ways to represent categorical variables using this container.
    The first one is to pass pre-computed representations, with the :param repr_dict: argument.
    Otherwise it is possible to specify some pre-defined encoders to transform the categorical
    values stored in the data frame into suitable representations.

    :param ann_df: The data frame storing the original values.
    :type ann_df: class: `pandas.DataFrame`

    :param repr_dict: Dictionary storing the pre-computed representations.
    :type repr_dict: class: `dict[str, MappedArray]`

    :param categorical_encoders: Mapping storing the covariate encoders class.
    :type categorical_encoders: Mapping[str, TargetCovariatesEncoderCls]
    """

    ann_df: pd.DataFrame
    repr_dict: Mapping[str, MappedArray] = dc_field(default_factory=dict)
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=dict)
    categorical_reps_map: Mapping[str, str] = dc_field(default_factory=dict)

    def __post_init__(self):
        # check that the representations are provided
        for col in self.ann_df.columns:
            # each column needs to appear in the categorical reps map
            if col not in self.categorical_reps_map:
                raise KeyError(f"Column {col} is not mapped to any realm.")
            col_realm = self.categorical_reps_map[col]

            # we need an associated representation for the corresponding realms
            if col_realm not in self.repr_dict and col_realm not in self.categorical_encoders:
                raise KeyError(f"No representation found for column {col} associated to realm {col_realm}.")

    def __repr__(self) -> str:
        n_obs, n_vars = self.ann_df.shape
        cols = list(self.ann_df.columns)

        repr_keys = list(self.repr_dict.keys())
        encoder_keys = list(self.categorical_encoders.keys())

        return (
            f"{self.__class__.__name__}("
            f"n_obs={n_obs}, "
            f"n_vars={n_vars}, "
            f"columns={cols}, "
            f"repr_dict_keys={repr_keys}, "
            f"categorical_encoders_keys={encoder_keys})"
        )

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "CategoricalData":
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(
            ann_df,
            repr_dict=self.repr_dict,
            categorical_encoders=self.categorical_encoders,
            categorical_reps_map=self.categorical_reps_map,
        )

    def __len__(self) -> int:
        return self.ann_df.shape[0]

    def extract_reps(self) -> BatchMixin:
        """Extracts the representations for the underlying data, one array per realm.

        Each column resolves to a ``(n_obs, d)`` block -- looked up per value for a realm with stored
        representations, or computed by that realm's encoder -- and the columns sharing a realm are
        stacked into ``(n_obs, n_cols, d)``. That trailing pair is the set axis the conditioning
        encoder pools over, so the ``n_cols`` axis is kept even for a single-column realm.

        Only categorical covariates come through here: continuous ones are per-obs ``obsm`` arrays
        that the loaders stream straight into the batch, already ``(n_obs, d)``.
        """
        data_dict = defaultdict(list)

        for col in self.ann_df.columns:
            realm = self.categorical_reps_map[col]

            if realm in self.repr_dict:
                # ---- Stored representations: one lookup per value ----
                realm_repr_dict = self.repr_dict[realm]
                col_values = self.ann_df[col].drop_duplicates().values
                if col_values.shape[0] != 1:
                    raise ValueError(
                        "The node contains more than one unique value for the "
                        "categorical covariates. Ensure you extract the representation "
                        "from a leaf after having properly built the data tree."
                    )
                reprs = []
                for val in col_values:
                    rep = np.asarray(realm_repr_dict[val])
                    # A stored representation is one vector per value -- a drug embedding is `d`
                    # dimensional. `(1, d)` is accepted as the equivalent row form, but nothing is
                    # reshaped into place: flattening a `(2, 3)` embedding to `(1, 6)` would disagree
                    # with `DimsRegistry`, which reads the dimension as `shape[-1]` (3), and the model
                    # would build a layer of 3 and be fed 6.
                    if rep.ndim == 2 and rep.shape[0] == 1:
                        rep = rep[0]
                    elif rep.ndim != 1:
                        raise ValueError(
                            f"The representation for value {val!r} of column {col!r} has shape "
                            f"{rep.shape}; a categorical representation must be one vector per value, "
                            "so `(d,)` or `(1, d)`."
                        )
                    reprs.append(rep)
                col_repr = np.vstack(reprs)
                # Every value of a realm must encode to the same width. Nothing upstream checks this:
                # `__post_init__` only checks that a realm *has* representations, and
                # `DataDimensionalitiesRegistry` reads the realm's dimension off whichever value comes
                # first (`next(iter(...))`). A leaf holds one value, so a disagreement would never reach
                # `np.stack` below -- one leaf would emit `(1, 1, 5)` and the next `(1, 1, 3)` for the
                # same realm, and the model would be built for one and fed the other.
                widths = {int(np.asarray(r).shape[-1]) for r in realm_repr_dict.values()}
                if len(widths) > 1:
                    raise ValueError(
                        f"The representations for realm {realm!r} have differing widths {sorted(widths)}; "
                        "every value of a realm must map to a vector of the same length."
                    )

            elif realm in self.categorical_encoders:
                # ---- Encoder: sklearn wants `(n_samples, n_features)`, and this is one column ----
                # `[:, None]` rather than `reshape(-1, 1)`: it adds the feature axis to a known-1-D
                # column, so the result is `(n_obs, 1)` by construction rather than by inference.
                col_values = self.ann_df[col].values
                if col_values.ndim != 1:
                    raise ValueError(
                        f"Column {col!r} of the annotation frame is {col_values.ndim}-dimensional; a "
                        "categorical column must be a single 1-dimensional series."
                    )
                # `LabelEncoder` is the one sklearn transformer that takes the raw 1-D column; the
                # rest want `(n_samples, n_features)`. Passing a column-vector to it "works" but
                # raises `DataConversionWarning`, so branch on what the encoder actually asks for.
                encoder = self.categorical_encoders[realm]
                col_repr = encoder.transform(col_values if isinstance(encoder, LabelEncoder) else col_values[:, None])
                if isinstance(col_repr, csr_matrix):
                    col_repr = col_repr.toarray()
                col_repr = np.asarray(col_repr)
                if col_repr.ndim == 1:
                    # `LabelEncoder` emits one integer code per row, `(n_obs,)`, where every other
                    # encoder emits `(n_obs, d)`. Restore the width axis so the realm stacks to
                    # `(n_obs, n_cols, d)` like the rest -- a `d` of 1 is still a width.
                    col_repr = col_repr[:, None]
                elif col_repr.ndim != 2:
                    raise ValueError(
                        f"The encoder for realm {realm!r} returned a rank-{col_repr.ndim} array "
                        f"(shape {col_repr.shape}) for column {col!r}; a categorical encoder must emit "
                        "one row per observation, so `(n_obs,)` or `(n_obs, d)`."
                    )

            else:  # unreachable: `__post_init__` requires every column's realm to have one or the other
                raise KeyError(f"No representation found for column {col} associated to realm {realm}.")

            data_dict[realm].append(col_repr)

        # Stack each realm's columns onto the set axis -> (n_obs, n_cols, d). `np.stack` requires the
        # blocks to agree exactly, so a realm whose columns disagree on width fails here; check first to
        # name the realm and the shapes, which numpy's own "all input arrays must have the same shape"
        # does not.
        for realm, blocks in data_dict.items():
            shapes = {np.asarray(b).shape for b in blocks}
            if len(shapes) > 1:
                raise ValueError(
                    f"The columns mapped to realm {realm!r} produced representations of differing shapes "
                    f"{sorted(shapes)}; columns sharing a realm must encode to the same width."
                )
        return BatchMixin(mapping={k: np.stack(v, axis=-2) for k, v in data_dict.items()})

    @classmethod
    def from_pandas(
        cls,
        ann_df: pd.DataFrame,
        repr_dict: Mapping[str, MappedArray] | None = None,
        categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] | None = None,
        inplace: bool = False,
        categorical_reps_map: Mapping[str, str] | None = None,
    ) -> "CategoricalData":
        """Create a CategoricalData object from a pandas DataFrame.

        TODO: document properly but most importantly note that for better performance it is recommended to pass the data in place.
        """
        if not inplace:
            ann_df = ann_df.copy()
        convert_to_categorical_in_place(ann_df, ann_df.columns)

        # prepare defaults containers
        categorical_reps_map = (
            {col: col for col in ann_df.columns} if categorical_reps_map is None else categorical_reps_map
        )
        repr_dict = {} if repr_dict is None else repr_dict
        categorical_encoders = {} if categorical_encoders is None else categorical_encoders

        # set default encoders
        for cov_realm in set(categorical_reps_map.values()):
            if cov_realm not in categorical_encoders and cov_realm not in repr_dict:
                cov_data = ann_df.loc[:, cov_realm].values
                ohe = get_one_hot_encoder(cov_data)
                categorical_encoders[cov_realm] = ohe

        return cls(
            ann_df,
            repr_dict={} if repr_dict is None else repr_dict,
            categorical_encoders={} if categorical_encoders is None else categorical_encoders,
            categorical_reps_map=categorical_reps_map,
        )

    @property
    def category_realms(self) -> list[str]:
        """Returns the category realms associated to the object."""
        return list(set(self.categorical_reps_map.values()))
