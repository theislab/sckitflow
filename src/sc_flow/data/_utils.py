import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_label_encoder(data: np.ndarray) -> LabelEncoder:
    """Fits a label encoder on the provided data."""
    if data.ndim == 2:
        if data.shape[1] != 1:
            msg = (
                'When using "label"  as target representation in `target_covariates`,'
                f"the last dimension should be singleton, but found data of shape {data.shape}."
            )
            raise ValueError(msg)
        data = data.reshape(-1)

    # we require only one dimension when using label encoding
    if data.ndim != 1:
        msg = (
            'When using "label" as target representation in `target_covariates`,'
            f"the data should have only one dimension, but found data of shape {data.shape}."
        )
        raise ValueError(msg)
    return LabelEncoder().fit(data)


def get_one_hot_encoder(data: np.ndarray) -> OneHotEncoder:
    """Fits a one hot encoder on the provided data."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.ndim != 2:
        msg = (
            'When using "one-hot" as target representation in `target_covariates`,'
            f"you need to pass a 2-dimensional array, found {data.ndim=}."
        )
        raise ValueError(msg)
    return OneHotEncoder().fit(data)
