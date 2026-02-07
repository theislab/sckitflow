import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow.data._utils import (
    get_covariate_encoder,
    get_covariates_encoders_from_dict,
    get_label_encoder,
    get_one_hot_encoder,
)


def test_get_label_encoder_1d():
    data = np.array(["a", "b", "a"])
    encoder = get_label_encoder(data)
    assert isinstance(encoder, LabelEncoder)
    np.testing.assert_array_equal(encoder.transform(["a", "b"]), [0, 1])


def test_get_label_encoder_2d_singleton():
    data = np.array([["a"], ["b"], ["a"]])
    encoder = get_label_encoder(data)
    assert isinstance(encoder, LabelEncoder)


def test_get_label_encoder_2d_invalid_shape():
    data = np.array([["a", "b"], ["b", "c"]])
    with pytest.raises(ValueError, match="last dimension should be singleton"):
        get_label_encoder(data)


def test_get_label_encoder_nd_invalid():
    data = np.array([[[1], [2]], [[3], [4]]])
    with pytest.raises(ValueError, match="should have only one dimension"):
        get_label_encoder(data)


def test_get_one_hot_encoder_1d():
    data = np.array(["a", "b", "a"])
    encoder = get_one_hot_encoder(data)
    assert isinstance(encoder, OneHotEncoder)
    transformed = encoder.transform([["a"], ["b"]]).toarray()
    assert transformed.shape[1] == 2


def test_get_one_hot_encoder_2d():
    data = np.array([["a"], ["b"], ["a"]])
    encoder = get_one_hot_encoder(data)
    assert isinstance(encoder, OneHotEncoder)


def test_get_one_hot_encoder_invalid_nd():
    data = np.array([[[1], [2]], [[3], [4]]])
    with pytest.raises(ValueError, match="need to pass a 2-dimensional array"):
        get_one_hot_encoder(data)


@pytest.mark.parametrize(
    "encoder_id, expected_cls",
    [
        ("label", LabelEncoder),
        ("one-hot", OneHotEncoder),
        ("functional", FunctionTransformer),
    ],
)
def test_get_covariate_encoder_known_ids(encoder_id, expected_cls):
    data = np.array(["a", "b", "a"])
    encoder = get_covariate_encoder(encoder_id, data)
    assert isinstance(encoder, expected_cls)


def test_get_covariate_encoder_unknown_id():
    data = np.array(["a", "b"])
    with pytest.raises(ValueError, match="Covariate Encoder unknown"):
        get_covariate_encoder("unknown", data)


def test_get_covariates_encoders_from_dict(sample_df):
    cov_dict = {"cat1": "label", "cat2": "one-hot"}
    encoders = get_covariates_encoders_from_dict(cov_dict, sample_df)

    assert set(encoders.keys()) == {"cat1", "cat2"}
    assert isinstance(encoders["cat1"], LabelEncoder)
    assert isinstance(encoders["cat2"], OneHotEncoder)
