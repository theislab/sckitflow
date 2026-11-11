"""Group encoders are portable components that list their categories explicitly."""

import pickle

import cloudpickle
import numpy as np
import pytest

from sckitflow.data._group_encoders import (
    AffineTransformerConfig,
    GroupEncoderConfig,
    GroupEncoderContext,
    IdentityTransformerConfig,
    LabelEncoderConfig,
    Log1pTransformerConfig,
    OneHotEncoderConfig,
)
from sckitflow.data._manager import DataManagerConfig

ALL_ENCODERS = [
    IdentityTransformerConfig(),
    LabelEncoderConfig(classes=("a", "b")),
    OneHotEncoderConfig(categories=("a", "b")),
    Log1pTransformerConfig(),
    AffineTransformerConfig(scale=2.0, shift=1.0),
]


class TestVocabularyIsExplicit:
    """The categorical encoders take their categories from the config, never from the data."""

    @pytest.mark.parametrize("encoder", [LabelEncoderConfig, OneHotEncoderConfig])
    def test_categories_are_required(self, encoder: type[GroupEncoderConfig]) -> None:
        with pytest.raises(ValueError, match="Field required"):
            encoder()

    def test_datamanager_rejects_string_ids(self) -> None:
        with pytest.raises(ValueError, match="groups_encoding"):
            DataManagerConfig(target_sample_rep="X", groups=("drug",), groups_encoding={"drug": "one-hot"})


class TestPortableSpec:
    """Every encoder round-trips through the scfit component registry."""

    @pytest.mark.parametrize("encoder", ALL_ENCODERS)
    def test_spec_round_trip(self, encoder: GroupEncoderConfig) -> None:
        restored = GroupEncoderConfig.from_spec(encoder.to_spec())
        assert restored == encoder
        assert type(restored) is type(encoder)

    def test_spec_is_json_shaped(self) -> None:
        assert AffineTransformerConfig(scale=2.0).to_spec() == {
            "type": "group_encoder.affine",
            "scale": 2.0,
            "shift": 0.0,
        }

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "group_encoder.affine", "scal": 2.0},  # typo'd field
            {"type": "group_encoder.nope"},  # unknown type
        ],
    )
    def test_rejects_bad_spec(self, spec: dict) -> None:
        with pytest.raises((ValueError, TypeError)):
            GroupEncoderConfig.from_spec(spec)


class TestVocabularyOrder:
    """The vocabulary fixes the mapping in the order given, and unknown categories raise."""

    def test_one_hot_pins_column_order(self) -> None:
        encoder = OneHotEncoderConfig(categories=("c", "a", "b"))
        # order is preserved as given (not sorted) and independent of the fit data
        for data in (np.array(["a", "b"]), np.array(["b", "a", "a"])):
            built = encoder.build(GroupEncoderContext(data))
            assert list(built.categories_[0]) == ["c", "a", "b"]

    def test_one_hot_raises_on_unknown_category(self) -> None:
        built = OneHotEncoderConfig(categories=("a", "b")).build(GroupEncoderContext(np.array(["a"])))
        with pytest.raises(ValueError, match="unknown categories"):
            built.transform(np.array([["zzz"]]))

    def test_one_hot_raises_when_fit_data_exceeds_vocabulary(self) -> None:
        with pytest.raises(ValueError, match="unknown categories"):
            OneHotEncoderConfig(categories=("a", "b")).build(GroupEncoderContext(np.array(["a", "QQ"])))

    def test_label_codes_follow_the_given_order(self) -> None:
        built = LabelEncoderConfig(classes=("z", "x", "y")).build(GroupEncoderContext(np.array(["x"])))
        assert built.transform(np.array(["x", "y", "z"])).tolist() == [1, 2, 0]
        assert built.inverse_transform([0, 1, 2]).tolist() == ["z", "x", "y"]

    def test_label_raises_on_unseen_label(self) -> None:
        built = LabelEncoderConfig(classes=("x", "y")).build(GroupEncoderContext(np.array(["x"])))
        with pytest.raises(ValueError, match="unseen labels"):
            built.transform(np.array(["nope"]))


class TestSerialization:
    """Configs carry no callables, so a DataManager holding one survives plain pickle."""

    @pytest.mark.parametrize("encoder", ALL_ENCODERS)
    def test_encoder_pickles(self, encoder: GroupEncoderConfig) -> None:
        assert pickle.loads(pickle.dumps(encoder)) == encoder

    def test_datamanager_round_trips(self) -> None:
        dm = DataManagerConfig(
            target_sample_rep="X",
            groups=("cell_line",),
            groups_encoding={"cell_line": AffineTransformerConfig(scale=2.0, shift=1.0)},
        ).build()

        for restored in (pickle.loads(pickle.dumps(dm)), cloudpickle.loads(cloudpickle.dumps(dm))):
            encoder = restored.groups_data_schema.groups_encoders["cell_line"]
            assert encoder == AffineTransformerConfig(scale=2.0, shift=1.0)
            # the fitted transformer is rebuilt on demand
            built = encoder.build(GroupEncoderContext(np.arange(3.0)))
            np.testing.assert_allclose(built.transform(np.arange(3.0)), np.arange(3.0) * 2.0 + 1.0)
            np.testing.assert_allclose(built.inverse_transform(built.transform(np.arange(3.0))), np.arange(3.0))


@pytest.mark.parametrize("make", [lambda v: LabelEncoderConfig(classes=v), lambda v: OneHotEncoderConfig(categories=v)])
def test_a_vocabulary_lists_each_category_once(make) -> None:
    with pytest.raises(ValueError, match=r"lists \['a'\] more than once"):
        make(("a", "b", "a"))


@pytest.mark.parametrize("make", [lambda v: LabelEncoderConfig(classes=v), lambda v: OneHotEncoderConfig(categories=v)])
def test_a_vocabulary_is_not_empty(make) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make(())


def test_integer_categories_encode_integer_columns() -> None:
    data = np.array([3, 1, 2])
    label = LabelEncoderConfig(classes=(3, 1, 2)).build(GroupEncoderContext(data))
    assert label.transform(data).tolist() == [0, 1, 2]
    one_hot = OneHotEncoderConfig(categories=(1, 2, 3)).build(GroupEncoderContext(data))
    assert one_hot.transform(data.reshape(-1, 1)).toarray().argmax(axis=1).tolist() == [2, 0, 1]
    assert LabelEncoderConfig.from_spec(LabelEncoderConfig(classes=(3, 1)).to_spec()).classes == (3, 1)
