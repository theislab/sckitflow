"""Group encoders are portable components, with string ids accepted only at the interfaces."""

import pickle

import cloudpickle
import numpy as np
import pytest

from sckitflow.data._group_encoders import (
    Affine,
    GroupEncoder,
    GroupEncoderContext,
    Identity,
    Label,
    Log1p,
    OneHot,
    as_group_encoder,
)
from sckitflow.data._manager import DataManager

ALL_ENCODERS = [Identity(), Label(), OneHot(), Log1p(), Affine(scale=2.0, shift=1.0)]


class TestStringIds:
    """Strings are coerced to components at the interface; only parameter-free ids exist."""

    @pytest.mark.parametrize(("encoder_id", "expected"), [("label", Label()), ("one-hot", OneHot())])
    def test_coerces_string_id(self, encoder_id: str, expected: GroupEncoder) -> None:
        assert as_group_encoder(encoder_id) == expected

    def test_passes_instances_through(self) -> None:
        encoder = Affine(scale=2.0)
        assert as_group_encoder(encoder) is encoder

    @pytest.mark.parametrize("bad", ["functional", "identity", "log1p", "affine", "nope", None, 1])
    def test_rejects_unknown_id(self, bad: object) -> None:
        # "functional" is gone: its transform lived in the removed groups_encoding_transform_fn callables.
        # The parameterized encoders have no string form either -- pass the instance instead.
        with pytest.raises(ValueError, match="not available"):
            as_group_encoder(bad)

    def test_datamanager_accepts_strings_and_stores_components(self) -> None:
        dm = DataManager(groups=("drug", "ko"), groups_encoding={"drug": "one-hot", "ko": Label()})
        assert dm.groups_data_schema.groups_encoders == {"drug": OneHot(), "ko": Label()}

    def test_datamanager_rejects_unknown_id(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            DataManager(groups=("drug",), groups_encoding={"drug": "functional"})


class TestPortableSpec:
    """Every encoder round-trips through the scfit component registry."""

    @pytest.mark.parametrize("encoder", ALL_ENCODERS)
    def test_spec_round_trip(self, encoder: GroupEncoder) -> None:
        restored = GroupEncoder.from_spec(encoder.to_spec())
        assert restored == encoder
        assert type(restored) is type(encoder)

    def test_spec_is_json_shaped(self) -> None:
        assert Affine(scale=2.0).to_spec() == {
            "type": "group_encoder.affine",
            "version": 1,
            "config": {"scale": 2.0, "shift": 0.0},
        }

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "group_encoder.affine", "version": 1, "config": {"scal": 2.0}},  # typo'd field
            {"type": "group_encoder.nope", "version": 1, "config": {}},  # unknown type
            {"type": "group_encoder.affine", "version": 99, "config": {}},  # unsupported version
        ],
    )
    def test_rejects_bad_spec(self, spec: dict) -> None:
        with pytest.raises((ValueError, TypeError)):
            GroupEncoder.from_spec(spec)


class TestPinnedVocabulary:
    """A pinned vocabulary makes the mapping deterministic, and unknown categories raise."""

    def test_one_hot_pins_column_order(self) -> None:
        encoder = OneHot(categories=("c", "a", "b"))
        # order is preserved as given (not sorted) and independent of the fit data
        for data in (np.array(["a", "b"]), np.array(["b", "a", "a"])):
            built = encoder.build(GroupEncoderContext(data))
            assert list(built.categories_[0]) == ["c", "a", "b"]

    def test_one_hot_unpinned_fits_from_data(self) -> None:
        built = OneHot().build(GroupEncoderContext(np.array(["b", "a"])))
        assert list(built.categories_[0]) == ["a", "b"]

    def test_one_hot_raises_on_unknown_category(self) -> None:
        built = OneHot(categories=("a", "b")).build(GroupEncoderContext(np.array(["a"])))
        with pytest.raises(ValueError, match="unknown categories"):
            built.transform(np.array([["zzz"]]))

    def test_one_hot_raises_when_fit_data_exceeds_vocabulary(self) -> None:
        with pytest.raises(ValueError, match="unknown categories"):
            OneHot(categories=("a", "b")).build(GroupEncoderContext(np.array(["a", "QQ"])))

    def test_label_pins_mapping(self) -> None:
        built = Label(classes=("x", "y", "z")).build(GroupEncoderContext(np.array(["x"])))
        assert list(built.classes_) == ["x", "y", "z"]

    def test_label_raises_on_unseen_label(self) -> None:
        built = Label(classes=("x", "y")).build(GroupEncoderContext(np.array(["x"])))
        with pytest.raises(ValueError, match="unseen labels"):
            built.transform(np.array(["nope"]))


class TestSerialization:
    """Configs carry no callables, so a DataManager holding one survives plain pickle."""

    @pytest.mark.parametrize("encoder", ALL_ENCODERS)
    def test_encoder_pickles(self, encoder: GroupEncoder) -> None:
        assert pickle.loads(pickle.dumps(encoder)) == encoder

    def test_datamanager_round_trips(self) -> None:
        dm = DataManager(groups=("cell_line",), groups_encoding={"cell_line": Affine(scale=2.0, shift=1.0)})

        for restored in (pickle.loads(pickle.dumps(dm)), cloudpickle.loads(cloudpickle.dumps(dm))):
            encoder = restored.groups_data_schema.groups_encoders["cell_line"]
            assert encoder == Affine(scale=2.0, shift=1.0)
            # the fitted transformer is rebuilt on demand
            built = encoder.build(GroupEncoderContext(np.arange(3.0)))
            np.testing.assert_allclose(built.transform(np.arange(3.0)), np.arange(3.0) * 2.0 + 1.0)
            np.testing.assert_allclose(built.inverse_transform(built.transform(np.arange(3.0))), np.arange(3.0))
