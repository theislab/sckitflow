import pytest
import torch

from sckitflow._types import LayersDict, NestedLayersDict
from sckitflow.core.nn._set_encoder import SetEncoder

# dimensions
batch_size = 32
n_combs = 3
output_dim = 16
condition0_input_dim = 4
condition0_output_dim = 2
condition1_input_dim = 8
condition1_output_dim = 4
pooling_proj_dim = 5

# dictionaries
input_layers_single_condition = {
    "condition0": {
        "input_dim": condition0_input_dim,
        "output_dim": condition0_output_dim,
    },
}
input_layers_double_condition = {
    "condition0": {
        "input_dim": condition0_input_dim,
        "output_dim": condition0_output_dim,
    },
    "condition1": {
        "input_dim": condition1_input_dim,
        "output_dim": condition1_output_dim,
    },
}


class TestSetEncoder:
    @pytest.mark.parametrize(
        "input_layers, covariates_not_pooled, expected_decoder_input_dim",
        [
            (input_layers_single_condition, [], pooling_proj_dim),  # one pooled covariate
            (input_layers_single_condition, ["condition0"], condition0_output_dim),  # one not pooled
            (
                input_layers_double_condition,
                [],
                pooling_proj_dim,
            ),  # two pooled -> concat along set dim, projected to pooling_proj_dim
            (
                input_layers_double_condition,
                ["condition0"],
                pooling_proj_dim + condition0_output_dim,
            ),  # condition0 not pooled, condition1 pooled
            (
                input_layers_double_condition,
                ["condition0", "condition1"],
                condition0_output_dim + condition1_output_dim,
            ),  # both not pooled
        ],
    )
    @pytest.mark.parametrize("pooling_mode", ["mean", "sum"])
    @pytest.mark.parametrize("output_layers_kwargs", [{}])
    def test_set_encoder_forward_and_properties(
        self,
        input_layers: NestedLayersDict,
        covariates_not_pooled: list[str],
        expected_decoder_input_dim: int,
        pooling_mode: str,
        output_layers_kwargs: LayersDict,
    ) -> None:
        """Test forward pass shape, decoder_input_dim property, and internal concatenation logic."""
        encoder = SetEncoder(
            input_layers=input_layers,
            output_dim=output_dim,
            pooling_mode=pooling_mode,
            pooling_kwargs=None,
            pooling_proj_dim=pooling_proj_dim,
            pooling_proj_bias=True,
            covariates_not_pooled=covariates_not_pooled,
            output_layers_kwargs=output_layers_kwargs,
        )

        # Check decoder_input_dim property
        assert encoder.decoder_input_dim == expected_decoder_input_dim

        # Build condition dictionary with all covariates present
        condition_dict = {}
        for cov_id, cfg in input_layers.items():
            n_entries = 1 if cov_id in covariates_not_pooled else n_combs
            condition_dict[cov_id] = torch.randn(batch_size, n_entries, cfg["input_dim"])

        # Forward pass
        encoded = encoder(condition_dict)
        assert encoded.shape == (batch_size, output_dim)

        # Also check that the internal modules exist as expected
        pooled_covs = encoder.covariates_pooled
        assert set(pooled_covs) == set(input_layers.keys()) - set(covariates_not_pooled)

        # projection layers exist for pooled covariates only
        proj_layers = encoder._condition_encoder["proj_layers"]
        assert set(proj_layers.keys()) == set(pooled_covs)
        for cov in pooled_covs:
            assert isinstance(proj_layers[cov], torch.nn.Linear)
            assert proj_layers[cov].in_features == input_layers[cov]["output_dim"]
            assert proj_layers[cov].out_features == pooling_proj_dim

    def test_not_pooled_continuous_covariate(self) -> None:
        """A continuous covariate streams as `[B, D]`, without a set axis, and is used as is."""
        encoder = SetEncoder(
            input_layers=input_layers_double_condition, output_dim=output_dim, covariates_not_pooled=["condition0"]
        )
        condition_dict = {
            "condition0": torch.randn(batch_size, condition0_input_dim),
            "condition1": torch.randn(batch_size, n_combs, condition1_input_dim),
        }
        assert encoder(condition_dict).shape == (batch_size, output_dim)

    def test_not_pooled_needs_one_entry(self) -> None:
        """A not-pooled covariate with several entries would have to drop or reorder them, so it is refused."""
        encoder = SetEncoder(
            input_layers=input_layers_double_condition, output_dim=output_dim, covariates_not_pooled=["condition0"]
        )
        condition_dict = {
            "condition0": torch.randn(batch_size, n_combs, condition0_input_dim),
            "condition1": torch.randn(batch_size, n_combs, condition1_input_dim),
        }
        with pytest.raises(ValueError, match="'condition0' is not pooled.*found 3"):
            encoder(condition_dict)

    @pytest.mark.xfail(
        reason="https://github.com/theislab/sckitflow/issues/144 - validation raises ValueError, not KeyError",
        strict=True,
    )
    def test_set_encoder_no_covariates_error(self) -> None:
        """Test that providing an empty condition dict raises ValueError."""
        encoder = SetEncoder(
            input_layers=input_layers_single_condition,
            output_dim=output_dim,
            pooling_mode="mean",
        )
        with pytest.raises(ValueError, match="No condition covariate found"):
            encoder({})

    @pytest.mark.xfail(
        reason="https://github.com/theislab/sckitflow/issues/144 - validation raises ValueError, not KeyError",
        strict=True,
    )
    def test_set_encoder_missing_covariate_error(self) -> None:
        """Test that missing a required covariate raises KeyError."""
        encoder = SetEncoder(
            input_layers=input_layers_double_condition,
            output_dim=output_dim,
            pooling_mode="mean",
        )
        condition_dict = {"condition0": torch.randn(batch_size, n_combs, condition0_input_dim)}
        with pytest.raises(KeyError, match="Input encoder not found for covariate condition1"):
            encoder(condition_dict)

    @pytest.mark.xfail(
        reason="https://github.com/theislab/sckitflow/issues/144 - validation raises ValueError, not KeyError",
        strict=True,
    )
    def test_set_encoder_extra_covariate_error(self) -> None:
        """Test that providing an extra covariate not in input_layers raises KeyError."""
        encoder = SetEncoder(
            input_layers=input_layers_single_condition,
            output_dim=output_dim,
            pooling_mode="mean",
        )
        condition_dict = {
            "condition0": torch.randn(batch_size, n_combs, condition0_input_dim),
            "condition1": torch.randn(batch_size, n_combs, condition1_input_dim),
        }
        with pytest.raises(KeyError, match="Input encoder not found for covariate condition1"):
            encoder(condition_dict)


@pytest.mark.parametrize(
    ("pooling_mode", "pooling_kwargs", "pooled_dim"),
    [
        ("attention-token", {"num_heads": 2, "qkv_dim": 8}, pooling_proj_dim),
        ("attention-seed", {"num_heads": 2, "v_dim": 8, "seed_dim": 4}, 8),
    ],
)
def test_attention_pooling(pooling_mode: str, pooling_kwargs: dict, pooled_dim: int) -> None:
    """Attention pooling sizes the decoder, encodes the set order-free, and trains its query."""
    encoder = SetEncoder(
        input_layers=input_layers_double_condition,
        output_dim=output_dim,
        pooling_mode=pooling_mode,
        pooling_kwargs=pooling_kwargs,
        pooling_proj_dim=pooling_proj_dim,
        covariates_not_pooled=["condition1"],
    )
    assert encoder.decoder_input_dim == pooled_dim + condition1_output_dim

    x = torch.randn(batch_size, n_combs, condition0_input_dim)
    other = torch.randn(batch_size, 1, condition1_input_dim)
    encoded = encoder({"condition0": x, "condition1": other})
    assert encoded.shape == (batch_size, output_dim)
    torch.testing.assert_close(encoder({"condition0": x.flip(-2), "condition1": other}), encoded)

    encoded.sum().backward()
    assert encoder._condition_encoder["pooling_layer"].query.grad is not None


def test_attention_pooling_rejects_uneven_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SetEncoder(
            input_layers=input_layers_single_condition,
            output_dim=output_dim,
            pooling_mode="attention-token",
            pooling_kwargs={"num_heads": 3, "qkv_dim": 8},
        )
