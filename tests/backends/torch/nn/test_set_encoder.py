import pytest
import torch

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow.flow._set_encoder import SetEncoder

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
            input_dim = cfg["input_dim"]
            condition_dict[cov_id] = torch.randn(batch_size, n_combs, input_dim)

        # Forward pass
        encoded = encoder(condition_dict)
        assert encoded.shape == (batch_size, output_dim)

        # Also check that the internal modules exist as expected
        pooled_covs = encoder.covariates_pooled
        not_pooled_covs = [c for c in input_layers.keys() if c not in covariates_not_pooled]
        assert set(pooled_covs) == set(input_layers.keys()) - set(covariates_not_pooled)

        # Verify projection layers exist for pooled covariates
        for cov in pooled_covs:
            assert f"{cov}_proj" in encoder._condition_encoder
            proj_layer = encoder._condition_encoder[f"{cov}_proj"]
            assert isinstance(proj_layer, torch.nn.Linear)
            assert proj_layer.in_features == input_layers[cov]["output_dim"]
            assert proj_layer.out_features == pooling_proj_dim

        # Verify no projection layers for non-pooled covariates
        for cov in not_pooled_covs:
            assert f"{cov}_proj" not in encoder._condition_encoder

    def test_set_encoder_no_covariates_error(self) -> None:
        """Test that providing an empty condition dict raises ValueError."""
        encoder = SetEncoder(
            input_layers=input_layers_single_condition,
            output_dim=output_dim,
            pooling_mode="mean",
        )
        with pytest.raises(ValueError, match="No condition covariate found"):
            encoder({})

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

    @pytest.mark.parametrize("pooling_mode", ["attention-token", "attention-seed"])
    def test_attention_pooling_not_implemented(self, pooling_mode: str) -> None:
        """Test that attention-based pooling modes raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SetEncoder(
                input_layers=input_layers_single_condition,
                output_dim=output_dim,
                pooling_mode=pooling_mode,
            )
