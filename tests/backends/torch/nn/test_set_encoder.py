import json
import os
import subprocess
import sys

import pytest
import torch

from sc_flow.core.nn import BaseModule
from sc_flow.flow._pooling import BasePooling, PoolingSpec
from sc_flow.flow._set_encoder import SetEncoder
from sc_flow.flow._vf import MLPVelocity

INPUT_LAYERS = {
    "drug": {"input_dim": 4, "output_dim": 6},
    "dose": {"input_dim": 2, "output_dim": 3},
}
MEAN_POOLING = PoolingSpec(type="sc_flow.mean", version=1, config={})


def token_pooling(**changes) -> PoolingSpec:
    config = {
        "num_heads": 2,
        "qkv_dim": 8,
        "dropout": 0.0,
        "num_layers": 1,
        "transformer_block": False,
        "layer_norm": False,
        "ff_dim": None,
        **changes,
    }
    return PoolingSpec(type="sc_flow.attention_token", version=1, config=config)


@pytest.mark.parametrize("pooling_type", ["sc_flow.mean", "sc_flow.sum"])
def test_set_encoder_builtin_pooling(pooling_type: str) -> None:
    encoder = SetEncoder(
        input_layers=INPUT_LAYERS,
        output_dim=7,
        pooling=PoolingSpec(type=pooling_type, version=1, config={}),
        pooling_proj_dim=5,
    )
    condition = {
        "drug": torch.randn(4, 3, 4),
        "dose": torch.randn(4, 2, 2),
    }

    mean, logvar = encoder(condition)

    assert mean.shape == (4, 7)
    assert logvar is None
    assert encoder.decoder_input_dim == 5
    assert set(encoder._condition_encoder["proj_layers"]) == set(INPUT_LAYERS)


def test_set_encoder_mixed_pooled_and_bypassed_covariates() -> None:
    encoder = SetEncoder(
        input_layers=INPUT_LAYERS,
        output_dim=7,
        pooling=MEAN_POOLING,
        pooling_proj_dim=5,
        covariates_not_pooled=["dose"],
    )

    mean, _ = encoder({"drug": torch.randn(4, 3, 4), "dose": torch.randn(4, 2)})

    assert mean.shape == (4, 7)
    assert encoder.decoder_input_dim == 8
    assert set(encoder._condition_encoder["proj_layers"]) == {"drug"}


def test_attention_seed_controls_decoder_input_dim() -> None:
    encoder = SetEncoder(
        input_layers={"drug": INPUT_LAYERS["drug"]},
        output_dim=7,
        pooling=PoolingSpec(
            type="sc_flow.attention_seed",
            version=1,
            config={
                "num_heads": 4,
                "v_dim": 12,
                "seed_dim": 9,
                "dropout": 0.0,
                "transformer_block": False,
                "layer_norm": False,
                "ff_dim": None,
            },
        ),
        pooling_proj_dim=5,
    )

    mean, _ = encoder({"drug": torch.randn(4, 3, 4)})

    assert mean.shape == (4, 7)
    assert encoder.pooling_output_dim == 12
    assert encoder.decoder_input_dim == 12


def test_set_encoder_combines_per_realm_masks() -> None:
    encoder = SetEncoder(
        input_layers=INPUT_LAYERS,
        output_dim=7,
        pooling=token_pooling(),
        pooling_proj_dim=5,
    ).eval()
    condition = {
        "drug": torch.randn(2, 3, 4),
        "dose": torch.randn(2, 2, 2),
    }
    masks = {
        "drug": torch.tensor([[True, True, False], [True, False, False]]),
        "dose": torch.tensor([[True, False], [True, True]]),
    }

    expected, _ = encoder(condition, condition_mask=masks)
    condition = {realm: values.masked_fill(~masks[realm].unsqueeze(-1), 100_000) for realm, values in condition.items()}
    actual, _ = encoder(condition, condition_mask=masks)

    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match=r"must contain every pooled covariate.*missing=\['dose'\]"):
        encoder(condition, condition_mask={"drug": masks["drug"]})


class MaxPooling(BasePooling):
    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._valid_mask(x, mask)
        if mask is None:
            return x.amax(dim=-2)
        return x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=-2)


def test_custom_pooling_runs_but_disables_portable_export(tmp_path) -> None:
    encoder = SetEncoder(
        input_layers={"drug": INPUT_LAYERS["drug"]},
        output_dim=7,
        pooling=MaxPooling(input_dim=5),
        pooling_proj_dim=5,
    )

    mean, _ = encoder({"drug": torch.randn(2, 3, 4)})

    assert mean.shape == (2, 7)
    assert encoder.pooling_spec is None
    assert encoder._runtime_module_constructor_args == {"pooling": "MaxPooling"}
    assert not any(key.startswith("_pooling.") for key in encoder.state_dict())
    with pytest.raises(ValueError, match=r"pooling=MaxPooling.*cannot be serialized"):
        encoder.save_pretrained(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_builtin_pooling_config_and_weights_round_trip(tmp_path) -> None:
    spec = token_pooling()
    encoder = SetEncoder(
        input_layers={"drug": INPUT_LAYERS["drug"]},
        output_dim=7,
        pooling=spec,
        pooling_proj_dim=5,
    ).eval()
    condition = {"drug": torch.randn(2, 3, 4)}
    expected, _ = encoder(condition)

    encoder.save_pretrained(tmp_path)
    restored = SetEncoder.from_pretrained(tmp_path).eval()
    actual, _ = restored(condition)

    torch.testing.assert_close(actual, expected)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["pooling"] == spec


def test_builtin_pooling_round_trip_in_fresh_process(tmp_path) -> None:
    spec = token_pooling()
    encoder = SetEncoder(
        input_layers={"drug": INPUT_LAYERS["drug"]},
        output_dim=7,
        pooling=spec,
        pooling_proj_dim=5,
    ).eval()
    condition = {"drug": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10}
    expected, _ = encoder(condition)
    encoder.save_pretrained(tmp_path)

    script = """
import json
import sys
import torch
from sc_flow.flow._set_encoder import SetEncoder

encoder = SetEncoder.from_pretrained(sys.argv[1]).eval()
condition = {"drug": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10}
output, _ = encoder(condition)
print("SC_FLOW_RESULT=" + json.dumps(output.tolist()))
"""
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = "/tmp/sc-flow-tools-matplotlib"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result_line = next(line for line in result.stdout.splitlines() if line.startswith("SC_FLOW_RESULT="))
    actual = torch.tensor(json.loads(result_line.removeprefix("SC_FLOW_RESULT=")))

    torch.testing.assert_close(actual, expected)


def test_conditional_velocity_export_waits_for_enclosing_component_spec(tmp_path) -> None:
    encoder = SetEncoder(
        input_layers={"drug": INPUT_LAYERS["drug"]},
        output_dim=7,
        pooling=MEAN_POOLING,
    )
    vf = MLPVelocity(state_dim=4, condition_encoder=encoder)

    assert vf._runtime_module_constructor_args == {"condition_encoder": "SetEncoder"}
    with pytest.raises(ValueError, match=r"condition_encoder=SetEncoder.*cannot be serialized"):
        vf.save_pretrained(tmp_path)


def test_runtime_module_guard_finds_modules_nested_in_constructor_containers(tmp_path) -> None:
    class ContainerModule(BaseModule):
        def __init__(self, components: dict[str, list[torch.nn.Module]]) -> None:
            super().__init__()
            self.components = torch.nn.ModuleList(components["items"])

        def _make_modules(self) -> torch.nn.Module:
            return self.components

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for component in self.components:
                x = component(x)
            return x

    module = ContainerModule({"items": [torch.nn.Identity()]})

    assert module._runtime_module_constructor_args == {"components['items'][0]": "Identity"}
    with pytest.raises(ValueError, match=r"components\['items'\]\[0\]=Identity.*cannot be serialized"):
        module.save_pretrained(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_set_encoder_rejects_missing_and_extra_covariates() -> None:
    encoder = SetEncoder(input_layers=INPUT_LAYERS, output_dim=7, pooling=MEAN_POOLING)

    with pytest.raises(ValueError, match="missing from the query"):
        encoder({"drug": torch.randn(2, 3, 4)})
    with pytest.raises(ValueError, match="dont appear in the reference"):
        encoder(
            {
                "drug": torch.randn(2, 3, 4),
                "dose": torch.randn(2, 2, 2),
                "unknown": torch.randn(2, 1, 1),
            }
        )


def test_set_encoder_rejects_empty_condition() -> None:
    encoder = SetEncoder(input_layers={"drug": INPUT_LAYERS["drug"]}, output_dim=7, pooling=MEAN_POOLING)
    with pytest.raises(ValueError, match="No condition covariate found"):
        encoder({})
