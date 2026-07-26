import json
from dataclasses import dataclass

import pytest

from sc_flow._registry import Component, parse, to_spec
from sc_flow.concept import ContrastiveObjective, ContrastiveObjectiveConfig, GeneEncoderConfig, ObjectiveConfig


def test_gene_encoder_config_roundtrips_through_json():
    cfg = GeneEncoderConfig(n_tokens=50, dim_model=16, n_layers=1, n_heads=2, activation="gelu")
    spec = to_spec(cfg)
    assert spec["type"] == "sc_flow.gene_encoder" and spec["version"] == 1
    assert parse(json.loads(json.dumps(spec))) == cfg  # dispatched by the registry


def test_objective_family_scoped_entry_builds_runtime():
    cfg = ContrastiveObjectiveConfig(logit_scale_init=2.0)
    back = ObjectiveConfig.from_spec(cfg.to_spec())  # typed, family-enforced classmethod
    assert isinstance(back, ContrastiveObjectiveConfig)
    assert isinstance(back.build(), ContrastiveObjective)


def test_version_mismatch_rejected():
    spec = GeneEncoderConfig(n_tokens=10).to_spec()
    spec["version"] = 99
    with pytest.raises(ValueError, match="version"):
        parse(spec)


def test_unknown_field_rejected_loudly():
    spec = GeneEncoderConfig(n_tokens=10).to_spec()
    spec["config"]["dim_modell"] = 8  # typo
    with pytest.raises(ValueError, match="dim_modell"):
        parse(spec)


def test_family_mis_wire_rejected():
    enc_spec = GeneEncoderConfig(n_tokens=10).to_spec()
    with pytest.raises(ValueError, match="not a ObjectiveConfig"):
        ObjectiveConfig.from_spec(enc_spec)  # a gene_encoder is not an ObjectiveConfig


def test_duplicate_type_id_raises():
    with pytest.raises(ValueError, match="already registered"):

        @dataclass
        class _Dup(Component, type_id="sc_flow.gene_encoder", version=1):
            def build(self, ctx=None):
                return None
