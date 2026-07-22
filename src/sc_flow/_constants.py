import numpy as np

# Architecture dimensions/counts carry NO module-level defaults on purpose: the low-level velocity field,
# embedders, time featurizer, and resnet require them explicitly (a `None`-to-magic-constant fallback would
# silently bake a hidden width into a saved model — the §14 footgun). Ergonomic defaults live on the
# ``FlowMatching`` facade, which passes concrete values down.

PI = np.pi


BASE_LEVEL_NAME = "base"
GROUP_LEVEL_NAME = "groups"
CONDITION_LEVEL_NAME = "conditions"

DEFAULT_BATCH_SIZE = 512
DEFAULT_N_GROUPS = 1
DEFAULT_MAX_N_OBS = 10_000

MAX_ITER_STEPS = 100_000

CONDITION_KEY = "condition"

SOURCE_STATE = "src_cell"
TARGET_STATE = "tgt_cell"

SOURCE_COUPLING_STATE_LIN = "src_xy_cell_coupling"
TARGET_COUPLING_STATE_LIN = "tgt_xy_cell_coupling"
SOURCE_COUPLING_STATE_QUAD = "src_xx_cell_coupling"
TARGET_COUPLING_STATE_QUAD = "tgt_yy_cell_coupling"

ORIGINAL_INDEX_KEY = "_scflow_original_index"  # TODO: rename this
