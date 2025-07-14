import numpy as np

PI = np.pi

DEFAULT_VF_LATENT_STATE_DIM = 32
DEFAULT_VF_LATENT_TIME_DIM = 16
DEFAULT_NUM_TIME_FEATURES = 256
DEFAULT_NUM_RESNET_LAYERS = 3

MU_T_FN_KEY = "compute_mu_t"
U_T_FN_KEY = "compute_ut"
SIGMA_T_FN_KEY = "compute_sigma_t"
