# import libraries
import scanpy as sc
import torch
import yaml

from sc_flow import SCFlow
from sc_flow.backends.torch.probability_paths import (
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)
from sc_flow.trainer._callbacks import WandBLogger


def train(config):
    """Trains the model based on the provided configuration."""
    # Extract configuration parameters
    output_dir = config["output_dir"]
    run_name = config["run_name"]

    data_path = config["data"]["path"]

    vf_decoder_mlp_kwargs = config["model"]["vf_decoder_mlp_kwargs"]
    device_id = config["model"]["device_id"]

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print(
                "MPS not available because the current macOS version is not 14.0+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    method_id = config["model"]["method_id"]
    # state_encoder_mlp_kwargs = config["model"]["state_encoder_mlp_kwargs"]
    time_features_id = config["model"]["time_features_id"]
    condition_id = config["model"]["conditioning_id"]
    # time_encoder_mlp_kwargs = config["model"]["time_encoder_mlp_kwargs"]
    probability_paths_id = config["model"]["probability_path_id"]
    probability_paths_sigma = config["model"]["probability_path_sigma"]

    batch_size = config["training"]["batch_size"]
    lr = float(config["training"]["lr"])
    n_train_steps = config["training"]["n_train_steps"]

    if probability_paths_sigma is None:
        probability_paths_sigma = 1.0

    # Parse configs
    if probability_paths_id == "schrodinger_bridge":
        probability_path = SchrodingerBridgeProbabilityPath(probability_paths_sigma)
    elif probability_paths_id == "linear_gaussian":
        probability_path = LinearGaussianProbabilityPath(probability_paths_sigma)
    elif probability_paths_id == "linear_dirac":
        probability_path = LinearDiracProbabilityPath(probability_paths_sigma)
    elif probability_paths_id == "variance_preserving_dirac":
        probability_path = VariancePreservingDiracProbabilityPath(probability_paths_sigma)
    else:
        raise ValueError(f"Unknown probability path: {probability_paths_id}")

    # Load dataset
    adata = sc.read_h5ad(data_path)
    adata.X = adata.X.toarray().astype("float32")

    # Register data
    SCFlow.register_adata(adata, sample_rep="X_pca")

    # Initiallize logger and model
    logger = WandBLogger(project_name=f"{config['run_name']}", log_dir="./logs", config=config)
    try:
        model = SCFlow(
            method_id=method_id,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
            # time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            # state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_features_id=time_features_id,
            conditioning_id=condition_id,
            device_id=device_id,
            probability_path=probability_path,
        )
    except NameError:
        model = SCFlow(
            method_id=method_id,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
            # time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            # state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_features_id=time_features_id,
            conditioning_id=condition_id,
            device_id=device_id,
            probability_path=probability_path,
        )

    print(next(model._method._module.parameters()).device)
    # train model

    model.train(
        adata, n_train_steps=n_train_steps, train_batch_size=batch_size, optim_kwargs={"lr": lr}, callbacks=[logger]
    )

    # Save the trained model
    model.save(filepath=f"{output_dir}/{run_name}/model.pt", allow_overwrite=True)


if __name__ == "__main__":
    # Load configuration from YAML file
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Train the model
    train(config)
