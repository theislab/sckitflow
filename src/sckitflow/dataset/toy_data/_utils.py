# Import all dataset classes
from sckitflow.dataset.toy_data._blobs import BlobsDataset
from sckitflow.dataset.toy_data._checkerboard import CheckerboardDataset
from sckitflow.dataset.toy_data._circles import CirclesDataset
from sckitflow.dataset.toy_data._moons import MoonsDataset
from sckitflow.dataset.toy_data._scurve import SCurveDataset
from sckitflow.dataset.toy_data._swissroll import SwissRollDataset

_DUMMY_DATASETS_CLS_REGISTRY = {
    "blobs": BlobsDataset,
    "checkerboard": CheckerboardDataset,
    "circles": CirclesDataset,
    "moons": MoonsDataset,
    "s_curve": SCurveDataset,
    "swiss_roll": SwissRollDataset,
}


def get_toy_dataset(dataset_name: str, **kwargs):
    """
    Unified function to generate any dummy dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to generate. Options:
        - 'blobs': Gaussian clusters
        - 'checkerboard': Grid pattern
        - 'circles': Concentric circles
        - 'moons': Interleaving half-circles
        - 's_curve': S-shaped 3D manifold
        - 'swiss_roll': Rolled 3D manifold

    **kwargs : additional parameters
        Dataset-specific parameters

    Returns
    -------
    adata : AnnData
        Generated dataset wrapped in AnnData format

    Raises
    ------
    ValueError : If dataset_name is not recognized
    """
    # Instantiate the appropriate dataset class
    dataset_class = _DUMMY_DATASETS_CLS_REGISTRY[dataset_name]
    adata = dataset_class(**kwargs)

    return adata
