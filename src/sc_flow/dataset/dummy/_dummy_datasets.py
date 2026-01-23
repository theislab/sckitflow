# Import all dataset classes
from sc_flow.dataset.dummy._blobs import BlobsDataset
from sc_flow.dataset.dummy._checkerboard import CheckerboardDataset
from sc_flow.dataset.dummy._circles import CirclesDataset
from sc_flow.dataset.dummy._moons import MoonsDataset
from sc_flow.dataset.dummy._scurve import SCurveDataset
from sc_flow.dataset.dummy._swissroll import SwissRollDataset

_DUMMY_DATASETS_CLS_REGISTRY = {
    "blobs": BlobsDataset,
    "checkerboard": CheckerboardDataset,
    "circles": CirclesDataset,
    "moons": MoonsDataset,
    "s_curve": SCurveDataset,
    "swiss_roll": SwissRollDataset,
}


def create_dummydata(dataset_type: str, **kwargs):
    """
    Unified function to generate any dummy dataset.

    Parameters
    ----------
    dataset_type : str
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
    adata : sc.AnnData
        Generated dataset wrapped in AnnData format

    Raises
    ------
    ValueError : If dataset_type is not recognized
    """
    # Instantiate the appropriate dataset class
    dataset_class = _DUMMY_DATASETS_CLS_REGISTRY[dataset_type]
    adata = dataset_class(**kwargs)

    return adata
