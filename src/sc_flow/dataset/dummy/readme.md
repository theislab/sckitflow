# Dummy Dataset Generator Module

## Overview
This module generates synthetic datasets from scikit-learn and wraps them in `sc.AnnData` format for benchmarking.

## Module Structure

```
dummy_datasets/
├── _dummy_datasets.py        # Main unified interface
├── _base.py                  # Abstract base class and common utilities
├── _blobs.py
├── _checkerboard.py
├── _circles.py
├── _moons.py
├── _s_curve.py
├── _swiss_roll.py
└── readme.md
```

### Architecture Design

- **`dummy_datasets.py`**: Main entry point with `dummy()` class
  - 3 Methods: `get()`, `list_datasets()`, `get_dataset_info()`

- **`base.py`**: Contains `BaseDummyDataset` abstract class
  - Defines the interface all datasets must follow

- **Individual dataset files**: Each inherits from `BaseDummyDataset`

## Available Datasets

### 1. [**Blobs**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
- `n_samples`: 'Number of data points (default: 1000)',
- `n_features`: 'Number of dimensions (default: 2)',
- `centers`: 'Number of clusters (default: 3)',
- `cluster_std`: 'Cluster spread (default: 1.0)',
- `center_box`: 'Bounding box for cluster centers (default: (-10.0, 10.0))',
- `shuffle`: 'Whether to shuffle samples (default: True)',
- `random_state`: 'Seed for random number generator (default: 42)'

### 2. [**Checkerboard**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_checkerboard.html)
- `n_samples`: 'Number of data points (default: 1000)',
- `n_features`: 'Number of dimensions (default: 2)',
- `noise`: 'Noise level (default: 0.0)',
- `minval`: 'Minimum value for grid (default: 10)',
- `maxval`: 'Maximum value for grid (default: 100)',
- `shuffle`: 'Whether to shuffle samples (default: True)',
- `random_state`: 'Seed for random number generator (default: 42)',
- `n_clusters`: 'Grid size as (rows, cols) (default: (3, 3))'

### 3. [**Circles**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
- `n_samples`: 'Number of data points (default: 1000)',
- `factor`: 'Inner to outer circle ratio (default: 0.8)',
- `noise`: 'Noise level (default: 0.0)',
- `random_state`: 'Seed for random number generator (default: 42)'

### 4. [**Moons**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- `n_samples`: 'Number of data points (default: 1000)',
- `noise`: 'Noise level (default: 0.05)',
- `shuffle`: 'Whether to shuffle samples (default: True)',
- `random_state`: 'Seed for random number generator (default: 42)'

### 5. [**S-Curve**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html)
- `n_samples`: 'Number of data points (default: 1000)',
- `noise`: 'Noise level (default: 0.0)',
- `random_state`: 'Seed for random number generator (default: 42)'

### 6. [**Swiss Roll**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html#sklearn.datasets.make_swiss_roll)
- `n_samples`: 'Number of data points (default: 1000)',
- `noise`: 'Noise level (default: 0.0)',
- `hole`: 'Whether to include a hole in the Swiss roll (default: False)',
- `random_state`: 'Seed for random number generator (default: 42)'

## Usage Examples

```python
# Import
from sc_flow.data.dummy import dummy

dummy_func = dummy()
# Generate any dataset by name
adata_blobs   = dummy_func.get("blobs", n_samples=1000, centers=4, cluster_std=1.5).adata
adata_circles = dummy_func.get("circles", n_samples=800, factor=0.6, noise=0.1).adata
adata_moons   = dummy_func.get("moons", n_samples=1000, noise=0.05).adata
adata_s_curve = dummy_func.get("s_curve", n_samples=15000, noise=0.9, random_state=73).adata
adata_swiss   = dummy_func.get("swiss_roll", n_samples=2000, noise=0.5).adata
adata_checker = dummy_func.get("checkerboard", n_samples=500, n_clusters=(4, 4)).adata
```

```python
# Import
from sc_flow.data.dummy import dummy

dummy_func = dummy()
dummy_func.list_datasets()
```

```python
# Import
from sc_flow.data.dummy import dummy

dummy_func = dummy()
dymmy_func.get_dataset_info('blobs')
```

## AnnData Structure

### Observation Names (obs_names)
- Format: `cell_0`, `cell_1`, ..., `cell_n`

### Variable Names (var_names)
- Format: `feature_0`, `feature_1`, ..., `feature_m`

### Metadata (.uns['dataset_info'])

All datasets include:
- `name`: Dataset type identifier
- `n_observations`: Number of data points
- `n_features`: Number of dimensions
- `random_state`: Random seed used
- `generator`: Class name that generated the dataset
