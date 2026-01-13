# Preprocessing Module

Module contains scripts to preprocess data (np.array).

## Overview

This module provides preprocessors the following:
- `fit(X)`: Learn parameters from data
- `transform(X)`: Apply learned transformation
- `inverse_transform(X)`: Reverse the transformation
- `fit_transform(X)`: Convenience method combining fit and transform

## Architecture

```
preprocessing/
├── _base.py
├── _zscore.py
├── _pca.py
└── __init__.py
```

## Base Classes

### `PreprocessorParams`
Base dataclass for storing fitted parameters.

**Attributes:**
- `is_fitted`: Boolean flag if preprocessor is fitted
- `n_features`: Number of features in the original data

**Methods:**
- `validate_fitted()`: Was the preprocesser fitted?
- `validate_n_features(X)`: Inut has correct no. of features?

### `BasePreprocessor`
Abstract base class all preprocessors inherit from.

**Abstract Methods**:
- `fit(X)`: Learn parameters from data
- `transform(X)`: Apply transformation
- `inverse_transform(X)`: Reverse transformation

**Methods:**
- `fit_transform(X)`: Convenience method that combines fit() and transform()
- `is_fitted`: Property to check if fitted
- `params`: Property to access fitted parameters

## Preprocessors

### ZScorePreprocessor

Standardizes features to zero mean and unit variance.

**Parameters:**
- `epsilon` (float, default=1e-8): Small constant added to std to prevent division by zero

**Fitted Parameters** (`ZScoreParams`):
- `mean`: Mean of each feature, shape (n_features,)
- `std`: Standard deviation of each feature, shape (n_features,)

**Example:**
```python
from preprocessing import ZScorePreprocessor

# Create and fit
zscore = ZScorePreprocessor(epsilon=1e-8)
X_train_normalized = zscore.fit_transform(X_train)

# Transform new data using training statistics
X_test_normalized = zscore.transform(X_test)

# Reverse transformation
X_train_original = zscore.inverse_transform(X_train_normalized)
```

### PCAPreprocessor

Reduces dimensionality using Principal Component Analysis.

`Note: Does not account for data sparsity right now`

**Parameters:**
- `n_components` (int or None): Number of principal components to keep
  - If None: Keep all components (no reduction)
  - If int: Keep that many components

**Fitted Parameters** (`PCAParams`):
- `mean`: Mean of each feature, shape (n_features,)
- `components`: Principal component directions, shape (n_components, n_features)
- `explained_variance`: Variance explained by each component, shape (n_components,)
- `n_components`: Number of components kept

**Example:**
```python
from preprocessing import PCAPreprocessor

# Reduce to 50 dimensions
pca = PCAPreprocessor(n_components=50)
X_train_pca = pca.fit_transform(X_train)  # (n_samples, 50)

# Transform new data
X_test_pca = pca.transform(X_test)  # (n_samples, 50)

# Reconstruct
X_train_recon = pca.inverse_transform(X_train_pca)  # (n_samples, n_features)

# Check variance explained
variance_ratio = pca.explained_variance_ratio()
print(f"Variance explained: {variance_ratio.sum():.2%}")
```

## Parameter Storage Details

**ZScorePreprocessor** stores:
```python
ZScoreParams(
    mean=np.array([...]),      # Mean per feature
    std=np.array([...]),       # Std per feature
    is_fitted=True,
    n_features=1000
)
```

**PCAPreprocessor** stores:
```python
PCAParams(
    mean=np.array([...]),                # Mean per feature
    components=np.array([[...], ...]),   # PC directions
    explained_variance=np.array([...]),  # Variance per PC
    n_components=50,
    is_fitted=True,
    n_features=1000
)
```

### Accessing Parameters

```python
# Access parameters
print(zscore.params.mean)
print(pca.params.components)
```
