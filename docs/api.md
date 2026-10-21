# API

## Model

```{eval-rst}
.. module:: sckitflow
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    Model
    ModelConfig
    predict_adata
    get_logger
```

### Samplers

```{eval-rst}
.. autosummary::
    :toctree: generated

    SamplerConfig
    TimeSamplerConfig
    NoiseSamplerConfig
    UniformTimeSamplerConfig
    IsotropicGaussianNoiseSamplerConfig
```

## Data

```{eval-rst}
.. module:: sckitflow.data
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    data.DataManagerConfig
    data.DataManager
    data.FlowDataModule
    data.DataDimensions
    data.Loader
    data.EvalLoader
    data.LoaderKwargs
```

### Group encoders

```{eval-rst}
.. autosummary::
    :toctree: generated

    data.LabelEncoderConfig
    data.OneHotEncoderConfig
```

### Toy datasets

```{eval-rst}
.. module:: sckitflow.dataset.toy_data
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    dataset.toy_data.get_toy_dataset
```

## Training

```{eval-rst}
.. module:: sckitflow.trainer
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    trainer.TrainingPlan
```

## Methods

```{eval-rst}
.. module:: sckitflow.core.methods
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.methods.CFMTrainingConfig
    core.methods.CFMTraining
    core.methods.MatchedTrainingConfig
    core.methods.MatchedTrainingMethod
    core.methods.ODEInferenceConfig
    core.methods.ODEInference
    core.methods.TrainingMethodConfig
    core.methods.InferenceMethodConfig
    core.methods.FlowMethodConfig
    core.methods.AbstractMethod
    core.methods.AbstractFlowMethod
    core.methods.SupportsTraining
    core.methods.SupportsInference
    core.methods.BaseMatcher
    core.methods.Matcher
```

## Networks

```{eval-rst}
.. module:: sckitflow.core.nn
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.nn.ModuleConfig
    core.nn.MLPVelocityConfig
    core.nn.MLPVelocity
    core.nn.BaseVelocityField
    core.nn.MLP
    core.nn.Resnet1d
    core.nn.SetEncoder
    core.nn.TokenAttentionPooling
    core.nn.SeedAttentionPooling
    core.nn.FunctionalModule
    core.nn.get_time_features_fn
    core.nn.make_custom_time_features
    core.nn.torch_cfm_time_features
    core.nn.ott_jax_time_features
```

## Couplings

```{eval-rst}
.. module:: sckitflow.core.coupling
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.coupling.CouplingConfig
    core.coupling.IndependentCouplingConfig
    core.coupling.ExactCouplingConfig
    core.coupling.SinkhornCouplingConfig
    core.coupling.PartialCouplingConfig
    core.coupling.UnbalancedCouplingConfig
    core.coupling.GromovWassersteinCouplingConfig
    core.coupling.FusedGromovWassersteinCouplingConfig
    core.coupling.independent_coupling
    core.coupling.ot_linear_coupling
    core.coupling.ot_quadratic_coupling
```

## Probability paths

```{eval-rst}
.. module:: sckitflow.core.probability_paths
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.probability_paths.ProbabilityPathConfig
    core.probability_paths.LinearDiracProbabilityPathConfig
    core.probability_paths.LinearGaussianProbabilityPathConfig
    core.probability_paths.SchrodingerBridgeProbabilityPathConfig
    core.probability_paths.VariancePreservingDiracProbabilityPathConfig
    core.probability_paths.BaseProbabilityPath
    core.probability_paths.LinearDiracProbabilityPath
    core.probability_paths.LinearGaussianProbabilityPath
    core.probability_paths.SchrodingerBridgeProbabilityPath
    core.probability_paths.VariancePreservingDiracProbabilityPath
```

## Solvers

```{eval-rst}
.. module:: sckitflow.core.solvers
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.solvers.BaseSolver
    core.solvers.ODESolver
    core.solvers.SDESolver
```

## Metrics

```{eval-rst}
.. module:: sckitflow.core.metrics
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.metrics.EnergyDistance
    core.metrics.MaximumMeanDiscrepancy
```

## Surrogate

```{eval-rst}
.. module:: sckitflow.core.surrogate
.. currentmodule:: sckitflow

.. autosummary::
    :toctree: generated

    core.surrogate.SurrogatePotential
    core.surrogate.ReadoutConfig
    core.surrogate.MeanSamplesReadoutConfig
    core.surrogate.PotentialConfig
    core.surrogate.SquaredErrorPotentialConfig
    core.surrogate.mean_samples
    core.surrogate.squared_error
```
