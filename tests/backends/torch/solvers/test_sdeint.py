import pytest
import torch
from torch import Tensor


class DriftWrapperVF:
    def __init__(self, drift_fn):
        self.drift_fn = drift_fn
        self.last_get_vf_fn_kwargs: dict[str, object] | None = None

    def get_vf_fn(self, **vf_kwargs):
        self.last_get_vf_fn_kwargs = dict(vf_kwargs)

        def f(t: Tensor, x: Tensor) -> Tensor:
            return self.drift_fn(t, x)

        return f


@pytest.fixture
def SDESolver():
    from sc_flow.backends.torch.solvers.sde_solver import SDESolver

    return SDESolver


class TestSDESolverInitialization:
    def test_default_initialization(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics)
        assert solver.method == "euler"
        assert solver.noise_type == "diagonal"
        assert solver.device_id.type == "cpu"

    def test_ito_sde_type(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, sde_type="ito")
        assert solver.sde_type.__name__ == "SDEIto"

    def test_stratonovich_sde_type(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, sde_type="stratonovich")
        assert solver.sde_type.__name__ == "SDEStratonovich"

    @pytest.mark.parametrize("noise_type", ["scalar", "diagonal", "general", "additive"])
    def test_noise_types(self, SDESolver, noise_type: str):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type=noise_type)
        assert solver.noise_type == noise_type

    def test_custom_method(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, method="milstein")
        assert solver.method == "milstein"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, device_id="cuda")
        assert solver.device.type == "cuda"


class TestSDESolverBasicFunctionality:
    @pytest.fixture
    def simple_drift(self):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        return drift

    @pytest.fixture
    def constant_diffusion(self):
        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        return diffusion

    def test_solve_returns_trajectory(self, SDESolver, simple_drift, constant_diffusion):
        drift_vf = DriftWrapperVF(simple_drift)
        dynamics = (drift_vf, constant_diffusion)

        solver = SDESolver(dynamics=dynamics)
        source = torch.randn(10, 2)
        time = torch.linspace(0, 1, 11)

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert result.shape[0] == len(time)
        assert result.shape[1:] == source.shape

    def test_solve_returns_final_state(self, SDESolver, simple_drift, constant_diffusion):
        drift_vf = DriftWrapperVF(simple_drift)
        dynamics = (drift_vf, constant_diffusion)

        solver = SDESolver(dynamics=dynamics)
        source = torch.randn(10, 2)
        time = torch.linspace(0, 1, 11)

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=False,
            solver_kwargs={},
        )

        assert result.shape == source.shape

    def test_solve_with_scalar_noise(self, SDESolver, simple_drift):
        def scalar_diffusion(t: Tensor, x: Tensor) -> Tensor:
            batch_size = x.shape[0]
            state_dim = x.shape[1]
            return torch.ones(batch_size, state_dim, 1) * 0.1

        drift_vf = DriftWrapperVF(simple_drift)
        dynamics = (drift_vf, scalar_diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="scalar")
        source = torch.randn(5, 3)
        time = torch.linspace(0, 1, 6)

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert result.shape[0] == len(time)


class TestSDEMathematicalProperties:
    def test_ornstein_uhlenbeck_mean_reversion(self, SDESolver):
        theta = 0.15
        mu = 0.0
        sigma = 0.2

        def drift(t: Tensor, x: Tensor) -> Tensor:
            return theta * (mu - x)

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * sigma

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="diagonal")
        source = torch.ones(1000, 1) * 2.0
        time = torch.linspace(0, 10, 101)

        torch.manual_seed(42)
        trajectory = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        final_mean = trajectory[-1].mean().item()
        assert abs(final_mean - mu) < abs(source.mean().item() - mu)

    def test_geometric_brownian_motion(self, SDESolver):
        mu = 0.05
        sigma = 0.2

        def drift(t: Tensor, x: Tensor) -> Tensor:
            return mu * x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return sigma * x

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="diagonal")
        source = torch.ones(100, 1) * 100.0
        time = torch.linspace(0, 1, 11)

        torch.manual_seed(42)
        trajectory = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert (trajectory > 0).all()

    def test_deterministic_limit(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def zero_diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.zeros_like(x)

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, zero_diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="diagonal")
        source = torch.ones(1, 1)
        time = torch.linspace(0, 1, 11)

        torch.manual_seed(42)
        result1 = solver.solve(
            source,
            time,
            rtol=1e-5,
            atol=1e-6,
            return_trajectory=True,
            solver_kwargs={},
        )

        torch.manual_seed(123)
        result2 = solver.solve(
            source,
            time,
            rtol=1e-5,
            atol=1e-6,
            return_trajectory=True,
            solver_kwargs={},
        )

        torch.testing.assert_close(result1, result2, rtol=1e-4, atol=1e-5)


class TestSDEEdgeCases:
    def test_single_time_point(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics)
        source = torch.randn(5, 2)
        time = torch.tensor([0.0])

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        torch.testing.assert_close(result[0], source, rtol=1e-5, atol=1e-6)

    def test_different_batch_sizes(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics)
        time = torch.linspace(0, 1, 6)

        for batch_size in [1, 10, 100]:
            source = torch.randn(batch_size, 3)
            result = solver.solve(
                source,
                time,
                rtol=1e-3,
                atol=1e-4,
                return_trajectory=True,
                solver_kwargs={},
            )
            assert result.shape == (len(time), batch_size, 3)

    def test_tolerance_parameters(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics)
        source = torch.ones(10, 1)
        time = torch.linspace(0, 1, 21)

        result_loose = solver.solve(
            source,
            time,
            rtol=1e-2,
            atol=1e-3,
            return_trajectory=True,
            solver_kwargs={},
        )

        result_tight = solver.solve(
            source,
            time,
            rtol=1e-5,
            atol=1e-6,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert result_loose.shape == result_tight.shape
        assert result_loose.shape == (len(time), 10, 1)


class TestSDEReproducibility:
    def test_different_seeds_give_different_results(self, SDESolver):
        import torchsde

        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -0.5 * x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.3

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics)
        source = torch.randn(20, 2)
        time = torch.linspace(0, 1, 11)

        torch.manual_seed(111)
        bm1 = torchsde.BrownianInterval(
            t0=time[0],
            t1=time[-1],
            size=(20, 2),
            levy_area_approximation="space-time",
        )
        result1 = solver.solve(
            source,
            time,
            rtol=1e-4,
            atol=1e-5,
            return_trajectory=True,
            solver_kwargs={"bm": bm1},
        )

        torch.manual_seed(222)
        bm2 = torchsde.BrownianInterval(
            t0=time[0],
            t1=time[-1],
            size=(20, 2),
            levy_area_approximation="space-time",
        )
        result2 = solver.solve(
            source,
            time,
            rtol=1e-4,
            atol=1e-5,
            return_trajectory=True,
            solver_kwargs={"bm": bm2},
        )

        assert not torch.allclose(result1, result2, rtol=1e-3)


class TestSDEMultidimensional:
    def test_2d_coupled_system(self, SDESolver):
        def drift(t: Tensor, x: Tensor) -> Tensor:
            dx0 = -x[..., 0] + 0.5 * x[..., 1]
            dx1 = 0.5 * x[..., 0] - x[..., 1]
            return torch.stack([dx0, dx1], dim=-1)

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.1

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="diagonal")
        source = torch.randn(10, 2)
        time = torch.linspace(0, 2, 21)

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert result.shape == (len(time), 10, 2)

    def test_high_dimensional(self, SDESolver):
        dim = 50

        def drift(t: Tensor, x: Tensor) -> Tensor:
            return -0.1 * x

        def diffusion(t: Tensor, x: Tensor) -> Tensor:
            return torch.ones_like(x) * 0.05

        drift_vf = DriftWrapperVF(drift)
        dynamics = (drift_vf, diffusion)

        solver = SDESolver(dynamics=dynamics, noise_type="diagonal")
        source = torch.randn(5, dim)
        time = torch.linspace(0, 1, 11)

        result = solver.solve(
            source,
            time,
            rtol=1e-3,
            atol=1e-4,
            return_trajectory=True,
            solver_kwargs={},
        )

        assert result.shape == (len(time), 5, dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
