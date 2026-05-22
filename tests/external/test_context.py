import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import pytest

from sc_flow.external._context import ExternalModelContext


# -----------------------------------------------------------------------------
# Dummy model classes (modified to accept *args)
# -----------------------------------------------------------------------------
class DummyModel:
    def __init__(self, value=42):
        self.value = value

    def __call__(self, *args, **kwargs):
        return sum(args) + self.value

    def predict(self, *args, **kwargs):
        x = args[0] if args else 0
        return x * 2 + self.value

    def forward(self, *args, **kwargs):
        x = args[0] if args else 0
        return x - self.value


# -----------------------------------------------------------------------------
# Fixtures (unchanged)
# -----------------------------------------------------------------------------
@pytest.fixture
def dummy_model_path():
    model = DummyModel(value=10)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        cloudpickle.dump(model, f)
        path = f.name
    yield path
    Path(path).unlink()


@pytest.fixture
def dummy_torch_path():
    import torch

    tensor = torch.tensor([1, 2, 3])
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(tensor, f.name)
        path = f.name
    yield path
    Path(path).unlink()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestExternalModelContext:
    def test_lazy_loading(self, dummy_model_path):
        ctx = ExternalModelContext(model_path=dummy_model_path)
        assert not ctx.is_loaded()
        model = ctx.load()
        assert ctx.is_loaded()
        assert isinstance(model, DummyModel)
        assert model.value == 10

    def test_call_without_forward_method(self, dummy_model_path):
        ctx = ExternalModelContext(model_path=dummy_model_path)
        # Use only positional arguments – keyword 'y' would not be forwarded to __call__
        result = ctx(5, 3)  # DummyModel.__call__(5, 3) -> 5+3+10 = 18
        assert result == 18

    def test_forward_method(self, dummy_model_path):
        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            forward_method="predict",
            forward_args=(5,),
            forward_kwargs={"extra": 1},
            runtime_args_first=False,  # Put forward_args before call-time args
        )
        # final args = (5, 2) -> predict(5) uses x=5 -> 5*2+10 = 20
        result = ctx(2, kw=3)  # call-time args (2,) appended after forward_args
        assert result == 20

    def test_runtime_args_first_false(self, dummy_model_path):
        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            forward_method="forward",
            forward_args=(3,),
            runtime_args_first=False,
        )
        # forward_args first, then call-time args => final args = (3, 2) if called with (2,)
        result = ctx(2)  # forward(x) -> x - value = 3 - 10 = -7
        assert result == -7

    def test_post_load_callable(self, dummy_model_path):
        def post_load(model, factor):
            model.value = model.value * factor
            return model

        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            post_load_callable=post_load,
            post_load_kwargs={"factor": 2},
        )
        model = ctx.load()
        assert model.value == 20  # original 10 * 2

    def test_custom_loader(self, dummy_model_path):
        def custom_loader(path, args, kwargs):
            with open(path, "rb") as f:
                data = pickle.load(f)
            # modify the loaded model
            data.value = 100
            return data

        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            loader=custom_loader,
        )
        model = ctx.load()
        assert model.value == 100

    def test_extension_loader_registration(self, tmp_path):
        # Create a dummy file with .test extension
        model_data = {"a": 1}
        file_path = tmp_path / "model.test"
        with open(file_path, "wb") as f:
            cloudpickle.dump(model_data, f)

        def test_loader(path, args, kwargs):
            with open(path, "rb") as f:
                return cloudpickle.load(f, *args, **kwargs)

        ExternalModelContext.register_loader(".test", test_loader)
        ctx = ExternalModelContext(model_path=str(file_path))
        data = ctx.load()
        assert data == {"a": 1}

    def test_missing_file_raises(self):
        ctx = ExternalModelContext(model_path="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ctx.load()

    def test_unload(self, dummy_model_path):
        ctx = ExternalModelContext(model_path=dummy_model_path)
        ctx.load()
        assert ctx.is_loaded()
        ctx.unload()
        assert not ctx.is_loaded()

    def test_reload(self, dummy_model_path):
        ctx = ExternalModelContext(model_path=dummy_model_path)
        model1 = ctx.load()
        ctx.unload()
        model2 = ctx.load()
        assert model1 is not model2

    def test_forward_method_missing_raises(self, dummy_model_path):
        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            forward_method="non_existent",
            allow_missing_forward_method=False,
        )
        with pytest.raises(AttributeError, match="non_existent"):
            ctx()

    def test_forward_method_missing_allowed(self, dummy_model_path):
        ctx = ExternalModelContext(
            model_path=dummy_model_path,
            forward_method="non_existent",
            allow_missing_forward_method=True,
        )
        # Should fall back to the object itself (callable)
        result = ctx(10)  # DummyModel.__call__(10) -> 10 + 0 + 10 = 20
        assert result == 20

    def test_not_callable_without_forward_method(self):
        # Create a context pointing to a non-callable object
        data = {"key": "value"}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cloudpickle.dump(data, f)
            path = f.name
        ctx = ExternalModelContext(model_path=path, forward_method=None)
        with pytest.raises(TypeError, match="not callable"):
            ctx()
        Path(path).unlink()

    def test_default_loader_torch(self, dummy_torch_path):
        ctx = ExternalModelContext(model_path=dummy_torch_path)
        tensor = ctx.load()
        import torch

        assert isinstance(tensor, torch.Tensor)
        assert tensor.tolist() == [1, 2, 3]

    @patch("cloudpickle.load")
    @patch("builtins.open")
    def test_fallback_loader_cloudpickle(self, mock_open, mock_cloudpickle_load, tmp_path):
        # For extension without registered loader, cloudpickle is used
        file_path = tmp_path / "model.custom"
        file_path.touch()
        mock_cloudpickle_load.return_value = "loaded"
        ctx = ExternalModelContext(model_path=str(file_path))
        result = ctx.load()
        assert result == "loaded"

    def test_repr(self, dummy_model_path):
        ctx = ExternalModelContext(model_path=dummy_model_path)
        assert "not loaded" in repr(ctx)
        ctx.load()
        assert "loaded" in repr(ctx)

    def test_model_directly_passed(self):
        """Pass an already instantiated model; should be ‘loaded’ immediately."""
        model = DummyModel(value=99)
        ctx = ExternalModelContext(model=model)
        assert ctx.is_loaded()
        loaded = ctx.load()
        assert loaded is model
        # Calling works as usual
        result = ctx(1, 2)  # 1+2+99 = 102
        assert result == 102

    def test_model_directly_with_forward_method(self):
        """Direct model with a forward_method and extra args."""
        model = DummyModel(value=10)
        ctx = ExternalModelContext(
            model=model,
            forward_method="predict",
            forward_args=(5,),
            runtime_args_first=False,
        )
        # forward_method="predict" -> predict(5) -> 5*2+10 = 20
        assert ctx() == 20

    def test_model_directly_with_forward_args_and_order(self):
        """Test runtime_args_first with a directly provided model."""
        model = DummyModel(value=0)
        # forward_method="forward" uses x -> x - value
        # default forward_args=(), we will set call-time and forward_args
        ctx = ExternalModelContext(
            model=model,
            forward_method="forward",
            forward_args=(10,),
            runtime_args_first=True,  # call-time args first
        )
        # call-time args: (3,) -> final args = (3, 10) -> forward uses first arg
        # forward(3) -> 3 - 0 = 3
        assert ctx(3) == 3

        # Reverse order
        ctx_runtime_last = ExternalModelContext(
            model=model,
            forward_method="forward",
            forward_args=(10,),
            runtime_args_first=False,
        )
        # final args = (10, 3) -> forward(10) -> 10 - 0 = 10
        assert ctx_runtime_last(3) == 10

    def test_model_directly_post_load_callable_not_applied(self):
        """When model is passed directly, post_load_callable is NOT called (current implementation)."""
        model = DummyModel(value=10)

        def post_load(m, factor):
            m.value *= factor
            return m

        ctx = ExternalModelContext(
            model=model,
            post_load_callable=post_load,
            post_load_kwargs={"factor": 2},
        )
        loaded = ctx.load()
        # Value should remain unchanged because load() returns the stored model without calling post_load
        assert loaded.value == 10
        assert loaded is model

    def test_model_and_path_both_passed(self, dummy_model_path):
        """If both model and model_path are given, the passed model is used until reload()."""
        direct_model = DummyModel(value=777)
        ctx = ExternalModelContext(model=direct_model, model_path=dummy_model_path)
        # initially, the direct model is active
        assert ctx.is_loaded()
        assert ctx.load() is direct_model
        assert ctx() == 777  # call with no args -> 777

        # reload discards the direct model and loads from path
        reloaded = ctx.reload()
        assert reloaded is not direct_model
        assert reloaded.value == 10  # value from the fixture (DummyModel(value=10))

    def test_model_directly_no_reload_possible(self):
        """Reload without a model_path should raise an error."""
        model = DummyModel(value=1)
        ctx = ExternalModelContext(model=model)
        with pytest.raises((TypeError, FileNotFoundError)):
            # Path(None) raises TypeError, or if the None gets converted – either way it fails
            ctx.reload()

    def test_model_directly_unload_and_reuse(self):
        """After unloading a direct model, trying to load without a path raises an error."""
        model = DummyModel()
        ctx = ExternalModelContext(model=model, model_path=None)
        ctx.unload()
        assert not ctx.is_loaded()
        with pytest.raises((TypeError, FileNotFoundError)):
            ctx.load()

    def test_model_directly_repr(self):
        """Repr should show ‘loaded’ when model is passed directly."""
        model = DummyModel()
        ctx = ExternalModelContext(model=model)
        assert "loaded" in repr(ctx)
        ctx.unload()
        assert "not loaded" in repr(ctx)
