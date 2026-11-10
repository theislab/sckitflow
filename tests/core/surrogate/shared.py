import torch


class Field(torch.nn.Module):
    """A velocity module wrapping a plain ``v(t, x)``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def get_vf_fn(self, condition_dict=None, source=None):
        return self.fn
