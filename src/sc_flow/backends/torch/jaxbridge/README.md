# JAX-compute / torch-optimize CellFlow

A CellFlow variant that trains under **torch + PyTorch Lightning** while **all
numerical computation stays in JAX**. torch's role is strictly: hold the
parameters, run the Lightning training loop, and drive the optimizer step. The
flow-matching math — probability path, velocity-field evaluation, FM/encoder
loss, and its gradient — is CellFlow's own JAX code, called here, never
reimplemented in torch.

## Who owns the parameters

**torch owns them.** They live as `torch.nn.Parameter` leaves inside
`CellFlowJaxModule` (a `torch.nn.ParameterList`) — the single source of truth.
The torch optimizer updates them in place. JAX keeps **no** persistent parameter
state; it only ever sees a per-step view.

Initial values come from `vf.init(...)` (a flax pytree). At construction we
`tree_flatten` that pytree once, remember the `treedef`, and clone each leaf into
an independent `nn.Parameter`. `treedef` is how we reassemble a flax pytree from
the flat torch list later.

## How the DLPack bridge works

Per training step:

1. **Mirror params → JAX.** Each `nn.Parameter` is `.detach().contiguous()`-ed
   and exported through DLPack (`jax.dlpack.from_dlpack`) into a `jax.Array` — a
   **zero-copy view** over the same device buffer, not a copy. `tree_unflatten`
   with the stored `treedef` rebuilds the flax params pytree.
2. **Compute in JAX.** A jitted `jax.value_and_grad` of CellFlow's OT-FM loss
   returns `(loss, grads)`, with `grads` sharing the params' tree structure.
3. **Move the gradient back to torch.** This is the key mechanism —
   `JaxLossFunction`, a `torch.autograd.Function`:
   - `forward` runs steps 1–2 and stashes the per-leaf JAX gradients (mirrored to
     torch tensors via DLPack) on `ctx`; it returns the loss as a torch scalar.
   - `backward` returns `grad_output * grad` for each parameter (chain rule for a
     scalar loss). torch accumulates these into `param.grad`.
4. **Optimize in torch.** Lightning calls `loss.backward()` then
   `optimizer.step()`, updating the torch-owned parameters.

Because the value *and* the gradient are exactly what JAX produced, a torch
`loss.backward()` reproduces the pure-JAX CellFlow gradient bit-for-bit (the
bridge adds only DLPack views and a flatten/unflatten). This is verified in
`tests/backends/torch/jaxbridge/test_jax_bridge.py`.

## Scope: this mirrors the *loss + gradient*, not all of `step_fn`

`make_fm_value_and_grad` reproduces the inner `loss_fn` of CellFlow's
`OTFlowMatching` **verbatim**, so for identical inputs the loss and gradient are
bit-exact. It does **not** include the parts of `OTFlowMatching.step_fn` that live
*outside* the loss: OT `match_fn` resampling of the source/target pairs, and the
per-step sampling of `time` and `encoder_noise`. Those are the dataloader's job
here — a batch already carries `time`, `source`, `target`, `encoder_noise`, and
`conditions`. So "equivalent to CellFlow training" holds only if the caller
reproduces matching + sampling upstream; the narrow, tested claim is *same
loss+grad for identical inputs*.

## Device & dtype

DLPack only shares a buffer when **both frameworks address the same physical
device** (e.g. both on `cuda:0`); `assert_same_device` guards that the torch
parameters share one device (it does not, and cannot portably, assert JAX's
default device — keep JAX and the batch on that same device). A grad-requiring
torch tensor cannot be DLPack-exported, so `torch_to_jax` detaches first — safe,
because the gradient path runs through `JaxLossFunction`, not through the buffer
view. The parameter dtype is **preserved** from the flax pytree (not forced to
`float32`), so the bit-exact claim also holds at float64 under
`jax.config.update("jax_enable_x64", True)`; non-floating leaves are rejected.

**GPU is untested here.** The dev machine is CPU-only. On CUDA, torch and JAX use
separate streams, so the zero-copy hand-off needs the producer kernel to finish
before the consumer reads (see the `.. warning::` in `_bridge.py`). Validate on a
real GPU and add an explicit device sync around the DLPack transfer if a race
appears — it will not surface on CPU.

> Note: JAX's older `jax.dlpack.to_dlpack` was removed in jax ≥ 0.11. This code
> uses the current protocol: `jax.dlpack.from_dlpack(torch_tensor)` and
> `torch.utils.dlpack.from_dlpack(jax_array)`, both of which consume the source's
> `__dlpack__` directly.

## Usage sketch

```python
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow._compat import ConstantNoiseFlow
from sc_flow.backends.torch.jaxbridge import CellFlowJaxModule

vf = ConditionalVelocityField(output_dim=D, max_combination_length=..., ...)
pp = ConstantNoiseFlow(sigma=0.0)
params = vf.init(...)["params"]

module = CellFlowJaxModule(vf, pp, params, lr=1e-4)
# batches are dicts of torch tensors: time (n,1), source/target (n,d),
# encoder_noise (n, embedding_dim), conditions {covariate: (n, max_comb, cond_dim)}
trainer.fit(module, dataloader)
```
