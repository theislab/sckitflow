from sc_flow.flow.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

# NB: ``_device`` (torch↔jax DLPack coupling) is intentionally NOT imported here — it pulls torch, and this
# package must stay importable in a jax-only environment. The torch objective imports it directly.
__all__ = ["independent_coupling", "ot_linear_coupling", "ot_quadratic_coupling"]
