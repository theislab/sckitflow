"""Quarantined subsystems not on the active torch + Lightning train path.

Moved here during the train-path rewire (data-layer strip + backend flatten). Nothing on the active
path imports this package; modules here may have stale internal imports and are kept only for reference
/ future salvage. Do not import from ``sc_flow.legacy`` in active code.
"""
