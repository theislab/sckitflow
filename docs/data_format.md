# Input data

A run reads one `AnnData`. Its schema, `DataManagerConfig` (`run.data.manager` in a run spec), names where
everything lives; nothing is inferred from the data.

## Layout

| What | Where | Schema field | Requirement |
| --- | --- | --- | --- |
| State | `.X`, or `.obsm[sample_rep]` | `sample_rep` | One float matrix, the same width for every cell. |
| Condition | `.obs[col]` | `categorical_conditions` | Per condition level, one or more columns. Several columns form a set, e.g. `drug: [drug1, drug2]` for combinations. |
| Condition embeddings | `.uns[rep]` | `categorical_reps` | A dict from every value in those columns to a vector, `(d,)` or `(1, d)`, all of one width. Include the control value when it fills an unused slot, e.g. `drug2 == "control"` for a single drug. |
| Continuous condition covariates | `.obsm[key]` | `continuous_conditions` | Per-cell arrays. |
| Groups | `.obs[col]` | `groups` | Columns controls are matched within, e.g. `cell_line`. |
| Group embeddings | `.uns[rep]`, or an encoder | `groups_reps`, `groups_encoding` | Like condition embeddings, or an encoder with an explicit vocabulary. |
| Response covariates | `.obs[col]` / `.obsm[key]` | `target_categorical_covs_dict` / `target_continuous_covs` | Categorical in `.obs`, continuous in `.obsm`. |
| Coupling representations | `.obsm[key]` | `rep_lin`, `source_rep_quad`, `target_rep_quad` | Only for Gromov-Wasserstein style couplings. |
| Split | `.obs[split_by]` | `split_by` | Written before the data reaches a run, e.g. by a `sckitflow.data.splitters` splitter: the split is the experiment's choice, not the model's. A `"split"` column the schema does not declare is an error. |

## Controls

Controls are not passed separately: they are cells of the same `AnnData`, picked out by the schema.

- `control_values_dict` maps a condition level to its control value. A cell is a control when every
  column of every listed level holds that value (`drug1 == drug2 == "control"`).
- `matched_pairs` names source and target groups outright instead.

A target is matched to controls sharing its `groups` values. The training loader keeps **only the control
cells in memory**, shared by every split, and streams the rest; the memory cost is the number of controls
times the state width. Prediction reads them per group.

## Several AnnData

Wherever an `AnnData` is taken, a list of them is too, read as one dataset: their `.obs` are stacked in list
order with categories unioned, and the controls of any of them can serve the targets of any other. They must
agree on the state width and on the feature names, and a split column (`split_by`) must be in each. Each
`.uns` table the schema reads must be in at least one of them: its entries are merged, an entry in several
must be the same in each, and every value looked up must have one. A list of zarr stores is opened lazily with `scfit.data.open_source(paths,
keys=..., cols=..., uns=...)`, listing every store explicitly. A store holding only controls is just one more
entry. A store only needs the representations of the cells it holds: the target side reads the stores holding
targets and the source side the stores holding sources, so e.g. `source_rep_quad` can live only with the controls.
