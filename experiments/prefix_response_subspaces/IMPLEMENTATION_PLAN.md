# Implementation and reuse plan

- Reuse `prefix_displacement.model_loading` and `prefix_displacement.runtime` for exact checkpoint resolution, precision, seeding, and single-process `torch.nn.DataParallel`.
- Reuse the replica-safe output-container and position-replacement hook primitives from `experiments.prefix_successor_subspaces.src.hooks`.
- Reuse the existing GSM8K trajectory schema and deterministic trajectories produced by `scripts/prepare_gsm8k_trajectories.py`.
- Implement split-local double centering independently because the earlier successor-subspace experiment cross-fits a training-token prefix mean into held-out tokens, which this experiment explicitly forbids.
- Keep candidate selection, prefix matching, geometry, permutation null, rank-0 recovery, figures, tables, and gate evaluation as independently resumable config-driven stages.

