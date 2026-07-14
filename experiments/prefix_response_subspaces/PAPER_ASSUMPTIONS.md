# Final paper assumptions

1. Interaction energy is reported as `sum ||r||^2 / sum ||z-rowmean(z)||^2`.
2. Transformer block 0 output is the earliest available input-stage sanity
   condition; no embedding-to-block continuation intervention is implied.
3. Conditioned Global pools only analysis-training prefixes in the same fixed
   length and reasoning-progress bins. Each prefix covariance is divided by
   its trace before averaging, so every prefix contributes equal total energy.
4. Wrong-prefix controls use up to five prefixes from the analysis-training and
   dedicated donor pools, always from other problems in the same bins and
   preferring the same final token. Donor states are extracted explicitly;
   shortages are saved and cannot be silently replaced with prediction-matched
   donors. Only analysis-training prefixes enter the conditioned-Global basis.
5. Rank 64 is an evaluation label; because split-local centering removes one
   token degree of freedom, its effective numerical rank can be 63.
6. Content is appendix-only and never enters a primary paper gate.
7. The general and Math Qwen2.5 checkpoints are required to expose identical
   tokenizer IDs/text for every forced candidate before replication proceeds.
8. Replication supports only the within-Qwen2.5-family statement; it is not an
   architecture-general claim.
9. Rotation is the normalized projection distance
   `1 - ||U_i.T @ U_k||_F^2 / r`. Split halves are a deterministic seeded
   partition of each fold's training-token set and never use held-out tokens.
   If a small smoke/pilot half cannot identify the selected rank, the actual
   common effective rank is recorded per row; the full profile has enough
   training tokens per half for ranks through 64.
10. Wrong-prefix controls always match reasoning-progress bin. If an evaluation
    prefix has no other problem anywhere in its exact length/progress bin, the
    functional completeness path uses the nearest length bin and records the
    fallback. Primary Wrong-prefix geometry is also reported on the exact-bin
    subset, which must retain at least 99% of evaluation problems.
11. The paper permutation null keeps fitted subspaces fixed and permutes Local
    subspace ownership among evaluation prefixes within the same fixed
    length/progress stratum. This directly tests prefix-to-subspace alignment
    and avoids re-estimating identical SVD workloads 500 times. Legacy
    tokenwise-recentered permutation code remains available for audit but is
    not the fast paper inferential path.
