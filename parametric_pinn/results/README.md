# Paired PINN comparison data

All physical quantities are dimensionless. The original direct-solver study remains under the repository-root `results/`. These files describe a new, independent, same-field comparison.

| File | Meaning |
|---|---|
| `training_manifest.json` | Training parameters, seed, streams, data fingerprints, versions, selected epoch and depth, offline times and explicitly unmet validation target |
| `training_history.csv` | Actual physics-loss history, curriculum depth, gradient norms, epoch timings and independent validation errors |
| `validation_depths.csv` | Frozen-checkpoint depth sweep 0,32,64,128,256,512,1024,2048 on independent validation fields |
| `untrained_validation.json` | K=256 pretraining control on the same validation data |
| `evaluation_manifest.json` | Independent test streams, sample counts, evaluation times and comparison conventions |
| `paired_predictions.csv.gz` | Every paired prediction, head/flux/conservation metric and case identifier |
| `test_summary.csv` | Accuracy, physics diagnostics and stochastic summaries by test split and method |
| `scenario_comparison.csv` | Same summaries by δ,σ and method for the nine in-domain scenarios |
| `physical_parameters_only_baseline.csv` | Non-identifiability demonstration: variation in reference G when all δ,p,σ values are identical |
| `sobol_paired_outputs.csv.gz` | Paired reference/fast/accurate outputs for 2,048 base-pair rows × seven hybrids per method |
| `sobol_comparison.csv` | First/total Sobol indices, bootstrap intervals and paired index differences |
| `timing_benchmark.csv` | All three repetitions; batch 1,16,64; graph preprocessing, tensor/neural core, total, sparse scalar/full-field time, speedup and break-even |
| `example_fields.npz` | 24 small stored examples: edge fields, reference heads and PINN heads; metadata in the adjacent JSON |
| `difficult_example.npz` | Full field and predictions for the largest scenario-test free-node head RMSE; selection metadata in JSON |
| `verification.json` | Actual automated test totals parsed from pytest JUnit output |
| `audit.json` | Independent integrity audit, frozen-checkpoint hash and verified checks |

## Test splits

`scenario`: 9 × 1,000 new realizations with δ=0,0.1,0.25 and σ=0,1,2. A replicate shares underlying fields across scenarios, so scenario-to-scenario samples are paired; independence holds within a scenario. `continuous`: 1,000 new continuous-domain fields. OOD families have 250 each: `ood_sigma3`, `ood_fully_connected` (p=1), `ood_size15` and `ood_size41`. For size transfer, neural weights are unchanged and graph-index buffers are regenerated. These are exploratory out-of-domain tests, not advertised as a uniformly valid deployment domain.

The test seed is 20261105 with separate SeedSequence streams 201–205. Training uses 101, validation 102, Sobol 301, bootstrap 302 and timings 401. The generation order and all physical fields can be reproduced from `scripts/train.py` and `scripts/evaluate.py`. Training retains spanning cases because zero-flow cases are imposed exactly by topology. No reference solution labels are used by the training loss; validation reference solutions guide checkpoint selection.

## Prediction columns and conventions

`g_ref` is the verified sparse solver's inlet flux for unit head difference. `g_pred` is **Dirichlet energy of the approximate neural head**. It is an upper-bound conductance estimate for an admissible head, up to arithmetic error; it is not automatically identical to `inlet_flow` or `outlet_flow`. Those raw fluxes, their imbalance and an energy/flux gap are stored separately. `max_residual_abs` is the maximum interior Kirchhoff balance defect; `normalized_residual` is the maximum normalized `h−Ph` defect on spanning free nodes.

`relative_g_error = g_pred/g_ref−1` is undefined and stored as blank/NaN when `g_ref=0`. Relative-error summaries include **positive reference cases only**. Exact zero-flow agreement comes from connected-component preprocessing, not neural classification. `head_rmse` and `head_max_error` consider spanning free nodes, excluding prescribed and one-boundary constant-head nodes; summary head RMSE is the average of per-realization RMSEs. `relative_edge_flow_l2_error` uses the norm of the full exact edge-flow vector on positive-flow cases. Floating components have undefined absolute head and zero edge flow.

Paired mean-bias intervals use ±1.96 times the sample standard error of within-field model-minus-reference differences. They quantify finite test-sample uncertainty in mean bias, not prediction uncertainty or repeated-training uncertainty. One training seed was used. The 95th/99th percentile error columns describe the empirical **distribution of approximation error**, not an ensemble-flow percentile.

Methods: `geodesic` uses resistance shortest-distance head alone; `untrained_control` uses the same graph physics recurrence at K=256 with α=1.90, β=0.96 and no learned correction; `pinn_fast` uses trained weights at K=256; `pinn_accurate` uses those same weights at K=2048. The latter is the most accurate tested capped mode; the desired validation 99th-percentile G error under 0.1% was **not achieved**, and this is explicitly recorded. No adaptive stopping, sparse post-correction or per-input weight fitting is used. The model's own fixed-depth iterative head updates still occur per new input.

## Sensitivity and timing

Sensitivity columns follow `method__A`, `method__B`, and `method__AB_factor`, for independent groups δ,σ,U,W,Z. The Jansen estimator uses pooled A/B variance. Bootstrap resamples complete base-pair rows, preserving all methods and hybrids; paired difference intervals compare the neural estimate with the same-field sparse reference. Raw indices and intervals are not clipped. The original 16,384-pair study has a different random sample; only this paired design isolates approximation error from sampling differences.

Timing excludes field generation and checkpoint loading for all warm runs, but includes all per-input graph preprocessing and tensor assembly for the PINN. The neural model uses two CPU threads and float64. The direct method is timed both for G only and for full head/flux output. `speedup_vs_sparse_g` is sparse elapsed time divided by complete neural elapsed time; **values below one mean the neural method is slower**. A blank break-even means the measured per-case neural cost is not lower, so no positive offline-cost amortization point exists. Offline training time is separate in the training manifest.
