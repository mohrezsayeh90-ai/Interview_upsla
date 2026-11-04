# Saved simulation outputs

All conductances, flows and heads are dimensionless. `manifest.json` records sample counts, RNG streams, software versions and measured runtime. The full experiment uses base seed 20261104.

| File | Contents |
|---|---|
| `scenario_samples.csv.gz` | 90,000 rows: replicate, n, delta, sigma, realized p, geff, spanning, geff_unit_mean |
| `scenario_summary.csv` | One statistical summary per (n,delta,sigma); nine rows |
| `paired_comparisons.csv` | Mean differences and paired Monte Carlo 95% intervals between endpoint δ or σ scenarios |
| `mc_convergence.csv` | Prefix summaries at 250,500,1000,2000,5000,10000 samples |
| `connectivity_samples.csv.gz` | Fixed-p sweep: 22,500 rows, n=25, σ=1, nine p values |
| `connectivity_summary.csv` | Summary of each fixed-p ensemble |
| `finite_size_samples.csv.gz` | n=15,25,41, δ=0 or 0.25, σ=1; n=25 reuses main-scenario samples |
| `finite_size_summary.csv` | Finite-size ensemble summaries |
| `sobol_samples.csv.gz` | 16,384 independent rows of seven flow outputs: A, B and one AB hybrid for each input group |
| `sobol_spanning_samples.csv.gz` | Corresponding binary outputs for existence of a spanning path |
| `sobol_summary.csv` | Jansen first/total estimates and 1,000-bootstrap percentile 95% intervals for geff and spanning |
| `sobol_convergence.csv` | Sobol prefix estimates at 1024,2048,4096,8192,16384 base pairs |
| `solver_diagnostics.csv` | 1,000 additional independent networks with balance, energy and head diagnostics |
| `audit.json` | Result-integrity audit outcome |

`geff` is the total inlet flux for unit head difference. It is **exactly zero** when no left-to-right path exists. `spanning` is independent of the positive conductance values. No numerical threshold is used to declare zero flow. `geff_unit_mean = geff * exp(-sigma**2/2)` is a supplementary exact rescaling, corresponding to an alternative active-link family with mean one.

Summary columns include the unconditional mean and sample standard deviation (`sd`); Monte Carlo standard error (`mcse`); normal 95% intervals for the mean; Wilson intervals for the spanning probability; unconditional 5th,50th,95th percentiles; a binomial order-statistic interval for the 95th percentile; positive-flow conditional statistics; coefficient of variation (`cv`); and the rescaled mean where applicable. `mean_unit_mean_g` is intentionally blank for experiments without that supplementary control. Probability and quantile intervals are under the assumed model; none include structural uncertainty.

Within each scenario, replicate rows are independent. Main scenarios share the same latent uniform connectivity draw and the same edge fields for a given replicate; cross-scenario comparisons are therefore paired. Within a fixed-p sweep replicate, fields are also shared. Size n=15 and n=41 use independent RNG streams. Do not pool correlated scenario rows or reused n=25 rows as additional independent samples.

Sensitivity uses independent groups in the order: `delta ~ U(0,0.25)`, `sigma ~ U(0,2)`, `p_latent ~ U(-1,1)`, the 1,200-dimensional topology uniform field, and the 1,200-dimensional standard-normal conductance field. Realized `p=0.5+delta*p_latent`. `AB_factor` replaces only that group of A with the corresponding group of B. The output is deterministic conditional on all groups. Jansen first and total estimates use the pooled A/B sample variance. Bootstrap resamples whole paired rows, preserving their dependence. Estimates are **not clipped** to [0,1]; small negative estimates/interval limits reflect estimator uncertainty. σ and the conductance field have exactly zero true total effect on spanning. First-order estimates for these known inactive groups can deviate slightly from zero.

This is grouped sensitivity to the stochastic simulator's complete output, not sensitivity of a noise-free conditional mean response surface. Importance depends on the chosen priors, their ranges, and the latent parameterization. Total indices include interactions and overlap. Existence-of-flow sensitivity and flow-magnitude sensitivity answer different questions.

Latent fields are reproducible from the seed/stream and replicate index using the generation order in `scripts/run_study.py`; they are not redundantly stored. Each array realization uses 1,200 potential edges in horizontal-then-vertical, row-major order. Entire replicate sets, outputs, summaries and scripts are retained so the analysis can be independently reproduced without external data.
