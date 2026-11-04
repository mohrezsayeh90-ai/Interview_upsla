# Presentation and speaker notes

## 1. Flow in a 2D random network

- Connectivity dominates flow variability over the selected ranges.
- 248,188 model evaluations; 24 independent verification tests.
- Completed analysis • 04 November 2026

Presenter notes: Open by distinguishing the probability of any flow from the amount of flow. All results are dimensionless.

## 2. A hierarchical random-network model

- Draw ONE p per realization: p = 0.5 + δU, U uniform on [−1, 1].
- Conditional on p: each link active independently with probability p.
- Active links: ln(g) ~ Normal(0, σ²); inactive links: g = 0.
- Left head = 1; right head = 0; no flow across top and bottom.

Presenter notes: The plotted examples use n=15 for legibility; the main study uses n=25. Unconditional edges share p and are correlated.

## 3. Solve only hydraulically relevant components

- Find components that touch both fixed-head boundaries.
- Solve L_FF h_F = −L_FB h_B on their free nodes.
- One-boundary clusters: constant head. Floating islands: arbitrary head, zero flow.
- G_eff = inlet flow / Δh; here Δh = 1.

Presenter notes: No artificial conductivity is added to disconnected links. Verification establishes numerical correctness, not field validation.

## 4. An explicit, reproducible experiment

- δ = 0, 0.10, 0.25; σ = 0, 1, 2.
- Compare n = 15, 25 and 41 to expose finite-size dependence.
- Seed 20261104; all realization-level outputs are retained.

Presenter notes: Parameter ranges are exploratory. They are justified by covering the transition and strong heterogeneity, rather than claimed to be calibrated priors.

## 5. More uncertain connectivity raises mean flow

- At σ=1, mean G: 0.0183 → 0.0992 as δ: 0 → 0.25.
- Mean ratio: 5.44×, despite unchanged E[p]=0.5.
- Reference spanning probability: 51.38%.
- High-p realizations contribute substantial flow; low-p realizations often remain at zero.

Presenter notes: Error bars are sampling confidence intervals, not predictive intervals. The trend is empirical over these ranges.

## 6. A mean cannot describe the output distribution

- Reference: δ=0.25, σ=1; P(G=0) = 48.62%.
- Mean 0.0992; median 0.0195; 95th percentile 0.3838.
- Mean given G>0: 0.1931.
- A single lognormal law is unsuitable for a distribution with an atom at zero.

Presenter notes: The histogram shows density in log space, conditional on positive flow. Full raw samples are provided.

## 7. A transition controls the response

- For an infinite square bond lattice, p_c = 0.5 (Kesten, 1980).
- A finite network has a smooth connectivity crossover.
- Nonlinear flow response explains the strong effect of the width δ.
- Shared p induces covariance δ²/3 between distinct link indicators.

Presenter notes: Do not interpret p=0.5 as a deterministic no-flow condition in a finite network, or sample a different p for each link.

## 8. Sensitivity of a stochastic model

- Use five independent groups; keep δ and realized p out of the same independent-input list.
- Jansen first and total indices: interactions are explicitly retained.
- 16,384 base pairs; 1,000 paired-row bootstrap resamples.
- The estimator also passes the analytic Ishigami benchmark.

Presenter notes: The expanded simulator is deterministic given these inputs. Importance depends on priors and latent parameterization. Sobol indices target unconditional G, including zeros.

## 9. Global connectivity is the dominant influence

- U: total 0.825; first 0.516.
- δ: total 0.358; first 0.096. Strong interactions.
- Topology: 0.104; σ: 0.061; Z: 0.010.
- σ and Z have exactly zero total effect on path existence. Do not sum overlapping total indices.

Presenter notes: Small first-order estimates for known inactive factors reflect Monte Carlo error. Raw estimates and intervals are retained, including negatives.

## 10. Sampling precision improves; size effects remain

- Final relative 95% half-widths: 2.19–2.91% for scenario means.
- At δ=0, σ=1: mean G decreases substantially with n.
- Changing n changes the random-network ensemble.
- A size-independent continuum limit has not been demonstrated.

Presenter notes: Prefix Sobol convergence is supplied as figure 08 and CSV. The leading rank is resolved; tiny indices remain uncertain.

## 11. Heterogeneity is not a pure variance change

- Active-link E[g] = exp(σ²/2): 1, 1.6487, 7.3891 for σ = 0, 1, 2.
- Mean network conductance still falls with σ over these scenarios.
- Weak links on series paths act as bottlenecks.
- Constant-mean control: g* = g exp(−σ²/2), so G* = G exp(−σ²/2) exactly.

Presenter notes: The constant-mean normalization is a supplementary control. It does not replace the assigned ln(g) ~ Normal(0, σ²) model.

## 12. Interpretation and next steps

- Measure connectivity carefully; report connection probability and flow magnitude separately.
- Limits: regular 2D geometry, exploratory priors, steady linear single-phase flow, finite size.
- Next: 3D fracture geometry, aperture-based conductance, correlation, matrix exchange and calibration.
- No field-data validation or geological prediction accuracy is claimed.

Presenter notes: References: Jansen (1999), CPC 117, 35–43, doi:10.1016/S0010-4655(98)00154-4; Kesten (1980), CMP 74, 41–59, doi:10.1007/BF01197577. Full references are in both reports.
