# Parametric graph PINN comparison

## 1. A reusable parametric graph PINN

- Main test: 9,000 unseen realizations; 4,518 carry positive flow.
- K=2048: mean relative conductance error 0.0200%.
- Weights are reused. Each input still receives fixed iterative physics updates.
- Completed study • 04 November 2026



## 2. Physical scalars do not identify a random network

- The same δ,p,σ can produce either a spanning or disconnected network.
- For a particular realization, input all 1,200 edge conductances.
- For only δ and σ, sample fields and predict ensemble statistics.
- Boundary heads and uniform conductance scaling use exact physical factors.



## 3. A discrete physics-informed architecture

- Component analysis handles zero-flow and fixed-head components exactly.
- Resistance Dijkstra distances initialize head.
- Small neural networks learn initial correction and graph-level α,β.
- K residual layers update h; no new sparse solve or weight fitting.

This is a custom unrolled graph PINN / learned iterative operator, not the original continuum coordinate PINN architecture. Fixed head updates are performed at every inference.

## 4. Train once without reference solution labels

- Reference solutions are used for validation and tests only.
- 16 epochs; selected epoch 15; depth curriculum 32→64→128→256.
- Training loop 297.4s; total offline work 334.7s.
- Training, validation and test random streams are disjoint.



## 5. Accuracy improves with trained physical layers

- K=256 trained mean error: 0.4705%.
- Untrained K=256 mean error: 1.1381%.
- K=2048: mean 0.0200%, p95 0.0393%.
- Maximum K=2048 error: 4.739%; accuracy is not uniform.

The validation target of a 99th-percentile error below 0.1% was not reached within the depth cap. The target failure is recorded; no test-based retuning was done.

## 6. Energy accuracy and field accuracy differ

- G is estimated from Dirichlet energy, an admissible-head upper bound.
- Approximate inlet flux can differ from the energy estimate.
- Mean free-node head RMSE: 0.001460.
- 95th-percentile inlet/outlet imbalance: 3.02%.

Local weakly connected branches can have significant head errors while contributing little energy. Raw flux and conservation diagnostics are retained.

## 7. Paired uncertainty comparison

- Nine scenarios × 1,000 new realizations each.
- Compare mean, standard deviation, median, upper quantile and conditional flow.
- P(G=0) agrees exactly through structural connectivity preprocessing.
- Old 10,000-sample study and new reference means differ by Monte Carlo variation.



## 8. Sensitivity rankings are reproduced

- Five groups: δ, σ, global-p latent U, topology W and conductance Z.
- 1,000 paired bootstrap resamples preserve input coupling.
- Compare model-minus-reference indices on the same fields.
- Total indices overlap; their sums are not independent variance shares.



## 9. Reusable weights do not guarantee a speedup

- Sparse G-only solver: 0.576 ms per realization.
- Fast PINN, batch 64: 2.508 ms per realization.
- Accurate PINN, batch 64: 15.097 ms per realization.
- For this 25×25 problem, direct solving is faster; no positive break-even was observed.

Two CPU threads, float64 inference, three repetitions on the same 64 new fields. Model loading and field generation are excluded. Offline training is additional.

## 10. Generalization has boundaries

- At σ=3, mean relative G error rises to 2.08%.
- The σ=3 maximum error is 20.26%.
- Other tests: p=1 and n=15,41; same learned weights, no refitting.
- Strong contrast and weak anchoring can require much larger depth.



## 11. Use the trained model directly

- Select fast K=256 or accurate K=2048.
- Provide a .npy conductance field, or distribution parameters plus a seed.
- Use multiple draws to obtain stochastic summaries.
- Keep inlet flux diagnostics separate from the energy-derived G estimate.



## 12. Conclusions and separate deliverables

- The graph PINN reuses trained parameters across unseen random fields.
- High in-domain conductance accuracy does not imply exact local conservation.
- The direct solver retains the advantage in precision and measured CPU cost here.
- All new outputs are isolated under parametric_pinn/.

References: Raissi et al. (2019), doi:10.1016/j.jcp.2018.10.045; E and Yu (2018), doi:10.1007/s40304-018-0127-z; Jansen (1999), doi:10.1016/S0010-4655(98)00154-4. See reports for full context.
