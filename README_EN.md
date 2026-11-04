# Uncertainty and sensitivity of flow in a 2D random network

Completed implementation of all four tasks in the supplied mini-project: numerical modeling and verification, uncertainty quantification, global sensitivity, and interpretation with limitations and extensions.

- [Persian report (7 pages)](reports/report_fa.pdf)
- [English report (7 pages)](reports/report_en.pdf)
- [Presentation (12 slides)](reports/presentation_en.pdf), [editable PowerPoint](reports/presentation_en.pptx), [speaker notes](reports/presentation_notes.md)
- [Executed analysis notebook](notebooks/analysis.ipynb)
- [Raw data and schema](results/README.md)

The study ran **248,188 model evaluations**. At n=25 and σ=1, increasing connectivity uncertainty δ from 0 to 0.25 raises mean effective conductance from 0.018253 to 0.099231 (5.44×), while spanning probabilities stay close to one half. At δ=0.25, σ=1, 48.62% of networks have exactly zero flow. The reference mean has a Monte Carlo 95% interval [0.096603, 0.101859]. The full output distribution combines an atom at zero with a broad positive component.

Grouped Jansen/Sobol analysis includes independent δ, σ, the latent global-connectivity draw U, topology uniforms W, and log-conductance normals Z. This preserves the dependence p=0.5+δU. The total index of U is 0.825; that of δ is 0.358. Total indices include overlapping interactions and must not be summed as independent shares. Conductance parameters have exactly zero effect on whether a path exists, while affecting flow magnitude. Rankings depend on the specified priors and latent parameterization.

The sparse solver removes floating components rather than adding artificial conductivity. **24 tests pass**, including analytical grids, series/parallel paths, conservation, energy, head bounds, scaling, monotonicity and the analytic Ishigami sensitivity benchmark. Independent diagnostics on 1,000 networks have a maximum conservation residual of 2.75e-13.

## Run

For the model and tests only:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/demo.py --n 25 --p 0.6 --sigma 1 --seed 42
```

For complete reproduction (Python 3.12 used):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
make all
```

WeasyPrint report rendering also requires Pango and an Arabic font. On Debian/Ubuntu, the relevant packages are `libpango-1.0-0`, `libpangoft2-1.0-0`, and `fonts-noto-core`. The report uses Noto Naskh Arabic. No credentials or external data downloads are needed for any experiment.

Without Make, run `python -m pytest -q`, then the scripts in this order: `run_study.py`, `make_figures.py`, `make_reports.py`, `make_presentation.py`, `make_notebook.py`, `audit_results.py`. Set `OPENBLAS_NUM_THREADS=1` for the reported execution settings. The numerical study took 188.1 seconds on this host, excluding rendering; timings vary. Seed, random streams, library versions, and sample counts are in `results/manifest.json`. Rendering metadata and elapsed timing prevent a promise of byte-identical artifacts on rerun.

To rebuild figures and deliverables from committed simulation data, run `make figures reports slides notebook audit`.

## Interpretation limits

This is an exploratory dimensionless regular 2D lattice with steady linear single-phase flow. No geological field calibration is claimed. Finite-size effects remain significant near p=0.5. Model verification is distinct from geological validation. Monte Carlo confidence intervals are distinct from network prediction intervals. The assigned lognormal family has median 1 and mean exp(σ²/2); an additional exact rescaling tests a constant-mean alternative without replacing the specified model.

References and full methods are in both reports. See the [Persian README](README.md) for file mapping and detailed instructions.
