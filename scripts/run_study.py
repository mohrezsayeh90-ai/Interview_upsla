#!/usr/bin/env python3
"""Reproduce all simulations. Defaults are the complete reported experiment.

Run from repository root: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_study.py
Common random numbers couple scenario comparisons; rows remain independent
within each scenario. Sensitivity uses a separate independent random stream.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, binom
from flow_network import Lattice

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 20261104
FACTORS = ["delta", "sigma", "p_latent", "topology", "conductance_field"]


def wilson(k, n):
    z = norm.ppf(.975)
    phat = k/n
    c = (phat+z*z/(2*n))/(1+z*z/n)
    d = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    return c-d, c+d


def describe(frame, group):
    rows = []
    for key, data in frame.groupby(group, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        g = data.geff.to_numpy()
        positive = g[g > 0]
        n = len(g)
        mean, sd = g.mean(), g.std(ddof=1)
        lo, hi = wilson(len(positive), n)
        q95 = np.quantile(g, .95)
        # Conservative, distribution-free binomial order-statistic interval.
        lower_rank = max(1, int(binom.ppf(.025, n, .95)))
        upper_rank = min(n, int(binom.ppf(.975, n, .95))+1)
        sorted_g = np.sort(g)
        rows.append(dict(zip(group, key)) | {
            "samples": n, "mean": mean, "sd": sd, "mcse": sd/np.sqrt(n),
            "mean_ci_low": mean-1.95996398454*sd/np.sqrt(n),
            "mean_ci_high": mean+1.95996398454*sd/np.sqrt(n),
            "p_span": len(positive)/n, "p_span_low": lo, "p_span_high": hi,
            "p_zero": 1-len(positive)/n,
            "q05": np.quantile(g, .05), "median": np.median(g), "q95": q95,
            "q95_low": sorted_g[lower_rank-1], "q95_high": sorted_g[upper_rank-1],
            "positive_mean": positive.mean() if len(positive) else np.nan,
            "positive_median": np.median(positive) if len(positive) else np.nan,
            "positive_q05": np.quantile(positive, .05) if len(positive) else np.nan,
            "positive_q95": np.quantile(positive, .95) if len(positive) else np.nan,
            "cv": sd/mean if mean else np.nan,
            "mean_unit_mean_g": data.geff_unit_mean.mean() if "geff_unit_mean" in data else np.nan,
        })
    return pd.DataFrame(rows)


def scenarios(reps):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1]))
    grid = Lattice(25)
    rows = []
    t = time.perf_counter()
    for i in range(reps):
        a = rng.uniform(-1, 1)
        w, z = rng.random(grid.edge_count), rng.standard_normal(grid.edge_count)
        for delta in [0., .1, .25]:
            p = .5+delta*a
            for sigma in [0., 1., 2.]:
                g = grid.generate(p, sigma, rng, w, z)
                r = grid.solve(g)
                rows.append((i, 25, delta, sigma, p, r.conductance, r.spanning,
                             r.conductance*np.exp(-sigma*sigma/2)))
        if (i+1) % 2000 == 0:
            print(f"scenarios {i+1}/{reps}, elapsed {time.perf_counter()-t:.1f}s", flush=True)
    frame = pd.DataFrame(rows, columns=["replicate", "n", "delta", "sigma", "p", "geff", "spanning", "geff_unit_mean"])
    frame.to_csv(OUT / "scenario_samples.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    describe(frame, ["n", "delta", "sigma"]).to_csv(OUT / "scenario_summary.csv", index=False)
    checkpoints = sorted(set(x for x in [250, 500, 1000, 2000, 5000, reps] if x <= reps))
    convergence = []
    for count in checkpoints:
        part = describe(frame[frame.replicate < count], ["n", "delta", "sigma"])
        part["checkpoint"] = count
        convergence.append(part)
    pd.concat(convergence).to_csv(OUT / "mc_convergence.csv", index=False)

    # Paired differences account for shared random numbers across scenarios.
    comparisons = []
    for varying, fixed, levels in [("delta", "sigma", [0., .25]), ("sigma", "delta", [0., 2.])]:
        for value, sub in frame.groupby(fixed):
            pivot = sub.pivot(index="replicate", columns=varying, values="geff")
            diff = (pivot[levels[1]]-pivot[levels[0]]).to_numpy()
            se = diff.std(ddof=1)/np.sqrt(len(diff))
            comparisons.append({"varying":varying, "fixed":fixed, "fixed_value":value,
                                "from":levels[0], "to":levels[1], "difference":diff.mean(),
                                "ci_low":diff.mean()-1.96*se, "ci_high":diff.mean()+1.96*se})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    return frame


def connectivity_sweep(reps):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    grid = Lattice(25)
    rows = []
    for i in range(reps):
        w, z = rng.random(grid.edge_count), rng.standard_normal(grid.edge_count)
        for p in np.linspace(.3, .7, 9):
            r = grid.solve(grid.generate(p, 1, rng, w, z))
            rows.append((i, 25, p, 1., r.conductance, r.spanning))
    data = pd.DataFrame(rows, columns=["replicate", "n", "p", "sigma", "geff", "spanning"])
    data.to_csv(OUT / "connectivity_samples.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    describe(data, ["n", "p", "sigma"]).to_csv(OUT / "connectivity_summary.csv", index=False)
    print("connectivity sweep complete", flush=True)


def finite_size(reps, scenarios_frame):
    rows = []
    for n in [15, 41]:
        grid = Lattice(n)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, n]))
        for i in range(reps):
            a = rng.uniform(-1, 1)
            w, z = rng.random(grid.edge_count), rng.standard_normal(grid.edge_count)
            for delta in [0., .25]:
                r = grid.solve(grid.generate(.5+delta*a, 1, rng, w, z))
                rows.append((i, n, delta, 1., r.conductance, r.spanning))
        print(f"finite size n={n} complete", flush=True)
    data = pd.DataFrame(rows, columns=["replicate", "n", "delta", "sigma", "geff", "spanning"])
    baseline = scenarios_frame[(scenarios_frame.sigma == 1) & scenarios_frame.delta.isin([0., .25])]
    data = pd.concat([data, baseline[data.columns]], ignore_index=True)
    data.to_csv(OUT / "finite_size_samples.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    describe(data, ["n", "delta", "sigma"]).to_csv(OUT / "finite_size_summary.csv", index=False)


def sensitivity_inputs(rng, grid):
    return [rng.uniform(0, .25), rng.uniform(0, 2), rng.uniform(-1, 1),
            rng.random(grid.edge_count), rng.standard_normal(grid.edge_count)]


def evaluate_factors(grid, x):
    delta, sigma, a, w, z = x
    return grid.solve(grid.generate(.5+delta*a, sigma, None, w, z))


def jansen(y_a, y_b, y_ab):
    variance = np.var(np.r_[y_a, y_b], ddof=1)
    first = 1 - np.mean((y_b[:, None]-y_ab)**2, axis=0)/(2*variance)
    total = np.mean((y_a[:, None]-y_ab)**2, axis=0)/(2*variance)
    return first, total


def sensitivity(base_samples, bootstrap):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 4]))
    grid = Lattice(25)
    output = np.empty((base_samples, 7))
    spans = np.empty_like(output)
    t = time.perf_counter()
    for i in range(base_samples):
        a, b = sensitivity_inputs(rng, grid), sensitivity_inputs(rng, grid)
        for k, x in enumerate([a, b]+[a[:j]+[b[j]]+a[j+1:] for j in range(5)]):
            r = evaluate_factors(grid, x)
            output[i, k], spans[i, k] = r.conductance, r.spanning
        if (i+1) % 2048 == 0:
            print(f"sensitivity {i+1}/{base_samples}, elapsed {time.perf_counter()-t:.1f}s", flush=True)
    names = ["A", "B"]+["AB_"+x for x in FACTORS]
    pd.DataFrame(output, columns=names).to_csv(OUT / "sobol_samples.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    pd.DataFrame(spans, columns=names).to_csv(OUT / "sobol_spanning_samples.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    summary = []
    convergence = []
    brng = np.random.default_rng(np.random.SeedSequence([SEED, 5]))
    for label, values in [("geff", output), ("spanning", spans)]:
        first, total = jansen(values[:, 0], values[:, 1], values[:, 2:])
        bs = np.empty((bootstrap, 2, 5))
        for k in range(bootstrap):
            ix = brng.integers(0, base_samples, base_samples)
            bs[k] = jansen(values[ix, 0], values[ix, 1], values[ix, 2:])
        low, high = np.quantile(bs, [.025, .975], axis=0)
        for j, factor in enumerate(FACTORS):
            summary.append({"output": label, "factor": factor, "first": first[j],
                            "first_low": low[0,j], "first_high": high[0,j],
                            "total": total[j], "total_low": low[1,j], "total_high": high[1,j]})
        for m in sorted(set(x for x in [1024, 2048, 4096, 8192, base_samples] if x <= base_samples)):
            s, st = jansen(values[:m, 0], values[:m, 1], values[:m, 2:])
            for j, factor in enumerate(FACTORS):
                convergence.append({"output":label,"samples":m,"factor":factor,"first":s[j],"total":st[j]})
    pd.DataFrame(summary).to_csv(OUT / "sobol_summary.csv", index=False)
    pd.DataFrame(convergence).to_csv(OUT / "sobol_convergence.csv", index=False)


def diagnostics(count):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 6]))
    grid = Lattice(25)
    rows = []
    for i in range(count):
        p, sigma = rng.uniform(.25, .75), rng.uniform(0, 2)
        r = grid.solve(grid.generate(p, sigma, rng), details=True)
        finite_h = r.heads[np.isfinite(r.heads)]
        rows.append({"replicate":i,"p":p,"sigma":sigma,"geff":r.conductance,
                     "spanning":r.spanning,"max_residual":r.max_residual,
                     "relative_balance_error":r.relative_balance_error,
                     "energy_error":abs(r.energy-r.conductance),
                     "min_head":finite_h.min(),"max_head":finite_h.max(),
                     "floating_nodes":r.floating_nodes})
    pd.DataFrame(rows).to_csv(OUT / "solver_diagnostics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-reps", type=int, default=10000)
    parser.add_argument("--sweep-reps", type=int, default=2500)
    parser.add_argument("--size-reps", type=int, default=5000)
    parser.add_argument("--sobol-base", type=int, default=16384)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--diagnostics", type=int, default=1000)
    args = parser.parse_args()
    if min(vars(args).values()) < 2:
        parser.error("sample counts must be >= 2")
    OUT.mkdir(exist_ok=True)
    t = time.perf_counter()
    data = scenarios(args.scenario_reps)
    connectivity_sweep(args.sweep_reps)
    finite_size(args.size_reps, data)
    sensitivity(args.sobol_base, args.bootstrap)
    diagnostics(args.diagnostics)
    manifest = {
        "seed":SEED,"streams":{"scenarios":1,"sweep":2,"size":3,"sobol":4,"bootstrap":5,"diagnostics":6},
        "arguments":vars(args),"n_baseline":25,"delta_range":[0,.25],"sigma_range":[0,2],
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "pandas":pd.__version__,"elapsed_seconds":time.perf_counter()-t,
        "solver_calls":9*args.scenario_reps+9*args.sweep_reps+4*args.size_reps+7*args.sobol_base+args.diagnostics,
        "model":"ln(g) ~ Normal(0, sigma^2); p=.5+delta*a, a~Uniform(-1,1)",
        "sobol":"5 independent groups; Jansen first/total; paired-row percentile bootstrap; raw estimates unclipped",
        "mean_ci":"normal Monte Carlo 95%, not a model prediction interval",
        "spanning_ci":"Wilson binomial 95%",
        "quantile_ci":"binomial order-statistic 95%, conservative in the presence of ties",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
