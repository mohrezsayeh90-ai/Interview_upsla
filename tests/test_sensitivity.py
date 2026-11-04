"""Check the estimator against a known analytic sensitivity benchmark."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("run_study", Path(__file__).parents[1]/"scripts"/"run_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_jansen_ishigami_benchmark():
    rng = np.random.default_rng(9753)
    a, b = rng.uniform(-np.pi, np.pi, (2, 120000, 3))
    def ishigami(x):
        return np.sin(x[:,0])+7*np.sin(x[:,1])**2+.1*x[:,2]**4*np.sin(x[:,0])
    ab = []
    for j in range(3):
        c = a.copy()
        c[:, j] = b[:, j]
        ab.append(ishigami(c))
    first, total = study.jansen(ishigami(a), ishigami(b), np.column_stack(ab))
    np.testing.assert_allclose(first, [.313905, .442411, 0], atol=.018)
    np.testing.assert_allclose(total, [.557589, .442411, .243684], atol=.018)


def test_spanning_independent_of_conductance_variability():
    from flow_network import Lattice
    grid = Lattice(15)
    rng = np.random.default_rng(2)
    for _ in range(30):
        x = study.sensitivity_inputs(rng, grid)
        r = study.evaluate_factors(grid, x)
        x[1] = 2-x[1]
        x[4] = rng.standard_normal(grid.edge_count)
        assert study.evaluate_factors(grid, x).spanning == r.spanning
