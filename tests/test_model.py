"""Independent analytical checks and physical invariants of the numerical model."""
import numpy as np
import pytest
from flow_network import Lattice


@pytest.mark.parametrize("n", [2, 3, 9, 25])
def test_uniform_grid_exact(n):
    grid = Lattice(n)
    for g in [0.1, 1.0, 7.0]:
        r = grid.solve(np.full(grid.edge_count, g), details=True)
        assert r.conductance == pytest.approx(g*n/(n-1), rel=1e-11)
        expected = np.tile(np.linspace(1, 0, n), n)
        np.testing.assert_allclose(r.heads, expected, atol=2e-12)


def test_series_path_and_parallel_paths():
    grid = Lattice(5)
    g = np.zeros(grid.edge_count)
    g[:4] = [1, 2, 3, 4]
    assert grid.solve(g).conductance == pytest.approx(1/np.sum(1/g[:4]))
    g[4:8] = [2, 4, 6, 8]
    assert grid.solve(g).conductance == pytest.approx(3/np.sum(1/g[:4]))


def test_disconnected_and_floating_components():
    grid = Lattice(7)
    g = np.zeros(grid.edge_count)
    g[2*6+2] = 4  # interior island disconnected from both boundaries
    r = grid.solve(g, details=True)
    assert r.conductance == 0 and not r.spanning
    assert r.floating_nodes > 0
    assert np.all(r.flows == 0)


def test_boundary_components_are_constant():
    grid = Lattice(4)
    g = np.zeros(grid.edge_count)
    g[0] = 2   # left-connected dangling node
    g[5] = 3   # right-connected dangling node
    r = grid.solve(g, details=True)
    assert r.heads[1] == 1 and r.heads[6] == 0
    assert r.conductance == 0


@pytest.mark.parametrize("seed", range(12))
def test_conservation_energy_maximum_principle_scaling_and_monotonicity(seed):
    grid = Lattice(15)
    rng = np.random.default_rng(seed)
    g = grid.generate(.65, 2, rng)
    r = grid.solve(g, details=True)
    assert r.max_residual < 1e-10
    assert r.outlet_flow == pytest.approx(r.conductance, abs=1e-10)
    assert r.energy == pytest.approx(r.conductance, abs=1e-10)
    h = r.heads[np.isfinite(r.heads)]
    assert h.min() >= -1e-10 and h.max() <= 1+1e-10
    assert grid.solve(3*g).conductance == pytest.approx(3*r.conductance, abs=1e-10)
    increased = g.copy()
    increased[rng.choice(len(g), 30, replace=False)] += 1
    assert grid.solve(increased).conductance >= r.conductance-1e-10


def test_lognormal_convention_and_reproducibility():
    grid = Lattice(3)
    w = np.linspace(.01, .99, grid.edge_count)
    z = np.linspace(-2, 2, grid.edge_count)
    g = grid.generate(.5, 2, np.random.default_rng(1), w, z)
    np.testing.assert_allclose(g[w < .5], np.exp(2*z[w < .5]))
    assert np.all(g[w >= .5] == 0)
    np.testing.assert_array_equal(grid.generate(.6, 1, np.random.default_rng(42)),
                                  grid.generate(.6, 1, np.random.default_rng(42)))


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        Lattice(1)
    grid = Lattice(3)
    with pytest.raises(ValueError):
        grid.generate(.5, -1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        grid.solve(np.full(grid.edge_count, -1))


def test_adding_vertical_cut_removes_flow():
    grid = Lattice(11)
    g = np.ones(grid.edge_count)
    g[(grid.u % grid.n == 5) & (grid.v == grid.u+1)] = 0
    assert grid.solve(g).conductance == 0
