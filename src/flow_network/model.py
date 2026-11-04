"""Sparse Kirchhoff solver with explicit treatment of disconnected components.

Nodes are indexed row-major, (row, column). Left/right heads are 1/0,
respectively. No exterior top/bottom edges are created (natural no-flow).
Conductances and geometric quantities are dimensionless.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


@dataclass
class FlowResult:
    conductance: float
    spanning: bool
    heads: np.ndarray | None = None
    flows: np.ndarray | None = None
    outlet_flow: float = 0.0
    energy: float = 0.0
    max_residual: float = 0.0
    relative_balance_error: float = 0.0
    floating_nodes: int = 0


class Lattice:
    def __init__(self, n: int = 25):
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer >= 2")
        self.n = int(n)
        nodes = np.arange(n*n).reshape(n, n)
        self.u = np.r_[nodes[:, :-1].ravel(), nodes[:-1, :].ravel()]
        self.v = np.r_[nodes[:, 1:].ravel(), nodes[1:, :].ravel()]
        self.left = nodes[:, 0]
        self.right = nodes[:, -1]
        self.boundary = np.zeros(n*n, dtype=bool)
        self.boundary[np.r_[self.left, self.right]] = True
        self.edge_count = len(self.u)
        self.xy = np.column_stack((nodes.ravel() % n, n-1-nodes.ravel() // n))

    def generate(self, p: float, sigma: float, rng: np.random.Generator,
                 uniforms: np.ndarray | None = None,
                 normals: np.ndarray | None = None) -> np.ndarray:
        if not (0 <= p <= 1) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("p must be in [0,1], sigma must be finite and >= 0")
        w = rng.random(self.edge_count) if uniforms is None else np.asarray(uniforms)
        z = rng.standard_normal(self.edge_count) if normals is None else np.asarray(normals)
        if w.shape != (self.edge_count,) or z.shape != (self.edge_count,):
            raise ValueError("one latent uniform and normal per edge is required")
        g = np.zeros(self.edge_count)
        mask = w < p
        g[mask] = np.exp(sigma*z[mask])
        return g

    def solve(self, conductances: np.ndarray, details: bool = False) -> FlowResult:
        g_all = np.asarray(conductances, dtype=float)
        if g_all.shape != (self.edge_count,):
            raise ValueError("incorrect number of edge conductances")
        if np.any(g_all < 0) or not np.all(np.isfinite(g_all)):
            raise ValueError("conductances must be finite and non-negative")
        on = g_all > 0
        u, v, g = self.u[on], self.v[on], g_all[on]
        adjacency = coo_matrix((np.ones(2*len(u)),
                               (np.r_[u, v], np.r_[v, u])),
                              shape=(self.n**2, self.n**2)).tocsr()
        nc, labels = connected_components(adjacency, directed=False)
        touches_left, touches_right = np.zeros(nc, bool), np.zeros(nc, bool)
        touches_left[labels[self.left]] = True
        touches_right[labels[self.right]] = True
        spanning_components = touches_left & touches_right
        spans = bool(np.any(spanning_components))
        if not spans and not details:
            return FlowResult(0.0, False)

        h = np.full(self.n**2, np.nan)
        h[touches_left[labels]] = 1.0
        h[touches_right[labels]] = 0.0
        h[self.left], h[self.right] = 1.0, 0.0
        active = spanning_components[labels]
        free = np.flatnonzero(active & ~self.boundary)
        active_edges = active[u]
        au, av, ag = u[active_edges], v[active_edges], g[active_edges]
        if len(free):
            lap = coo_matrix((np.r_[ag, ag, -ag, -ag],
                              (np.r_[au, av, au, av], np.r_[au, av, av, au])),
                             shape=(self.n**2, self.n**2)).tocsr()
            fixed_h = np.zeros(self.n**2)
            fixed_h[self.left] = 1.0
            rhs = -(lap @ fixed_h)[free]
            h[free] = spsolve(lap[free][:, free], rhs)

        # Only spanning clusters carry flux. Avoid cancellation from nonspanning clusters.
        q_active = ag * (h[au]-h[av])
        left_nodes = np.zeros(self.n**2, dtype=bool)
        left_nodes[self.left] = True
        inflow = float(np.sum(q_active[left_nodes[au]]) - np.sum(q_active[left_nodes[av]]))
        if not details:
            if inflow < -1e-12 or not np.isfinite(inflow):
                raise ArithmeticError("invalid flux from sparse solve")
            return FlowResult(max(inflow, 0.0), spans)

        q = np.zeros(self.edge_count)
        # Floating components have arbitrary constant head and exactly zero flux.
        known = on & np.isfinite(h[self.u]) & np.isfinite(h[self.v])
        q[known] = g_all[known] * (h[self.u[known]]-h[self.v[known]])
        balance = np.bincount(self.u, weights=q, minlength=self.n**2)
        balance -= np.bincount(self.v, weights=q, minlength=self.n**2)
        outlet = float(-balance[self.right].sum())
        energy = float(np.sum(q[known] * (h[self.u[known]]-h[self.v[known]])))
        residual = float(np.max(np.abs(balance[~self.boundary]))) if self.n > 2 else 0.0
        return FlowResult(max(inflow, 0.0), spans, h, q, outlet, energy, residual,
                          abs(inflow-outlet)/max(abs(inflow), 1e-30),
                          int(np.isnan(h).sum()))
