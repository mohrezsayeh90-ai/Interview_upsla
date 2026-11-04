#!/usr/bin/env python3
"""Generate and solve one random network without running the full study."""
import argparse
import json
import numpy as np
from flow_network import Lattice

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n",type=int,default=25)
parser.add_argument("--p",type=float,default=.6)
parser.add_argument("--sigma",type=float,default=1)
parser.add_argument("--seed",type=int,default=42)
args=parser.parse_args()
grid=Lattice(args.n)
r=grid.solve(grid.generate(args.p,args.sigma,np.random.default_rng(args.seed)),details=True)
print(json.dumps({"n":args.n,"p":args.p,"sigma":args.sigma,"seed":args.seed,
                  "geff":r.conductance,"spanning":r.spanning,
                  "outlet_flow":r.outlet_flow,"max_interior_residual":r.max_residual,
                  "energy_dissipation":r.energy,"floating_nodes":r.floating_nodes},indent=2))
