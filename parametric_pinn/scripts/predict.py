#!/usr/bin/env python3
"""Use trained weights for one physical realization, or ensemble statistics.

No reference solve is called unless --compare-reference is explicitly provided.
The random seed identifies the realization; delta,p,sigma alone do not.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from parametric_pinn.model import Predictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta',type=float,default=.25)
    parser.add_argument('--sigma',type=float,default=1.)
    parser.add_argument('--p',type=float,default=None)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--head-left',type=float,default=1.)
    parser.add_argument('--head-right',type=float,default=0.)
    parser.add_argument('--mode',choices=['fast','accurate'],default='accurate')
    parser.add_argument('--draws',type=int,default=1)
    parser.add_argument('--conductances',type=Path,help='.npy file with one 1,200-edge field')
    parser.add_argument('--save-heads',type=Path)
    parser.add_argument('--compare-reference',action='store_true')
    args=parser.parse_args()
    if not 0<=args.delta<=.5 or args.sigma<0 or args.scale<=0 or args.draws<1:
        parser.error('require delta in [0,.5], sigma>=0, scale>0 and draws>=1')
    manifest=json.loads((ROOT/'parametric_pinn/results/training_manifest.json').read_text())
    predictor=Predictor(ROOT/'parametric_pinn/models/parametric_pinn.pt',depth=manifest[args.mode+'_depth'])
    rng=np.random.default_rng(args.seed)
    grid=predictor.preprocessor.grid
    t=time.perf_counter()
    if args.draws==1:
        p=.5+args.delta*rng.uniform(-1,1) if args.p is None else args.p
        g=np.load(args.conductances) if args.conductances else args.scale*grid.generate(p,args.sigma,rng)
        result=predictor.predict(g,args.delta,p,args.sigma,args.head_left,args.head_right)
        heads=result.pop('heads')
        if args.save_heads:
            args.save_heads.parent.mkdir(parents=True,exist_ok=True)
            np.save(args.save_heads,heads)
        result.update(delta=args.delta,p=p,sigma=args.sigma,seed=args.seed,scale=args.scale)
        if args.compare_reference:
            reference=grid.solve(g,details=True)
            result['reference_geff']=reference.conductance
            result['relative_geff_error']=result['geff']/reference.conductance-1 if reference.conductance>0 else 0.
    else:
        if args.conductances:parser.error('--conductances is available for a single realization')
        ps=.5+args.delta*rng.uniform(-1,1,args.draws) if args.p is None else np.full(args.draws,args.p)
        fields=np.stack([args.scale*grid.generate(p,args.sigma,rng) for p in ps])
        g=predictor.predict_batch(fields,args.delta,ps,args.sigma)
        positive=g[g>0]
        result={'draws':args.draws,'delta':args.delta,'sigma':args.sigma,'seed':args.seed,
                'mean_geff':float(g.mean()),'mcse_of_surrogate_mean':float(g.std(ddof=1)/np.sqrt(len(g))),
                'p_zero':float(np.mean(g==0)),'median':float(np.median(g)),
                'q95':float(np.quantile(g,.95)),
                'conditional_positive_mean':float(positive.mean()) if len(positive) else None,
                'depth':predictor.depth,'reference_solver_used':False,
                'note':'Monte Carlo error excludes neural approximation error; see paired comparison report.'}
    result['elapsed_seconds_excluding_model_load']=time.perf_counter()-t
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
