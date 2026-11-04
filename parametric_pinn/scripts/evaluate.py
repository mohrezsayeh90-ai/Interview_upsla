#!/usr/bin/env python3
"""Locked-test paired comparison with the verified sparse Kirchhoff solver."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from parametric_pinn.model import Preprocessor,ParametricPINN,stack_prepared
from scripts.run_study import jansen,FACTORS

OUT=ROOT/'parametric_pinn'/'results'
SEED=20261105


class Engine:
    def __init__(self,n,manifest):
        self.pre=Preprocessor(n)
        state=torch.load(ROOT/'parametric_pinn/models/parametric_pinn.pt',weights_only=True)
        config=state['config']
        self.model=ParametricPINN(n,config['width'],config['depth']).double().eval()
        weights={k:v for k,v in state['state_dict'].items() if k not in ['neighbor','edge_u','edge_v']}
        self.model.load_state_dict(weights,strict=False)
        self.modes={'geodesic':(0,False),'untrained_control':(256,False),
                    'pinn_fast':(manifest['fast_depth'],True),
                    'pinn_accurate':(manifest['accurate_depth'],True)}

    @torch.inference_mode()
    def predict(self,rows,mode,batch_size=64):
        # Routing zero-flow components is an exact graph operation, not a learned classifier.
        output=np.stack([r.h0 for r in rows])
        active=[i for i,r in enumerate(rows) if r.spanning]
        depth,learned=self.modes[mode]
        for start in range(0,len(active),batch_size):
            indices=active[start:start+batch_size]
            batch=stack_prepared([rows[i] for i in indices],dtype=torch.float64)
            h=self.model(batch,depth=depth,learned=learned)
            output[indices]=h.numpy()
        return output


def summarize_predictions(data,group):
    records=[]
    for key,sub in data.groupby(group,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        positive=sub[sub.g_ref>0]
        err=sub.g_pred-sub.g_ref
        rel=positive.relative_g_error.to_numpy()
        n=len(sub)
        bias=err.mean()
        se=err.std(ddof=1)/np.sqrt(n) if n>1 else 0
        record=dict(zip(group,key))|{
            'cases':n,'positive_cases':len(positive),'mean_reference':sub.g_ref.mean(),
            'mean_predicted':sub.g_pred.mean(),'paired_bias':bias,
            'paired_bias_ci_low':bias-1.96*se,'paired_bias_ci_high':bias+1.96*se,
            'mean_relative_bias':bias/max(sub.g_ref.mean(),1e-30),
            'mae':err.abs().mean(),'rmse':np.sqrt(np.mean(err**2)),
            'positive_mean_abs_relative_error':np.mean(np.abs(rel)) if len(rel) else np.nan,
            'positive_median_abs_relative_error':np.median(np.abs(rel)) if len(rel) else np.nan,
            'positive_p95_abs_relative_error':np.quantile(np.abs(rel),.95) if len(rel) else np.nan,
            'positive_p99_abs_relative_error':np.quantile(np.abs(rel),.99) if len(rel) else np.nan,
            'positive_max_abs_relative_error':np.max(np.abs(rel)) if len(rel) else np.nan,
            'positive_fraction_within_1pct':np.mean(np.abs(rel)<=.01) if len(rel) else np.nan,
            'positive_fraction_within_0_1pct':np.mean(np.abs(rel)<=.001) if len(rel) else np.nan,
            'p_zero_reference':np.mean(sub.g_ref==0),'p_zero_predicted':np.mean(sub.g_pred==0),
            'median_reference':np.median(sub.g_ref),'median_predicted':np.median(sub.g_pred),
            'q95_reference':np.quantile(sub.g_ref,.95),'q95_predicted':np.quantile(sub.g_pred,.95),
            'sd_reference':sub.g_ref.std(ddof=1),'sd_predicted':sub.g_pred.std(ddof=1),
            'mean_free_node_rmse':positive.head_rmse.mean(),
            'p95_free_node_max_error':np.quantile(positive.head_max_error,.95) if len(positive) else np.nan,
            'p95_interior_residual_abs':np.quantile(positive.max_residual_abs,.95) if len(positive) else np.nan,
            'p95_interior_normalized_residual':np.quantile(positive.normalized_residual,.95) if len(positive) else np.nan,
            'p95_inlet_relative_error':np.quantile(np.abs(positive.inlet_relative_error),.95) if len(positive) else np.nan,
            'p95_relative_in_out_imbalance':np.quantile(positive.relative_imbalance,.95) if len(positive) else np.nan,
            'mean_relative_edge_flow_l2_error':positive.relative_edge_flow_l2_error.mean(),
        }
        records.append(record)
    return pd.DataFrame(records)


def metrics(engine,rows,refs,heads,metadata,mode):
    grid=engine.pre.grid
    records=[]
    for row,ref,h,meta in zip(rows,refs,heads,metadata):
        q=row.g*(h[grid.u]-h[grid.v])
        energy=float(np.sum(q*(h[grid.u]-h[grid.v])))
        balance=np.bincount(grid.u,weights=q,minlength=grid.n**2)-np.bincount(grid.v,weights=q,minlength=grid.n**2)
        inlet,outlet=float(balance[grid.left].sum()),float(-balance[grid.right].sum())
        free=row.free
        herror=h[free]-ref.heads[free]
        normalized=h-(row.transition*h[engine.pre.neighbor]).sum(-1)
        normq=np.linalg.norm(ref.flows)
        records.append(meta|{'method':mode,'g_ref':ref.conductance,'g_pred':energy,
            'relative_g_error':energy/ref.conductance-1 if ref.conductance>0 else np.nan,
            'head_rmse':float(np.sqrt(np.mean(herror**2))) if np.any(free) else 0.,
            'head_max_error':float(np.max(np.abs(herror))) if np.any(free) else 0.,
            'max_residual_abs':float(np.max(np.abs(balance[~grid.boundary]))),
            'normalized_residual':float(np.max(np.abs(normalized[free]))) if np.any(free) else 0.,
            'inlet_flow':inlet,'outlet_flow':outlet,
            'inlet_relative_error':inlet/ref.conductance-1 if ref.conductance>0 else np.nan,
            'relative_imbalance':abs(inlet-outlet)/max(energy,1e-30),
            'energy_flux_relative_gap':abs(energy-(inlet+outlet)/2)/max(energy,1e-30),
            'relative_edge_flow_l2_error':float(np.linalg.norm(q-ref.flows)/normq) if normq>0 else 0.,
            'reference_max_residual':ref.max_residual})
    return records


def run_test(manifest,scenario_reps,continuous_reps,stress_reps):
    cases=[]
    engine=Engine(25,manifest)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,201]))
    for i in range(scenario_reps):
        a=rng.uniform(-1,1)
        w,z=rng.random(1200),rng.standard_normal(1200)
        for delta in [0.,.1,.25]:
            for sigma in [0.,1.,2.]:
                p=.5+delta*a
                g=engine.pre.grid.generate(p,sigma,None,w,z)
                cases.append((g,{'case_id':f'scenario_{i}_{delta}_{sigma}','split':'scenario','n':25,
                                  'replicate':i,'delta':delta,'p':p,'sigma':sigma,'stream':201}))
    rng=np.random.default_rng(np.random.SeedSequence([SEED,202]))
    for i in range(continuous_reps):
        delta,sigma=rng.uniform(0,.25),rng.uniform(0,2)
        p=.5+delta*rng.uniform(-1,1)
        g=engine.pre.grid.generate(p,sigma,rng)
        cases.append((g,{'case_id':f'continuous_{i}','split':'continuous','n':25,
                           'replicate':i,'delta':delta,'p':p,'sigma':sigma,'stream':202}))
    rng=np.random.default_rng(np.random.SeedSequence([SEED,203]))
    for i in range(stress_reps):
        p=rng.uniform(.4,.7);sigma=3.
        cases.append((engine.pre.grid.generate(p,sigma,rng),{'case_id':f'stress_sigma_{i}',
                    'split':'ood_sigma3','n':25,'replicate':i,'delta':.25,'p':p,'sigma':sigma,'stream':203}))
    rng=np.random.default_rng(np.random.SeedSequence([SEED,204]))
    for i in range(stress_reps):
        sigma=rng.uniform(0,2);p=1.
        cases.append((engine.pre.grid.generate(p,sigma,rng),{'case_id':f'stress_full_{i}',
                    'split':'ood_fully_connected','n':25,'replicate':i,'delta':.25,'p':p,'sigma':sigma,'stream':204}))
    records=[];fingerprint=hashlib.sha256();saved=[]
    t=time.perf_counter()
    for start in range(0,len(cases),128):
        chunk=cases[start:start+128]
        rows=[engine.pre.prepare(g,m['delta'],m['p'],m['sigma']) for g,m in chunk]
        refs=[engine.pre.grid.solve(g,details=True) for g,m in chunk]
        metadata=[m for _,m in chunk]
        for g,m in chunk:fingerprint.update(g.tobytes())
        for mode in engine.modes:
            h=engine.predict(rows,mode)
            records.extend(metrics(engine,rows,refs,h,metadata,mode))
            if mode=='pinn_accurate' and len(saved)<24:
                saved.extend([(r.g,ref.heads,hh,m) for r,ref,hh,m in zip(rows,refs,h,metadata)][:24-len(saved)])
        if (start//128)%8==0:
            print(f'paired test {min(start+128,len(cases))}/{len(cases)}, {time.perf_counter()-t:.1f}s',flush=True)
    # Cross-size reuse is an explicit OOD experiment, not a trained-domain claim.
    for n in [15,41]:
        eng=Engine(n,manifest)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,205,n]))
        for start in range(0,stress_reps,64):
            rows=[];refs=[];metadata=[]
            for i in range(start,min(start+64,stress_reps)):
                delta,sigma=.25,rng.uniform(0,2)
                p=.5+delta*rng.uniform(-1,1)
                g=eng.pre.grid.generate(p,sigma,rng)
                rows.append(eng.pre.prepare(g,delta,p,sigma));refs.append(eng.pre.grid.solve(g,details=True))
                metadata.append({'case_id':f'ood_n{n}_{i}','split':f'ood_size{n}','n':n,'replicate':i,
                                 'delta':delta,'p':p,'sigma':sigma,'stream':205})
            for mode in ['pinn_fast','pinn_accurate']:
                h=eng.predict(rows,mode)
                records.extend(metrics(eng,rows,refs,h,metadata,mode))
    data=pd.DataFrame(records)
    data.to_csv(OUT/'paired_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summarize_predictions(data,['split','method']).to_csv(OUT/'test_summary.csv',index=False)
    scenarios=data[data.split=='scenario']
    summarize_predictions(scenarios,['delta','sigma','method']).to_csv(OUT/'scenario_comparison.csv',index=False)
    identical=scenarios[(scenarios.method=='pinn_accurate')&(scenarios.delta==0)&(scenarios.sigma==1)]
    mean_g=identical.g_ref.mean()
    pd.DataFrame([{'delta':0,'p':.5,'sigma':1,'cases':len(identical),'mean_g':mean_g,
        'sd_g':identical.g_ref.std(ddof=1),
        'constant_best_empirical_rmse':np.sqrt(np.mean((identical.g_ref-mean_g)**2)),
        'p_zero':np.mean(identical.g_ref==0)}]).to_csv(OUT/'physical_parameters_only_baseline.csv',index=False)
    np.savez_compressed(OUT/'example_fields.npz',g=np.stack([r[0] for r in saved]),
                        reference_heads=np.stack([r[1] for r in saved]),pinn_heads=np.stack([r[2] for r in saved]))
    (OUT/'example_fields_metadata.json').write_text(json.dumps([r[3] for r in saved],indent=2)+'\n')
    return {'main_cases':len(cases),'size_cases':2*stress_reps,'paired_evaluation_seconds':time.perf_counter()-t,
            'input_sha256_main_float64':fingerprint.hexdigest()}


def sobol(manifest,n_base,bootstrap=1000):
    engine=Engine(25,manifest)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,301]))
    arrays={k:np.empty((n_base,7)) for k in ['reference','pinn_fast','pinn_accurate']}
    t=time.perf_counter()
    def draw():
        return [rng.uniform(0,.25),rng.uniform(0,2),rng.uniform(-1,1),rng.random(1200),rng.standard_normal(1200)]
    for start in range(0,n_base,16):
        rows=[];index=[];geff=[]
        for i in range(start,min(start+16,n_base)):
            a,b=draw(),draw()
            inputs=[a,b]+[a[:j]+[b[j]]+a[j+1:] for j in range(5)]
            for k,(delta,sigma,u,w,z) in enumerate(inputs):
                p=.5+delta*u
                g=engine.pre.grid.generate(p,sigma,None,w,z)
                rows.append(engine.pre.prepare(g,delta,p,sigma))
                geff.append(engine.pre.grid.solve(g).conductance);index.append((i,k))
        for (i,k),value in zip(index,geff):arrays['reference'][i,k]=value
        for mode in ['pinn_fast','pinn_accurate']:
            h=engine.predict(rows,mode)
            for j,(i,k) in enumerate(index):
                arrays[mode][i,k]=np.sum(rows[j].g*(h[j,engine.pre.grid.u]-h[j,engine.pre.grid.v])**2)
        if start%256==0:print(f'paired Sobol {min(start+16,n_base)}/{n_base}, {time.perf_counter()-t:.1f}s',flush=True)
    columns=['A','B']+['AB_'+f for f in FACTORS]
    frame=pd.concat({k:pd.DataFrame(v,columns=columns) for k,v in arrays.items()},axis=1)
    frame.columns=['__'.join(col) for col in frame.columns]
    frame.to_csv(OUT/'sobol_paired_outputs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    point={mode:jansen(a[:,0],a[:,1],a[:,2:]) for mode,a in arrays.items()}
    rng=np.random.default_rng(np.random.SeedSequence([SEED,302]))
    bs={mode:np.empty((bootstrap,2,5)) for mode in arrays}
    for b in range(bootstrap):
        ix=rng.integers(0,n_base,n_base)
        for mode,a in arrays.items():bs[mode][b]=jansen(a[ix,0],a[ix,1],a[ix,2:])
    records=[]
    for mode in arrays:
        lo,hi=np.quantile(bs[mode],[.025,.975],axis=0)
        diff_bs=bs[mode]-bs['reference']
        dlo,dhi=np.quantile(diff_bs,[.025,.975],axis=0)
        for order,key in enumerate(['first','total']):
            for j,factor in enumerate(FACTORS):
                records.append({'method':mode,'order':key,'factor':factor,'index':point[mode][order][j],
                    'ci_low':lo[order,j],'ci_high':hi[order,j],
                    'difference_vs_reference':point[mode][order][j]-point['reference'][order][j],
                    'difference_ci_low':dlo[order,j],'difference_ci_high':dhi[order,j]})
    pd.DataFrame(records).to_csv(OUT/'sobol_comparison.csv',index=False)
    return {'base_pairs':n_base,'reference_calls':7*n_base,'bootstrap':bootstrap,'seconds':time.perf_counter()-t}


def benchmark(manifest,cases=64,repetitions=3):
    engine=Engine(25,manifest)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,401]))
    fields=[]
    for _ in range(cases):
        delta,sigma=rng.uniform(0,.25),rng.uniform(0,2)
        p=.5+delta*rng.uniform(-1,1)
        fields.append((engine.pre.grid.generate(p,sigma,rng),delta,p,sigma))
    rows=[engine.pre.prepare(*f) for f in fields]
    for mode in ['pinn_fast','pinn_accurate']:engine.predict(rows[:4],mode)
    records=[]
    for rep in range(repetitions):
        t=time.perf_counter()
        for g,_,_,_ in fields:engine.pre.grid.solve(g,details=False)
        sparse_g=time.perf_counter()-t
        t=time.perf_counter()
        for g,_,_,_ in fields:engine.pre.grid.solve(g,details=True)
        sparse_field=time.perf_counter()-t
        for mode in ['pinn_fast','pinn_accurate']:
            for batch_size in [1,16,64]:
                prep_time,core_time=0.,0.
                for start in range(0,cases,batch_size):
                    t=time.perf_counter()
                    prepared=[engine.pre.prepare(*f) for f in fields[start:start+batch_size]]
                    prep_time+=time.perf_counter()-t
                    t=time.perf_counter()
                    engine.predict(prepared,mode,batch_size=batch_size)
                    core_time+=time.perf_counter()-t
                total=prep_time+core_time
                per=total/cases
                offline=manifest['elapsed_seconds']
                gain=sparse_g/cases-per
                records.append({'repetition':rep,'method':mode,'batch_size':batch_size,'cases':cases,
                    'preprocessing_seconds':prep_time,'neural_core_seconds':core_time,'total_seconds':total,
                    'milliseconds_per_case':1000*per,'sparse_g_seconds':sparse_g,
                    'sparse_fullfield_seconds':sparse_field,'speedup_vs_sparse_g':sparse_g/total,
                    'break_even_predictions':offline/gain if gain>0 else np.nan})
        print(f'benchmark repetition {rep+1}/{repetitions}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'timing_benchmark.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-reps',type=int,default=1000)
    parser.add_argument('--continuous-reps',type=int,default=1000)
    parser.add_argument('--stress-reps',type=int,default=250)
    parser.add_argument('--sobol-base',type=int,default=2048)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest=json.loads((OUT/'training_manifest.json').read_text())
    t=time.perf_counter()
    test_info=run_test(manifest,args.scenario_reps,args.continuous_reps,args.stress_reps)
    sensitivity=sobol(manifest,args.sobol_base)
    benchmark(manifest)
    info={'seed':SEED,'test_streams':{'scenarios':201,'continuous':202,'sigma_stress':203,
          'fully_connected_stress':204,'size_stress':205,'sobol':301,'bootstrap':302,'timing':401},
          'training_stream':101,'validation_stream':102,'arguments':vars(args),
          'python':platform.python_version(),'test':test_info,'sensitivity':sensitivity,
          'elapsed_seconds':time.perf_counter()-t,'energy_estimator':True,
          'primary_test_is_unseen_topology':True,'test_time_sparse_refinement':False,
          'test_time_training':False,'nonneural_preprocessing':'connected components and shortest resistance distances',
          'comparison_scope':'paired independent test fields, not subtraction of two unpaired Monte Carlo studies'}
    (OUT/'evaluation_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
