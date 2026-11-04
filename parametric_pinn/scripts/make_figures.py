#!/usr/bin/env python3
"""Render comparison figures from paired held-out results and saved weights."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from parametric_pinn.model import Predictor

BASE=ROOT/'parametric_pinn'
R=BASE/'results';F=BASE/'reports'/'figures'
COLORS={'pinn_fast':'#e28b22','pinn_accurate':'#176b87','untrained_control':'#a43d52','geodesic':'#869b8f','reference':'#263f52'}
LABELS={'pinn_fast':'Trained PINN, K=256','pinn_accurate':'Trained PINN, K=2048','untrained_control':'Untrained physics network, K=256','geodesic':'Resistance-geodesic initialization','reference':'Sparse reference'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','figure.dpi':120,'savefig.dpi':190,'axes.labelcolor':'#23394b','text.color':'#23394b','grid.alpha':.2})


def save(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_figures():
    F.mkdir(parents=True,exist_ok=True)
    history=pd.read_csv(R/'training_history.csv')
    depths=pd.read_csv(R/'validation_depths.csv')
    data=pd.read_csv(R/'paired_predictions.csv.gz')
    summary=pd.read_csv(R/'test_summary.csv')
    scenarios=pd.read_csv(R/'scenario_comparison.csv')
    train=json.loads((R/'training_manifest.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].plot(history.epoch,history.physics_loss,color=COLORS['pinn_accurate'],marker='.')
    axes[0].set(xlabel='Epoch',ylabel='Physics loss (no training labels)',title='One-time training')
    axes[1].semilogy(history.epoch,100*history.mean_relative_g_error,color=COLORS['pinn_accurate'],marker='o',label='Mean positive-network error')
    axes[1].semilogy(history.epoch,100*history.p95_relative_g_error,color=COLORS['pinn_fast'],marker='.',label='95th-percentile error')
    axes[1].axvline(train['best_epoch'],color='grey',ls=':',label='Validation-selected checkpoint')
    axes[1].set(xlabel='Epoch',ylabel='Relative conductance error (%)',title='Independent validation at K=256');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid()
    save(fig,'01_training')

    test=data[(data.split.isin(['scenario','continuous']))&(data.g_ref>0)]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    accurate=test[test.method=='pinn_accurate']
    axes[0].scatter(accurate.g_ref,accurate.g_pred,s=5,alpha=.3,color=COLORS['pinn_accurate'],rasterized=True)
    lim=[max(accurate.g_ref.min()*.8,1e-8),accurate.g_ref.max()*1.2]
    axes[0].plot(lim,lim,color='grey',ls=':')
    axes[0].set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Reference G_eff',ylabel='PINN energy-derived G_eff',title='Paired unseen realizations (positive only)')
    for mode in ['geodesic','untrained_control','pinn_fast','pinn_accurate']:
        s=test[test.method==mode]
        err=np.sort(np.maximum(np.abs(s.relative_g_error.to_numpy())*100,1e-12))
        axes[1].plot(err,np.arange(1,len(err)+1)/len(err),label=LABELS[mode],color=COLORS[mode])
    axes[1].set(xscale='log',xlabel='Absolute relative G error (%)',ylabel='Empirical CDF',title='Ablation and accuracy comparison',ylim=(0,1.01))
    axes[1].legend(fontsize=7,loc='lower right')
    for ax in axes:ax.grid()
    save(fig,'02_accuracy')

    # Recover a difficult, independent in-domain example without using it for training.
    chosen=data[(data.split=='scenario')&(data.method=='pinn_accurate')&(data.g_ref>0)].sort_values('head_rmse',ascending=False).iloc[0]
    rng=np.random.default_rng(np.random.SeedSequence([20261105,201]))
    for i in range(int(chosen.replicate)+1):
        a=rng.uniform(-1,1);w,z=rng.random(1200),rng.standard_normal(1200)
    p=.5+chosen.delta*a
    predictor=Predictor(BASE/'models/parametric_pinn.pt',depth=train['accurate_depth'])
    grid=predictor.preprocessor.grid
    g=grid.generate(p,float(chosen.sigma),None,w,z)
    reference=grid.solve(g,details=True)
    predicted=predictor.predict(g,float(chosen.delta),p,float(chosen.sigma))
    href=reference.heads.reshape(25,25)
    hp=predicted['heads'].reshape(25,25)
    error=hp-href
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for ax,field,title in zip(axes[:2],[href,hp],['Sparse reference head','Parametric PINN head']):
        im=ax.imshow(field,cmap='viridis',vmin=0,vmax=1,origin='lower')
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes[:2],shrink=.65,label='h')
    bound=max(np.nanmax(np.abs(error)),1e-6)
    im=axes[2].imshow(error,cmap='RdBu_r',vmin=-bound,vmax=bound,origin='lower')
    axes[2].set(title='Signed head error',xticks=[],yticks=[])
    fig.colorbar(im,ax=axes[2],shrink=.65)
    fig.suptitle(f'Largest free-node head RMSE in scenario test: δ={chosen.delta:g}, σ={chosen.sigma:g}, p={p:.3f}',fontsize=10)
    save(fig,'03_head_fields')
    np.savez_compressed(R/'difficult_example.npz',g=g,reference_heads=reference.heads,pinn_heads=predicted['heads'])
    (R/'difficult_example.json').write_text(json.dumps({'case_id':chosen.case_id,'delta':float(chosen.delta),
        'p':p,'sigma':float(chosen.sigma),'free_node_rmse':float(chosen.head_rmse),
        'reference_geff':reference.conductance,'pinn_geff':predicted['geff'],
        'selection':'largest per-realization free-node head RMSE among the independent scenario tests'},indent=2)+'\n')

    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    ss=scenarios[scenarios.method=='pinn_accurate'].sort_values(['delta','sigma'])
    x=np.arange(len(ss))
    axes[0].bar(x-.18,ss.mean_reference,width=.34,color=COLORS['reference'],label='Sparse reference')
    axes[0].bar(x+.18,ss.mean_predicted,width=.34,color=COLORS['pinn_accurate'],label='PINN, K=2048')
    labels=[f'{r.delta:g}\n{r.sigma:g}' for _,r in ss.iterrows()]
    axes[0].set(xticks=x,xticklabels=labels,xlabel='δ (top label), σ (bottom label)',ylabel='Mean effective conductance',title='Paired Monte Carlo statistics')
    axes[0].legend(fontsize=8)
    for mode in ['pinn_fast','pinn_accurate']:
        sub=scenarios[scenarios.method==mode].sort_values(['delta','sigma'])
        mean=sub.mean_reference.to_numpy()
        bias=100*sub.mean_relative_bias.to_numpy()
        low=100*sub.paired_bias_ci_low.to_numpy()/mean;high=100*sub.paired_bias_ci_high.to_numpy()/mean
        axes[1].errorbar(x,bias,yerr=np.maximum(0,np.vstack([bias-low,high-bias])),marker='o',capsize=2,label=LABELS[mode],color=COLORS[mode])
    axes[1].set(xticks=x,xticklabels=labels,xlabel='δ (top), σ (bottom)',ylabel='Bias in ensemble mean (%)',title='Paired bias with 95% sampling CI')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y')
    save(fig,'04_uncertainty_comparison')

    sobol=pd.read_csv(R/'sobol_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    factors=['delta','sigma','p_latent','topology','conductance_field']
    labels=['δ','σ','Global p\nlatent','Topology\nfield','Conductance\nfield']
    for ax,order in zip(axes,['first','total']):
        for off,mode in [(-.22,'reference'),(0,'pinn_fast'),(.22,'pinn_accurate')]:
            sub=sobol[(sobol.method==mode)&(sobol.order==order)].set_index('factor').loc[factors]
            values=sub['index'].to_numpy()
            ax.bar(np.arange(5)+off,values,width=.21,color=COLORS[mode],label=LABELS[mode])
            if mode=='reference':
                ax.errorbar(np.arange(5)+off,values,yerr=np.maximum(0,np.vstack([values-sub.ci_low,sub.ci_high-values])),fmt='none',color='#607681',capsize=2)
        ax.set(xticks=np.arange(5),xticklabels=labels,ylabel='Sobol index',title=f'{order.title()}-order sensitivity');ax.grid(axis='y')
    axes[0].legend(fontsize=7)
    save(fig,'05_sensitivity_comparison')

    timing=pd.read_csv(R/'timing_benchmark.csv')
    tm=timing.groupby(['method','batch_size']).median(numeric_only=True).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    pos=np.arange(len(tm))
    prep=1000*tm.preprocessing_seconds/tm.cases
    core=1000*tm.neural_core_seconds/tm.cases
    axes[0].bar(pos,prep,color='#8db4c1',label='Graph preprocessing')
    axes[0].bar(pos,core,bottom=prep,color='#176b87',label='Neural core + tensor assembly')
    axes[0].axhline(float(np.median(timing.sparse_g_seconds/timing.cases)*1000),color=COLORS['reference'],ls='--',label='Sparse G-only solver')
    axes[0].set(xticks=pos,xticklabels=[f'{"fast" if r.method=="pinn_fast" else "accurate"}\nB={int(r.batch_size)}' for _,r in tm.iterrows()],yscale='log',ylabel='Milliseconds per realization',title='Warm CPU end-to-end inference')
    axes[0].legend(fontsize=7)
    axes[1].loglog(depths.depth[depths.depth>0],100*depths.mean_relative_g_error[depths.depth>0],marker='o',label='Validation mean error',color=COLORS['pinn_accurate'])
    axes[1].loglog(depths.depth[depths.depth>0],100*depths.p99_relative_g_error[depths.depth>0],marker='.',label='Validation 99th-percentile error',color=COLORS['pinn_fast'])
    axes[1].set(xlabel='Fixed graph-network depth K',ylabel='Positive-network relative G error (%)',title='Depth–accuracy tradeoff')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid()
    save(fig,'06_cost_accuracy')

    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    scenarios_order=['scenario','continuous','ood_sigma3','ood_fully_connected','ood_size15','ood_size41']
    sub=summary[summary.method=='pinn_accurate'].set_index('split').loc[scenarios_order]
    axes[0].bar(np.arange(len(sub)),100*sub.positive_mean_abs_relative_error,color=COLORS['pinn_accurate'],label='Mean error')
    axes[0].scatter(np.arange(len(sub)),100*sub.positive_p95_abs_relative_error,color=COLORS['pinn_fast'],marker='D',label='95th percentile')
    axes[0].set(xticks=np.arange(len(sub)),xticklabels=['Main\nscenarios','Continuous\nIID','σ=3\nOOD','p=1\nOOD','n=15\nOOD','n=41\nOOD'],yscale='log',ylabel='Relative G error (%)',title='Generalization boundaries');axes[0].legend(fontsize=8)
    for mode in ['pinn_fast','pinn_accurate']:
        sub=test[test.method==mode]
        axes[1].scatter(np.maximum(np.abs(sub.relative_g_error)*100,1e-12),np.maximum(np.abs(sub.inlet_relative_error)*100,1e-12),s=4,alpha=.2,color=COLORS[mode],label=LABELS[mode],rasterized=True)
    axes[1].set(xscale='log',yscale='log',xlabel='Energy-derived conductance error (%)',ylabel='Inlet-flux conductance error (%)',title='Energy accuracy does not imply exact flux')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid()
    save(fig,'07_generalization_physics')
    print(f'Saved comparison figures to {F}')


if __name__=='__main__':make_figures()
