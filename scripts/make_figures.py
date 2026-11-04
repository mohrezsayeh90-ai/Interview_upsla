#!/usr/bin/env python3
"""Build publication-quality figures exclusively from saved simulations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from flow_network import Lattice

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT/"results", ROOT/"reports"/"figures"
COLORS = ["#176b87", "#e28b22", "#a43d52"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.titleweight":"bold","figure.dpi":120,"savefig.dpi":200,
                     "axes.labelcolor":"#23394b","text.color":"#23394b",
                     "grid.alpha":.2})


def save(fig, name):
    fig.savefig(FIG/(name+".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(FIG/(name+".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def network():
    grid = Lattice(15)
    rng = np.random.default_rng(785)
    examples = []
    for wanted, p in [(False, .47), (True, .55)]:
        while True:
            g = grid.generate(p, 1, rng)
            r = grid.solve(g, details=True)
            if r.spanning == wanted:
                examples.append((p,g,r))
                break
    fig, axes = plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    segments = grid.xy[np.column_stack((grid.u,grid.v))]
    for ax, (p,g,r) in zip(axes,examples):
        ax.add_collection(LineCollection(segments[g == 0], colors="#dce2e7",linewidths=.4,linestyles="dotted"))
        ax.add_collection(LineCollection(segments[g > 0], colors="#99acb5",linewidths=1.1))
        if r.spanning:
            flowing = np.abs(r.flows)>1e-10
            widths = .6+3*np.sqrt(np.abs(r.flows[flowing])/max(np.abs(r.flows).max(),1e-30))
            ax.add_collection(LineCollection(segments[flowing],colors="#e28b22",linewidths=widths))
        known = np.isfinite(r.heads)
        ax.scatter(*grid.xy[~known].T,c="#c1c7cc",s=9,zorder=3)
        sc=ax.scatter(*grid.xy[known].T,c=r.heads[known],vmin=0,vmax=1,cmap="viridis",s=14,zorder=4)
        ax.set(xlim=(-.6,14.6),ylim=(-.6,14.6),aspect="equal",xticks=[],yticks=[])
        ax.set_title(f"{'Spanning' if r.spanning else 'Disconnected'} | p={p:.2f}\n$G_{{eff}}$={r.conductance:.4f}")
        ax.text(0,-.07,"h = 1 (left)     |     h = 0 (right)",transform=ax.transAxes,fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.colorbar(sc, ax=axes,shrink=.75,label="Hydraulic head h")
    save(fig,"01_network")


def uncertainty(summary):
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for color,(sigma,s) in zip(COLORS,summary.groupby("sigma")):
        axes[0].errorbar(s.delta,s["mean"],yerr=1.96*s.mcse,color=color,marker="o",capsize=3,label=f"σ = {sigma:g}")
    axes[0].set(xlabel="Connectivity uncertainty δ",ylabel="Mean effective conductance",title="Magnitude of flow")
    axes[0].legend()
    s=summary[summary.sigma==1]
    axes[1].errorbar(s.delta,s.p_span,yerr=[s.p_span-s.p_span_low,s.p_span_high-s.p_span],color=COLORS[0],marker="o",capsize=3)
    axes[1].axhline(.5,color="grey",linestyle=":",linewidth=1)
    axes[1].set(xlabel="Connectivity uncertainty δ",ylabel="Probability of a spanning path",title="Probability of any flow",ylim=(.45,.58))
    for ax in axes: ax.grid()
    save(fig,"02_uncertainty")


def distributions(samples):
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for color,sigma in zip(COLORS,[0.,1.,2.]):
        g=samples[(samples.delta==.25)&(samples.sigma==sigma)].geff.to_numpy()
        order=np.sort(g)
        axes[0].step(np.r_[0,order],np.r_[0,np.arange(1,len(g)+1)/len(g)],where="post",color=color,label=f"σ={sigma:g}")
        pos=g[g>0]
        axes[1].hist(np.log10(pos),bins=np.linspace(-4,.1,46),density=True,histtype="step",color=color,linewidth=1.8,label=f"σ={sigma:g}")
    axes[0].set(xlabel="Effective conductance",ylabel="Empirical CDF",title="Mixed distribution: an atom at zero",xlim=(-.01,.75),ylim=(0,1.02))
    axes[1].set(xlabel="log₁₀(G_eff), conditional on G_eff > 0",ylabel="Density in log space",title="Connected networks only")
    axes[0].legend();axes[1].legend()
    for ax in axes: ax.grid()
    save(fig,"03_distribution")


def connectivity():
    s=pd.read_csv(RES/"connectivity_summary.csv")
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    axes[0].errorbar(s.p,s.p_span,yerr=[s.p_span-s.p_span_low,s.p_span_high-s.p_span],marker="o",color=COLORS[0],capsize=3)
    axes[1].errorbar(s.p,s["mean"],yerr=1.96*s.mcse,marker="o",color=COLORS[1],capsize=3)
    axes[0].set(ylabel="Spanning probability",title="Connectivity transition",ylim=(-.02,1.02))
    axes[1].set(ylabel="Mean effective conductance",title="Nonlinear flow response")
    for ax in axes:
        ax.axvline(.5,color="#a43d52",ls=":",label="Infinite-lattice p_c = 0.5")
        ax.set_xlabel("Fixed link probability p");ax.grid();ax.legend(fontsize=8)
    save(fig,"04_connectivity")


def sensitivity():
    s=pd.read_csv(RES/"sobol_summary.csv")
    labels=["δ", "σ", "p latent\nvariable", "Topology\nfield", "Conductance\nfield"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for ax,out,title in zip(axes,["geff","spanning"],["Effective conductance", "Existence of a spanning path"]):
        t=s[s.output==out]
        x=np.arange(5)
        for offset,key,color,label in [(-.18,"first",COLORS[0],"First order"),(.18,"total",COLORS[1],"Total order")]:
            y=t[key].to_numpy()
            errors=np.maximum(0,np.vstack((y-t[key+"_low"].to_numpy(),t[key+"_high"].to_numpy()-y)))
            ax.bar(x+offset,y,width=.34,color=color,label=label,yerr=errors,capsize=3)
        ax.set(xticks=x,xticklabels=labels,ylabel="Sobol index",title=title,ylim=(-.1,.95))
        ax.axhline(0,color="grey",linewidth=.6);ax.legend(fontsize=9);ax.grid(axis="y")
    save(fig,"05_sensitivity")
    c=pd.read_csv(RES/"sobol_convergence.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    for ax,order in zip(axes,["first","total"]):
        for factor,sub in c[c.output=="geff"].groupby("factor",sort=False):
            ax.plot(sub.samples,sub[order],marker=".",label=factor)
        ax.set(xlabel="Independent base pairs",ylabel=order+"-order Sobol index",xscale="log",title="Estimator convergence")
        ax.grid()
    axes[1].legend(fontsize=8)
    save(fig,"08_sobol_convergence")


def convergence():
    c=pd.read_csv(RES/"mc_convergence.csv")
    fs=pd.read_csv(RES/"finite_size_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for color,(delta,s) in zip(COLORS,c[c.sigma==1].groupby("delta")):
        axes[0].errorbar(s.checkpoint,s["mean"],yerr=1.96*s.mcse,marker=".",color=color,capsize=2,label=f"δ={delta:g}")
    axes[0].set(xlabel="Number of Monte Carlo samples",ylabel="Mean G_eff ± MC 95% CI",xscale="log",title="Statistical convergence")
    for color,(delta,s) in zip(COLORS,fs.groupby("delta")):
        axes[1].errorbar(s.n,s["mean"],yerr=1.96*s.mcse,marker="o",color=color,capsize=3,label=f"δ={delta:g}")
    axes[1].set(xlabel="Nodes along each side n",ylabel="Mean G_eff ± MC 95% CI",title="Finite-size sensitivity (σ = 1)")
    for ax in axes: ax.grid();ax.legend(fontsize=9)
    save(fig,"06_convergence")


def normalization(summary):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    for color,(delta,s) in zip(COLORS,summary.groupby("delta")):
        axes[0].plot(s.sigma,s["mean"],marker="o",color=color,label=f"δ={delta:g}")
        axes[1].plot(s.sigma,s.mean_unit_mean_g,marker="o",color=color,label=f"δ={delta:g}")
    axes[0].set(title="Specified model: median(g) = 1",ylabel="Mean G_eff")
    axes[1].set(title="Control: mean(g) = 1",ylabel="Mean G_eff (rescaled exactly)")
    for ax in axes:ax.set_xlabel("Log-conductance standard deviation σ");ax.grid();ax.legend(fontsize=9)
    save(fig,"07_normalization")


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(RES/"scenario_summary.csv")
    samples=pd.read_csv(RES/"scenario_samples.csv.gz")
    network();uncertainty(summary);distributions(samples);connectivity();sensitivity();convergence();normalization(summary)
    print(f"Saved figures in {FIG}")


if __name__=="__main__":main()
