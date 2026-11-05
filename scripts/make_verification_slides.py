"""Create two evidence-based verification figures and interview slides."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flow_network import Lattice
OUT=ROOT/'reports/verification';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':220})
TEAL='#176b87';ORANGE='#e28b22';NAVY='#23394b'
d=pd.read_csv(ROOT/'results/solver_diagnostics.csv')
records=[]
def check(name,case,expected,computed,kind='equality'):
 error=abs(computed-expected)
 passed=bool(np.isclose(computed,expected,rtol=1e-10,atol=1e-10)) if kind=='equality' else bool(computed>=expected-1e-10)
 assert passed,(name,case,expected,computed)
 records.append(dict(test=name,case=case,criterion=kind,expected=expected,computed=computed,absolute_difference=error,passed=passed))
for n in [2,3,9,25]:
 grid=Lattice(n)
 for g in [.1,1.,7.]:
  r=grid.solve(np.full(grid.edge_count,g),details=True)
  check('Uniform grid',f'n={n}, g={g}',g*n/(n-1),r.conductance)
  np.testing.assert_allclose(r.heads,np.tile(np.linspace(1,0,n),n),atol=2e-12)
grid=Lattice(5);g=np.zeros(grid.edge_count);g[:4]=[1,2,3,4]
check('Series path','four unequal links',1/np.sum(1/g[:4]),grid.solve(g).conductance)
g[4:8]=[2,4,6,8]
check('Parallel paths','two unequal series branches',3/np.sum(1/g[:4]),grid.solve(g).conductance)
grid=Lattice(11);g=np.ones(grid.edge_count)
g[(grid.u%11==5)&(grid.v==grid.u+1)]=0
check('Disconnected cut','full vertical cut',0.,grid.solve(g).conductance)
for seed in range(12):
 grid=Lattice(15);rng=np.random.default_rng(seed);g=grid.generate(.65,2,rng);base=grid.solve(g).conductance
 check('Conductance scaling',f'seed={seed}, scale=3',3*base,grid.solve(3*g).conductance)
 increased=g.copy();increased[rng.choice(len(g),30,replace=False)]+=1
 check('Rayleigh monotonicity',f'seed={seed}, 30 edges increased',base,grid.solve(increased).conductance,'greater_or_equal')
a=pd.DataFrame(records);a.to_csv(OUT/'analytical_results.csv',index=False)
summary={'diagnostic_realizations':len(d),'spanning_realizations':int(d.spanning.sum()),'max_interior_residual':float(d.max_residual.max()),'max_relative_in_out_error':float(d.relative_balance_error.max()),'max_absolute_energy_error':float(d.energy_error.max()),'minimum_finite_head':float(d.min_head.min()),'maximum_finite_head':float(d.max_head.max()),'analytical_check_records':len(a),'analytical_passed':bool(a.passed.all()),'max_equality_absolute_error':float(a[a.criterion=='equality'].absolute_difference.max()),'minimum_monotonicity_margin':float((a[a.criterion=='greater_or_equal'].computed-a[a.criterion=='greater_or_equal'].expected).min())}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
figures=[]
fig,axes=plt.subplots(2,2,figsize=(16,9));fig.subplots_adjust(left=.07,right=.97,top=.84,bottom=.10,wspace=.25,hspace=.55)
fig.suptitle('Numerical verification: conservation, energy and head bounds',fontsize=24,color=NAVY,y=.97)
fig.text(.5,.91,f'{len(d):,} independent random networks | n=25 | float64 | unit head difference',ha='center',fontsize=15,color=TEAL)
for ax,col,title,xlabel in [(axes[0,0],'max_residual','Internal flow conservation','log10(maximum absolute node residual)'),(axes[0,1],'relative_balance_error','Inlet–outlet balance','log10(relative flow mismatch)'),(axes[1,0],'energy_error','Energy identity: E = Q_in × Δh','log10(absolute energy / inlet-flow discrepancy)')]:
 values=d[col].to_numpy();positive=values[values>0]
 if len(positive):ax.hist(np.log10(positive),bins=25,color=TEAL,edgecolor='white')
 ax.set(title=title,xlabel=xlabel,ylabel='Number of realizations');ax.grid(axis='y',alpha=.2)
 ax.text(.97,.95,f'Maximum: {values.max():.2e}\nExact zeros: {(values==0).sum():,}',transform=ax.transAxes,ha='right',va='top',fontsize=12,bbox=dict(facecolor='white',edgecolor='#d7e5e9',alpha=.95))
ax=axes[1,1]
ax.plot(d.replicate,d.min_head,color=TEAL,label='Minimum finite head',linewidth=2)
ax.plot(d.replicate,d.max_head,color=ORANGE,label='Maximum finite head',linewidth=2)
ax.set(title='Maximum principle: 0 ≤ h ≤ 1',xlabel='Realization index',ylabel='Finite hydraulic head',ylim=(-.15,1.15));ax.legend(fontsize=11,loc='center right');ax.grid(alpha=.2)
ax.text(.03,.48,f'Maximum excess above 1: {max(0.,d.max_head.max()-1):.2e}\nWithin floating-point roundoff',transform=ax.transAxes,fontsize=11)
fig.text(.07,.025,'Source: results/solver_diagnostics.csv. Histograms exclude exact zeros (counts shown). Floating heads are undefined and excluded from head bounds.',fontsize=10,color='#607681')
figures.append(('01_physical_checks',fig))
fig,axes=plt.subplots(1,2,figsize=(16,9));fig.subplots_adjust(left=.07,right=.97,top=.83,bottom=.25,wspace=.24)
fig.suptitle('Independent analytical benchmarks and physical invariants',fontsize=24,color=NAVY,y=.97)
fig.text(.5,.91,f'{len(a)} recorded checks | all passed | complements the original 24 automated tests',ha='center',fontsize=15,color=TEAL)
eq=a[(a.criterion=='equality')&(a.expected>0)]
for name,sub in eq.groupby('test'):
 axes[0].scatter(sub.expected,sub.computed,s=75,label=name,alpha=.8)
limits=[eq.expected.min()*.7,eq.expected.max()*1.4]
axes[0].plot(limits,limits,ls='--',color='grey',label='Exact agreement')
axes[0].set(xscale='log',yscale='log',xlim=limits,ylim=limits,title='Known solution versus numerical result',xlabel='Analytical / invariant prediction',ylabel='Numerical effective conductance');axes[0].legend(fontsize=11);axes[0].grid(alpha=.2)
labels=[];errors=[]
for name,sub in a[a.criterion=='equality'].groupby('test',sort=False):
 labels.append(name);errors.append(sub.absolute_difference.max())
axes[1].barh(labels,np.maximum(errors,1e-16),color=TEAL)
axes[1].set(xscale='log',xlabel='Maximum absolute error (zero shown at 1e−16)',title='Agreement of equality checks');axes[1].grid(axis='x',alpha=.2)
for i,value in enumerate(errors):axes[1].text(max(value,1e-16)*1.3,i,f'{value:.2e}',va='center',fontsize=11)
axes[1].set_xlim(1e-17,max(max(errors)*30,1e-13))
notes=['Uniform grid: G = g n/(n−1); linear head verified for 4 sizes × 3 conductances.', 'Series: G = 1/Σ(1/g); parallel: branch conductances add; disconnected cut: G = 0.', 'Scaling: G(3g) = 3G(g). Rayleigh monotonicity: increasing 30 edges never reduced G (12 seeds).']
for y,line in zip([.16,.115,.07],notes):fig.text(.07,y,'• '+line,fontsize=13,color=NAVY)
fig.text(.07,.025,'Source: reports/verification/analytical_results.csv; test definitions: tests/test_model.py. These are implementation checks, not geological field validation.',fontsize=10,color='#607681')
figures.append(('02_analytical_checks',fig))
with PdfPages(OUT/'verification_slides.pdf') as pdf:
 for name,figure in figures:
  figure.savefig(OUT/(name+'.png'),facecolor='white')
  figure.savefig(OUT/(name+'.pdf'),facecolor='white')
  pdf.savefig(figure,facecolor='white')
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
for name,figure in figures:
 slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.shapes.add_picture(str(OUT/(name+'.png')),0,0,width=Inches(16),height=Inches(9));plt.close(figure)
prs.save(OUT/'verification_slides.pptx')
print(json.dumps(summary,indent=2))
