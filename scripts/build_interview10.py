"""Extend the interview deck with PINN architecture and measured 3D responses."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Reuse the verified eight-slide content and renderer; preserve the old deck.
source=(ROOT/'scripts/build_interview8.py').read_text()
source=source.replace("OUT=ROOT/'reports/interview8'","OUT=ROOT/'reports/interview10'")
source=source.replace('interview_8_slides','interview_10_slides').replace('۸ اسلاید','۱۰ اسلاید').replace('۸ اسلایدی','۱۰ اسلایدی')
source=source.replace("'slide_count':len(slides)","'slide_count':len(slides)")
source=source.replace("'scope':'original direct-solver study; no PINN slides'","'scope':'direct-solver study with parametric graph PINN architecture and paired 3D results'")
source=source.replace("'reports/verification/summary.json'],'slides'", "'reports/verification/summary.json','parametric_pinn/results/scenario_comparison.csv','parametric_pinn/model.py'],'slides'")
source=source.replace("print('Created 8-slide PowerPoint, PDF and Persian notes.')","print('Created 10-slide PowerPoint, PDF and Persian notes.')")
insert=r'''
# Diagram: exact architecture, deterministic drawing for technical accuracy.
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
fig,ax=plt.subplots(figsize=(14.7,5.7));fig.subplots_adjust(left=.005,right=.995,bottom=.015,top=.985)
ax.set(xlim=(0,1),ylim=(-.015,1));ax.axis('off')
def block(x,y,w,h,title,body,face='#eff6f8',edge='#176b87'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=.018',facecolor=face,edgecolor=edge,linewidth=1.6))
 ax.text(x+w/2,y+h-.04,title,ha='center',va='top',fontsize=15,weight='bold',color='#20394b')
 ax.text(x+w/2,y+h/2-.035,body,ha='center',va='center',fontsize=13.5,color='#20394b',linespacing=1.6)
def arrow(points,color='#176b87'):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=16,color=color,lw=2))
block(.015,.67,.18,.24,'Parameter inputs','p, δ, σ','#fff0dd','#e28b22')
block(.015,.19,.18,.29,'Realization input','Full field g\n1,200 edge values'.replace('\\n','\n'))
block(.245,.69,.19,.20,'Global features','6 features\nincludes p, δ, σ'.replace('\\n','\n'),'#fff0dd','#e28b22')
block(.245,.19,.19,.29,'Graph preprocessing','Components + Dijkstra\nInitial h₀, 15 node features\nPhysical transition matrix P'.replace('\\n','\n'))
block(.49,.68,.21,.23,'Coefficient MLP','6 → 16 → 2\nOutputs α, β'.replace('\\n','\n'))
block(.49,.19,.21,.29,'Node encoder MLP','21 → 32 → 32 → 1\n15 node + 6 global features\nCorrects the initial head'.replace('\\n','\n'))
block(.76,.19,.22,.72,'Physics layers','Fixed head updates\nK = 256 or 2048\n\nPredicted head h\nG from Dirichlet energy\nEdge-flow diagnostics'.replace('\\n','\n'),'#e7f2fa')
arrow([(.195,.79),(.245,.79)],'#e28b22')
arrow([(.195,.335),(.245,.335)])
arrow([(.34,.48),(.34,.69)])
arrow([(.435,.79),(.49,.79)],'#e28b22')
arrow([(.435,.72),(.46,.72),(.46,.43),(.49,.43)],'#e28b22')
arrow([(.435,.28),(.49,.28)])
arrow([(.7,.79),(.76,.79)])
arrow([(.7,.335),(.76,.335)])
arrow([(.35,.19),(.35,.11),(.735,.11),(.735,.53),(.76,.53)])
ax.text(.58,.045,'Training: energy + conservation residual loss  •  1,939 trainable parameters',ha='center',fontsize=12.5,color='#617581')
save(fig,'pinn_architecture')
# Three 3D panels use the same matched test ensemble, never invented outputs.
comparison=pd.read_csv(ROOT/'parametric_pinn/results/scenario_comparison.csv')
measured=comparison[comparison.method=='pinn_accurate'].sort_values(['sigma','delta'])
measured.to_csv(OUT/'response3d_data.csv',index=False)
deltas=np.sort(measured.delta.unique());sigmas=np.sort(measured.sigma.unique());xx,yy=np.meshgrid(deltas,sigmas)
def field(name):return measured.pivot(index='sigma',columns='delta',values=name).loc[sigmas,deltas].to_numpy()
fig=plt.figure(figsize=(14.7,5.8));fig.subplots_adjust(left=.015,right=.97,bottom=.13,top=.9,wspace=.15)
for k,(column,title,zlabel,cmap) in enumerate([
 ('mean_reference','Mean conductance','Mean G_eff','viridis'),
 ('positive_mean_abs_relative_error','Conductance error','Mean relative error (%)','plasma'),
 ('mean_free_node_rmse','Head-field error','Mean free-node RMSE','cividis')],1):
 axis=fig.add_subplot(1,3,k,projection='3d');zz=field(column)
 if k==2:zz=100*zz
 axis.plot_surface(xx,yy,zz,cmap=cmap,alpha=.8,edgecolor='white',linewidth=.7,antialiased=True,shade=True)
 axis.scatter(xx.ravel(),yy.ravel(),zz.ravel(),s=22,color='#20394b',depthshade=False)
 if k==1:
  axis.plot_wireframe(xx,yy,field('mean_predicted'),color='#e28b22',linewidth=2)
  axis.legend(handles=[Line2D([0],[0],color='#176b87',lw=5,label='Direct surface'),Line2D([0],[0],color='#e28b22',lw=2,label='PINN wireframe')],fontsize=10,loc='upper left',bbox_to_anchor=(-.03,1.02))
 axis.set_title(title,fontsize=16,pad=12)
 axis.set_xlabel('Connectivity width δ',fontsize=12,labelpad=9);axis.set_ylabel('Spread σ',fontsize=12,labelpad=7)
 axis.set_zlabel(zlabel,fontsize=11,labelpad=8);axis.tick_params(labelsize=10)
 axis.set_xticks([0,.1,.25]);axis.set_yticks([0,1,2]);axis.view_init(elev=25,azim=-130)
 axis.set_box_aspect((1.2,1,.85))
 axis.zaxis.get_offset_text().set_fontsize(9)
fig.text(.5,.025,'Dots: 9 evaluated parameter pairs. Facets only join those points. Error panels use positive-flow cases.',ha='center',fontsize=12,color='#617581')
save(fig,'response3d')
newslides=[dict(title='One trained parametric graph PINN',section='03 / PARAMETRIC PINN',left=None,right=None,cards=None,wide='pinn_architecture',footer='Same weights for new fields • Per-input graph preprocessing and K fixed head updates still run • No new sparse solve.',note='در این مدل دو MLP داریم که با هم آموزش می‌بینند. پارامترهای p و دلتا و سیگما وارد هر دو بخش می‌شوند و میدان کامل رسانایی هم لازم است. یکی هد اولیه را اصلاح می‌کند و دیگری ضرایب به‌روزرسانی را می‌دهد. سپس لایه‌های فیزیکی هد و رسانایی را محاسبه می‌کنند؛ وزن‌ها دوباره آموزش نمی‌بینند.'),
 dict(title='How the response changes across δ and σ',section='04 / MEASURED 3D RESPONSE',left=None,right=None,cards=None,wide='response3d',footer='Matched test: 1,000 new fields per cell • n = 25 • PINN K = 2048 • Surface shapes are not additional simulations.',note='این سه نمودار از نتایج واقعی ۹ ترکیب دلتا و سیگما هستند؛ هر ترکیب هزار شبکه آزمون دارد. نمودار اول میانگین رسانایی حل مستقیم و PINN را مقایسه می‌کند که تقریباً روی هم هستند. دو نمودار دیگر خطای رسانایی و هد را نشان می‌دهند. سطح‌ها فقط نقاط محاسبه‌شده را به هم وصل می‌کنند و به معنی بررسی تمام ترکیب‌های ممکن نیستند.')]
slides[2:2]=newslides
for i,item in enumerate(slides,1):item['section']=f'{i:02d} / '+item['section'].split(' / ',1)[1]
'''
# A new full-width figure layout for the architecture and 3D result panels.
renderer=""" if spec.get('wide'):
  imagepath=FIG/(spec['wide']+'.png')
  from PIL import Image
  iw,ih=Image.open(imagepath).size;w=14.7;h=w*ih/iw
  if h>5.95:h=5.95;w=h*iw/ih
  slide.shapes.add_picture(str(imagepath),Inches(.65+(14.7-w)/2),Inches(2.1+(5.95-h)/2),width=Inches(w),height=Inches(h))
  bodyhtml='<div class="wide"><img src="figures/'+spec['wide']+'.png"></div>'
 elif spec['left']:
"""
source=source.replace('prs=Presentation();',insert+'\nprs=Presentation();',1)
source=source.replace(" if spec['left']:\n",renderer,1)
source=source.replace('.columns{display:flex;', '.wide{height:5.95in;margin-top:.2in;display:flex;align-items:center;justify-content:center}.wide img{max-width:14.7in;max-height:5.95in;object-fit:contain}.columns{display:flex;')
# __file__ must point to this build script so all paths resolve to the repository.
exec(compile(source,str(__file__),'exec'),{'__file__':str(__file__),'__name__':'__main__'})
