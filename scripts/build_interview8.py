"""Build an eight-slide interview deck and short Persian presenter notes."""
from pathlib import Path
import sys,json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from weasyprint import HTML
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flow_network import Lattice
OUT=ROOT/'reports/interview8';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
NAVY='20394B';TEAL='176B87';ORANGE='E28B22';GRAY='617581';PALE='EFF6F8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#20394b','text.color':'#20394b','axes.titleweight':'bold','savefig.dpi':220})
s=pd.read_csv(ROOT/'results/scenario_summary.csv');raw=pd.read_csv(ROOT/'results/scenario_samples.csv.gz');sob=pd.read_csv(ROOT/'results/sobol_summary.csv');verification=json.loads((ROOT/'reports/verification/summary.json').read_text())
base=s[(s.delta==.25)&(s.sigma==1)].iloc[0]
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),bbox_inches='tight',facecolor='white')
 plt.close(fig)
# Network illustration with the same verified solver.
grid=Lattice(11);rng=np.random.default_rng(77)
while True:
 g=grid.generate(.6,1,rng);r=grid.solve(g,details=True)
 if r.spanning:break
fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained');segments=grid.xy[np.column_stack((grid.u,grid.v))]
ax.add_collection(LineCollection(segments[g>0],colors='#b3c5cd',linewidths=1.1))
flow=np.abs(r.flows)>1e-10
ax.add_collection(LineCollection(segments[flow],colors='#e28b22',linewidths=.7+3*np.sqrt(np.abs(r.flows[flow])/np.max(np.abs(r.flows)))))
known=np.isfinite(r.heads);ax.scatter(*grid.xy[~known].T,c='#d1d9dd',s=14)
im=ax.scatter(*grid.xy[known].T,c=r.heads[known],cmap='viridis',vmin=0,vmax=1,s=33,zorder=3)
fig.colorbar(im,ax=ax,shrink=.7,label='Head h')
ax.text(-.8,5,'h = 1',rotation=90,ha='center',va='center',weight='bold');ax.text(10.8,5,'h = 0',rotation=90,ha='center',va='center',weight='bold')
ax.text(5,11,'No external flow',ha='center',fontsize=15);ax.text(5,-1,'No external flow',ha='center',fontsize=15)
ax.set(xlim=(-1.4,11.4),ylim=(-1.6,11.6),aspect='equal',xticks=[],yticks=[])
for sp in ax.spines.values():sp.set_visible(False)
save(fig,'network')
fig,ax=plt.subplots(figsize=(8.6,5.2),layout='constrained')
for color,(sigma,sub) in zip(['#176b87','#e28b22','#a43d52'],s.groupby('sigma')):
 ax.errorbar(sub.delta,sub['mean'],yerr=1.96*sub.mcse,color=color,marker='o',markersize=8,lw=2.5,capsize=5,label=f'σ = {sigma:g}')
ax.set(xlabel='Connectivity uncertainty δ',ylabel='Mean effective conductance',xticks=[0,.1,.25],ylim=(0,.14));ax.legend(fontsize=16);ax.grid(alpha=.2);save(fig,'uncertainty_mean')
values=np.sort(raw[(raw.delta==.25)&(raw.sigma==1)].geff.to_numpy())
fig,ax=plt.subplots(figsize=(8.6,5.2),layout='constrained')
ax.step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)/len(values)],where='post',color='#176b87',lw=2.5)
ax.scatter([0,base.q95],[base.p_zero,.95],color='#e28b22',s=90,zorder=5)
ax.annotate('48.62% at zero',xy=(0,base.p_zero),xytext=(.09,.3),arrowprops=dict(arrowstyle='->',color='#e28b22'),fontsize=17)
ax.axhline(.95,color='#b8c7cf',ls=':',lw=1);ax.text(.43,.89,'95th percentile',fontsize=14,color='#617581')
ax.set(xlabel='Effective conductance',ylabel='Cumulative probability',xlim=(-.025,.65),ylim=(0,1.04));ax.grid(alpha=.2);save(fig,'distribution')
factors=['p_latent','delta','topology','sigma','conductance_field'];t=sob[sob.output=='geff'].set_index('factor').loc[factors]
fig,ax=plt.subplots(figsize=(8.6,5.2),layout='constrained');x=np.arange(5)
for offset,key,color,label in [(-.18,'first','#176b87','First-order: individual effect'),(.18,'total','#e28b22','Total-order: includes interactions')]:
 y=t[key].to_numpy();err=np.maximum(0,np.vstack([y-t[key+'_low'].to_numpy(),t[key+'_high'].to_numpy()-y]));ax.bar(x+offset,y,width=.34,color=color,label=label,yerr=err,capsize=3)
ax.set(xticks=x,xticklabels=['Global p\ndraw U','Width\nδ','Topology\nfield W','Spread\nσ','Conductance\nfield Z'],ylabel='Sobol index',ylim=(-.04,1.04));ax.legend(fontsize=12,loc='upper right');ax.grid(axis='y',alpha=.2);save(fig,'sensitivity')
# Shared slide specification feeds editable PowerPoint and matching HTML/PDF.
slides=[]
def add(title,section,left=None,right=None,cards=None,footer='',note=''):
 slides.append(dict(title=title,section=section,left=left,right=right,cards=cards,footer=footer,note=note))
add('Flow through a random network','01 / THE PROBLEM',left='network',right=[
 ('A simplified fracture network','25 × 25 nodes; 1,200 potential links.'),
 ('A fixed driving force','Left head = 1; right head = 0. No external flow at top or bottom.'),
 ('What do we want to predict?','Total flow and effective conductance: G_eff = Q_total / Δh.')],footer='Dimensionless model • Illustration uses 11 × 11 nodes for clarity.',note='در این پروژه، سنگ شکسته را با یک شبکه ساده نمایش دادم. اختلاف هد چپ و راست، جریان ایجاد می‌کند؛ بالا و پایین جریان خارجی ندارند. هدفم محاسبه جریان کل و رسانایی مؤثر شبکه است.')
add('From uncertain links to a flow prediction','02 / MODEL AND SOLUTION',cards=[
 ('01  Connectivity','One global p per network:\np ∼ Uniform(0.5 − δ, 0.5 + δ).\nEach link is active with probability p.'),
 ('02  Link conductance','Active links: ln(g) ∼ Normal(0, σ²).\nInactive links: g = 0.\nLarger σ means greater heterogeneity.'),
 ('03  Solve node heads','Find spanning components.\nEnforce flow balance at free nodes:\nΣⱼ gᵢⱼ(hᵢ − hⱼ) = 0.'),
 ('04  Compute the output','Solve the sparse linear system for all unknown heads together.\nThen calculate link flows and G_eff.')],footer='δ controls uncertainty in connectivity; σ controls spread in log-conductance.',note='برای هر شبکه ابتدا یک احتمال اتصال انتخاب کردم و لینک‌ها را فعال یا غیرفعال کردم. رسانایی لینک‌های فعال هم تصادفی است. سپس هد همه گره‌های مجهول را هم‌زمان از بقای جریان حل کردم و جریان‌ها را به دست آوردم.')
add('Does the numerical solution obey physics?','03 / PHYSICAL VERIFICATION',cards=[
 ('Internal conservation','Maximum node residual\n2.75 × 10⁻¹³\nΣⱼ Qᵢⱼ ≈ 0'),
 ('Inlet–outlet balance','Maximum relative mismatch\n7.91 × 10⁻¹²\nQ_in ≈ Q_out'),
 ('Energy consistency','Maximum absolute discrepancy\n2.41 × 10⁻¹³\nΣₑ gₑ(Δhₑ)² ≈ Q_in × Δh'),
 ('Head bounds','0 ≤ h ≤ 1, within roundoff.\nMaximum excess above 1:\n4.66 × 10⁻¹⁴')],footer='1,000 independent random networks • float64 • Floating, undefined heads excluded from bounds.',note='روی هزار شبکه مستقل، چهار بررسی فیزیکی انجام دادم: بقای جریان گره‌ها، برابری ورودی و خروجی، تساوی انرژی و کران هد. خطاها در حدود دقت محاسباتی بودند. این بررسی‌ها صحت حل عددی را نشان می‌دهند.')
add('Does the solver reproduce known answers?','04 / ANALYTICAL BENCHMARKS',cards=[
 ('Uniform grid','Exact: G = g n / (n − 1).\nLinear head profile also verified.\n4 sizes × 3 conductances.'),
 ('Series and parallel paths','Series: G = 1 / Σ(1/g).\nParallel: branch conductances add.\nBoth agree with analytical answers.'),
 ('Disconnected network','A complete vertical cut removes\nevery spanning path.\nComputed conductance: exactly zero.'),
 ('Scaling and monotonicity','G(3g) = 3G(g).\nIncreasing edge conductances\nnever reduced G in 12 test cases.')],footer='39 recorded benchmark / invariant checks: all passed • Original automated suite: 24 tests.',note='برای بررسی مستقل، مسئله‌هایی با جواب معلوم هم حل کردم: شبکه یکنواخت، مسیر سری و موازی و قطع کامل شبکه. مقیاس‌پذیری و یکنوایی هم بررسی شدند. هر ۳۹ مورد ثبت‌شده موفق بود؛ این تعداد با ۲۴ آزمون خودکار اولیه فرق دارد.')
add('How do uncertainties change the mean flow?','05 / UNCERTAINTY ANALYSIS',left='uncertainty_mean',right=[
 ('Wider connectivity uncertainty','At σ = 1, mean G increases\n0.0183 → 0.0992 as δ: 0 → 0.25.'),
 ('More heterogeneous conductance','At fixed δ, mean G decreases\nas σ increases in these scenarios.'),
 ('Why this happens','Connectivity has a nonlinear response.\nWeak links can become bottlenecks.')],footer='9 scenarios × 10,000 realizations • Error bars: Monte Carlo 95% confidence intervals.',note='در هرکدام از ۹ سناریو، ده هزار شبکه ساختم. با افزایش دلتا، میانگین رسانایی بیشتر شد، چون سود شبکه‌های پراتصال از افت شبکه‌های کم‌اتصال بیشتر بود. با افزایش سیگما، لینک‌های ضعیف‌تر گلوگاه ایجاد کردند و میانگین کاهش یافت. این نتایج برای دامنه بررسی‌شده هستند.')
add('The mean alone hides the output distribution','06 / UNCERTAINTY ANALYSIS',left='distribution',right=[
 ('Almost half have no flow',f'{100*base.p_zero:.2f}% of realizations have G = 0.'),
 ('A broad positive component',f'Mean = {base["mean"]:.4f}\nMedian = {base["median"]:.4f}\n95th percentile = {base.q95:.4f}'),
 ('Report both parts','Connection probability and positive-flow magnitude answer different questions.')],footer='Reference scenario: δ = 0.25, σ = 1 • A mixed distribution: an atom at zero plus positive values.',note='در سناریوی مرجع، حدود ۴۹ درصد شبکه‌ها اصلاً جریان نداشتند. بقیه، جریان‌های مثبت با پراکندگی زیاد داشتند؛ برای همین میانگین و میانه خیلی متفاوت‌اند. پس فقط یک میانگین کافی نیست و باید احتمال صفر و صدک‌های توزیع را هم گزارش کنیم.')
add('Connectivity is the dominant uncertain influence','07 / SENSITIVITY ANALYSIS',left='sensitivity',right=[
 ('Method: grouped Sobol / Jansen','Global variance-based analysis.\nFirst-order: individual effect.\nTotal-order: includes interactions.'),
 ('Connectivity dominates',f'Global p draw U: total index {t.loc["p_latent","total"]:.3f}.\nδ: first {t.loc["delta","first"]:.3f}, total {t.loc["delta","total"]:.3f}.'),
 ('Interactions matter','δ has a much larger total effect.\nTotal indices overlap; do not add them.')],footer='p = 0.5 + δU; independent groups δ, σ, U, W, Z • 16,384 base pairs • 1,000 bootstrap resamples.',note='برای پیدا کردن عامل‌های مهم از Sobol استفاده کردم، چون اثرها غیرخطی و ترکیبی‌اند. آبی اثر مستقل و نارنجی اثر همراه با برهم‌کنش‌هاست. اتصال عامل غالب بود. فاصله زیاد دو میله دلتا نشان می‌دهد اثر آن با بقیه ورودی‌ها ترکیب می‌شود. رتبه‌بندی به دامنه ورودی‌ها وابسته است.')
add('What we learned — and how to extend the model','08 / DISCUSSION',cards=[
 ('01  Main finding: connectivity','Connection structure and its interactions\ndominate output variability\nover the selected input ranges.'),
 ('02  Main finding: distribution','Mean conductance is not enough.\nReport zero-flow probability,\nconditional flow and an upper quantile.'),
 ('03  Model limitations','Regular 2D geometry; steady linear flow.\nExploratory priors; finite-size effects.\nNo geological field calibration.'),
 ('04  Toward realistic fractured rocks','Use 3D discrete fracture networks,\naperture-based conductance and correlation.\nCalibrate with head, flow or tracer data.')],footer='Numerical verification establishes implementation correctness; it does not establish field prediction accuracy.',note='چهار پیام اصلی این است: اتصال مهم‌ترین عامل بود؛ میانگین به‌تنهایی کافی نیست؛ مدل ساده، دوبعدی و کالیبره‌نشده است؛ و برای سنگ واقعی باید هندسه شکستگی سه‌بعدی، رابطه رسانایی با بازشدگی و داده میدانی اضافه شود.')
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
def box(slide,x,y,w,h,text,size=22,bold=False,color=NAVY):
 shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
 tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0
 for i,line in enumerate(text.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(7)
 return shape
pages=[]
for i,spec in enumerate(slides,1):
 slide=prs.slides.add_slide(prs.slide_layouts[6]);box(slide,.65,.35,14.7,.35,spec['section'],12,True,TEAL);box(slide,.65,.94,14.7,.8,spec['title'],30,True)
 rule=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.65),Inches(1.85),Inches(14.7),Inches(.035));rule.fill.solid();rule.fill.fore_color.rgb=RGBColor.from_string(TEAL);rule.line.fill.background()
 if spec['left']:
  imagepath=FIG/(spec['left']+'.png')
  from PIL import Image
  iw,ih=Image.open(imagepath).size;w=9.25;h=w*ih/iw
  if h>5.65:h=5.65;w=h*iw/ih
  slide.shapes.add_picture(str(imagepath),Inches(.6+(9.25-w)/2),Inches(2.15+(5.65-h)/2),width=Inches(w),height=Inches(h))
  for k,(head,body) in enumerate(spec['right']):
   y=2.25+k*1.82;box(slide,10.15,y,5.15,.5,head,20,True,TEAL);box(slide,10.15,y+.53,5.15,1.25,body,19)
  bodyhtml='<div class="columns"><div class="plot"><img src="figures/'+spec['left']+'.png"></div><div class="right">'+''.join('<div class="item"><h2>'+html.escape(h)+'</h2><p>'+html.escape(b).replace('\n','<br>')+'</p></div>' for h,b in spec['right'])+'</div></div>'
 else:
  for k,(head,body) in enumerate(spec['cards']):
   x=.65+(k%2)*7.55;y=2.2+(k//2)*2.85
   shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(7.15),Inches(2.5));shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(PALE);shape.line.fill.background()
   box(slide,x+.3,y+.25,6.55,.55,head,22,True,TEAL);box(slide,x+.3,y+.9,6.55,1.45,body,21)
  bodyhtml='<div class="cards">'+''.join('<div class="card"><h2>'+html.escape(h)+'</h2><p>'+html.escape(b).replace('\n','<br>')+'</p></div>' for h,b in spec['cards'])+'</div>'
 box(slide,.65,8.38,14.1,.4,spec['footer'],11,False,GRAY);box(slide,15,8.38,.5,.4,str(i),12,True,TEAL)
 tf=slide.notes_slide.notes_text_frame;tf.text=spec['note']
 for p in tf.paragraphs:p.font.name='Noto Naskh Arabic';p.font.size=Pt(16)
 pages.append('<section><div class="section">'+spec['section']+'</div><h1>'+html.escape(spec['title'])+'</h1>'+bodyhtml+'<footer>'+html.escape(spec['footer'])+f'<span>{i}</span></footer></section>')
prs.save(OUT/'interview_8_slides.pptx')
css="""@page{size:16in 9in;margin:0}*{box-sizing:border-box}body{margin:0;font-family:'DejaVu Sans';color:#20394b}section{width:16in;height:9in;padding:.35in .65in;position:relative;break-after:page}section:last-child{break-after:auto}.section{font-size:12pt;font-weight:bold;color:#176b87}h1{font-size:30pt;margin:.3in 0 .28in;padding-bottom:.22in;border-bottom:2px solid #176b87;line-height:1.2}h2{font-size:20pt;color:#176b87;margin:0 0 .15in}p{font-size:19pt;line-height:1.4;margin:0}.columns{display:flex;gap:.3in;align-items:center;height:5.75in;margin-top:.25in}.plot{width:9.2in;flex-shrink:0}.plot img{width:100%;max-height:5.65in;object-fit:contain}.right{width:5.15in}.item{margin-bottom:.35in}.cards{display:grid;grid-template-columns:1fr 1fr;gap:.35in .4in;margin-top:.3in}.card{background:#eff6f8;border-radius:.15in;padding:.25in .3in;height:2.5in}.card h2{font-size:22pt}.card p{font-size:21pt}footer{position:absolute;bottom:.25in;left:.65in;right:.65in;font-size:11pt;color:#617581}footer span{float:right;color:#176b87;font-weight:bold}"""
path=OUT/'interview_8_slides.html';path.write_text('<!doctype html><meta charset="utf-8"><title>Random-network flow: interview presentation</title><style>'+css+'</style>'+''.join(pages))
HTML(filename=str(path)).write_pdf(OUT/'interview_8_slides.pdf')
notes='# یادداشت کوتاه ارائه — ۸ اسلاید\n\n'+ '\n\n'.join(f'## اسلاید {i}: {spec["title"]}\n\n{spec["note"]}' for i,spec in enumerate(slides,1))+'\n'
(OUT/'notes_fa.md').write_text(notes)
noteshtml='<html lang="fa" dir="rtl"><meta charset="utf-8"><style>@page{size:A4;margin:18mm}body{font-family:"Noto Naskh Arabic";font-size:13pt;line-height:1.8;color:#20394b}h1{font-size:20pt;color:#176b87}h2{font-size:15pt;color:#176b87;margin-bottom:2mm}article{break-inside:avoid}p{margin-top:0}</style><h1>یادداشت کوتاه برای ارائه ۸ اسلایدی</h1>'+''.join(f'<article><h2>اسلاید {i}</h2><p>{html.escape(spec["note"])}</p></article>' for i,spec in enumerate(slides,1))+'</html>'
(OUT/'notes_fa.html').write_text(noteshtml);HTML(string=noteshtml).write_pdf(OUT/'notes_fa.pdf')
(OUT/'slide_manifest.json').write_text(json.dumps({'slide_count':len(slides),'language':'English slides, Persian notes','scope':'original direct-solver study; no PINN slides','sources':['results/scenario_summary.csv','results/scenario_samples.csv.gz','results/sobol_summary.csv','reports/verification/summary.json'],'slides':[{'number':i,'title':z['title'],'note_fa':z['note']} for i,z in enumerate(slides,1)]},ensure_ascii=False,indent=2)+'\n')
print('Created 8-slide PowerPoint, PDF and Persian notes.')
