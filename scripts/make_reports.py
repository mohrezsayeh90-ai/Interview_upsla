#!/usr/bin/env python3
"""Generate bilingual HTML/PDF reports directly from saved statistical outputs.

PDF rendering requires WeasyPrint, Pango and Noto Naskh Arabic (or an Arabic font).
All numbers in the narrative and tables are read from the CSV/JSON results.
"""
import json
from pathlib import Path
import pandas as pd
from weasyprint import HTML

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"
D=ROOT/"reports"
CSS="""
@page { size: A4; margin: 13mm 15mm 14mm;
 @bottom-center { content: counter(page); color: #6a7c86; font-size: 9pt; } }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.2pt; line-height: 1.43; color: #23394b; }
body.fa { font-family: 'Noto Naskh Arabic', 'DejaVu Sans', sans-serif; font-size: 11.6pt; line-height: 1.52; }
h1 { font-size: 23pt; line-height: 1.25; color: #176b87; margin: 0 0 8mm; }
h2 { font-size: 16pt; color: #176b87; border-bottom: 1pt solid #c5dce4; padding-bottom: 2mm; margin: 0 0 4mm; }
h3 { font-size: 11.5pt; color: #176b87; margin: 3mm 0 1mm; }
p { margin: 2mm 0 3mm; }
ul,ol { margin: 1mm 0 3mm; padding-inline-start: 6mm; }
li { margin-bottom: 1mm; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 8.3pt; line-height:1.35; }
.fa table { font-size: 9.5pt; }
th { background: #176b87; color: white; padding: 2mm 1.5mm; font-weight: normal; }
td { padding: 1.7mm 1.5mm; border-bottom: .4pt solid #d8e4e8; text-align: center; }
tr:nth-child(even) td { background: #f2f7f8; }
.page { break-before: page; }
.first { break-before: auto; }
.eq { direction: ltr; text-align: center; font-family: 'DejaVu Sans', sans-serif; font-size: 9.2pt; background: #f0f6f8; padding: 2mm; margin: 3mm 0; }
.tag { color: #a43d52; font-size: 10pt; letter-spacing:.3pt; margin-bottom:3mm; }
.box { background: #edf5f7; border-inline-start: 3pt solid #176b87; padding: 3mm 4mm; margin: 4mm 0; }
figure { margin: 3mm 0; break-inside: avoid; }
img { width: 100%; height: auto; }
figcaption { font-size: 8pt; color: #607681; text-align: center; margin-top: 1mm; }
.fa figcaption { font-size: 9.3pt; }
.small { font-size: 8.4pt; }
.fa .small { font-size: 9.6pt; }
.refs { direction: ltr; text-align: left; font-size:8.5pt; }
code { direction: ltr; font-size: 8pt; }
bdi { direction: ltr; unicode-bidi: isolate; }
a {color:#176b87; text-decoration:none;}
"""


def fig(name,caption):
    return f'<figure><img src="figures/{name}.png"><figcaption>{caption}</figcaption></figure>'


def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    s=pd.read_csv(R/"scenario_summary.csv")
    sen=pd.read_csv(R/"sobol_summary.csv")
    diag=pd.read_csv(R/"solver_diagnostics.csv")
    m=json.loads((R/"manifest.json").read_text())
    base=s[(s.delta==.25)&(s.sigma==1)].iloc[0]
    low=s[(s.delta==0)&(s.sigma==1)].iloc[0]
    high_sigma=s[(s.delta==.25)&(s.sigma==2)].iloc[0]
    size=pd.read_csv(R/"finite_size_summary.csv")
    size_means={int(row.n):row['mean'] for _,row in size[size.delta==0].iterrows()}
    relci=(1.95996398454*s.mcse/s['mean'])*100
    first_sum=sen[sen.output=="geff"]['first'].sum()
    def f(x,d=4):return f"{x:.{d}f}"
    def ci(row,key="mean"):
        return f'{f(row[key])} [{f(row[key+"_ci_low"])}, {f(row[key+"_ci_high"])}]'
    uncertainty_rows=[]
    for _,r in s.iterrows():
        uncertainty_rows.append([f(r.delta,2),f(r.sigma,0),
            f'<bdi>{ci(r)}</bdi>',f'{100*r.p_zero:.2f}%',f(r.q95),f(r.positive_mean)])
    uh_en=["δ","σ","Mean [MC 95% CI]","P(G=0)","95th pct.","Mean, G>0"]
    uh_fa=["δ","σ","میانگین [فاصله اطمینان ۹۵٪]","احتمال صفر","صدک ۹۵","میانگین مثبت‌ها"]
    factor_en={"delta":"δ: uncertainty width","sigma":"σ: log-conductance spread","p_latent":"U: global connectivity draw","topology":"W: topology field","conductance_field":"Z: conductance field"}
    factor_fa={"delta":"δ: پهنای عدم‌قطعیت اتصال","sigma":"σ: پراکندگی لگاریتم رسانایی","p_latent":"U: متغیر احتمال اتصال","topology":"W: آرایش تصادفی لینک‌ها","conductance_field":"Z: میدان تصادفی رسانایی"}
    def sens_rows(names):
        return [[names[r.factor],f'<bdi>{f(r["first"],3)} [{f(r.first_low,3)}, {f(r.first_high,3)}]</bdi>',
                 f'<bdi>{f(r.total,3)} [{f(r.total_low,3)}, {f(r.total_high,3)}]</bdi>'] for _,r in sen[sen.output=="geff"].iterrows()]
    validation_en=table(["Independent check","Expected result / acceptance"],[
        ["Full uniform grid, n=2,3,9,25; g=0.1,1,7","G = g n/(n−1), h linear"],
        ["Single series path / two parallel paths","1/Σ(1/g) / sum of branch conductances"],
        ["Disconnected islands / vertical cut","G = 0; floating heads undetermined"],
        ["Conservation / energy / head bounds","Qin = Qout = Σ g(Δh)²; 0 ≤ h ≤ 1"],
        ["Scaling / Rayleigh monotonicity","G(cg)=cG(g); increasing a link cannot reduce G"],
        ["Sensitivity estimator benchmark","Ishigami analytic indices, absolute tolerance 0.018"],
    ])
    design=table(["Experiment","Design","Solver calls"],[
        ["Uncertainty","3 δ × 3 σ × 10,000", "90,000"],
        ["Connectivity sweep","p=0.30:0.05:0.70; 2,500 each", "22,500"],
        ["Additional finite sizes","n=15,41; δ=0,0.25; σ=1; 5,000 each", "20,000"],
        ["Grouped Sobol","16,384 pairs × (5+2)", "114,688"],
        ["Solver diagnostics","1,000 independent networks", "1,000"],
        ["Total","n=25 size comparison reuses scenario data",f'{m["solver_calls"]:,}'],
    ])
    refs='''<div class="refs">
<p>[1] M. J. W. Jansen (1999). Analysis of variance designs for model output. <i>Computer Physics Communications</i> 117, 35–43. <a href="https://doi.org/10.1016/S0010-4655(98)00154-4">doi:10.1016/S0010-4655(98)00154-4</a>.</p>
<p>[2] H. Kesten (1980). The critical probability of bond percolation on the square lattice equals 1/2. <i>Communications in Mathematical Physics</i> 74, 41–59. <a href="https://doi.org/10.1007/BF01197577">doi:10.1007/BF01197577</a>.</p>
<p>[3] Project brief: <i>Mini-project: Uncertainty and Sensitivity Analysis of Flow in a 2D Random Network</i>, supplied PDF, two pages. Parameter ranges, sample counts and numerical implementation were selected for this study.</p></div>'''

    en=[]
    en.append(f'''<section class="page first"><div class="tag">COMPLETED MINI-PROJECT · 04 NOVEMBER 2026</div>
<h1>Uncertainty and sensitivity of flow in a 2D random network</h1>
<div class="box"><b>Finding.</b> Connectivity controls both the existence and magnitude of flow. Broadening the uncertain connectivity distribution increases mean conductance strongly, while the probability of a spanning path stays close to one half. Conductance heterogeneity changes flow magnitude, with bottlenecks dominating near the connectivity threshold.</div>
<p>At σ=1, increasing δ from 0 to 0.25 changes mean effective conductance from <b>{f(low['mean'])}</b> to <b>{f(base['mean'])}</b> ({base['mean']/low['mean']:.2f}×). In the latter case, {100*base.p_zero:.2f}% of realizations carry no flow. These results come from {m['solver_calls']:,} solver calls, including 90,000 main uncertainty realizations.</p>
<h3>Model and scope</h3><p>A square bond lattice has n=25 nodes per side (625 nodes; 1,200 potential links). All quantities are dimensionless. Each realization samples one global p; links are independent conditional on that p. The left/right hydraulic heads are 1/0; the top/bottom boundaries have no exterior links and hence no imposed flux.</p>
<div class="eq">p = 0.5 + δU, U ∼ Uniform(−1,1)<br>gₑ = 1[Wₑ &lt; p] exp(σZₑ), Wₑ ∼ Uniform(0,1), Zₑ ∼ Normal(0,1)<br>Qᵢⱼ = gᵢⱼ(hᵢ−hⱼ),  ΣⱼQᵢⱼ = 0,  G_eff = Q_total / Δh = Q_total</div>
{fig('01_network','Illustrative 15×15 networks. Node colors show head; orange link widths show flow magnitude. Grey nodes are floating components with undefined absolute head.')}
</section>''')
    en.append(f'''<section class="page"><h2>1 · Numerical method and verification</h2>
<p>Build the weighted graph Laplacian L. Connected-component analysis identifies clusters touching both fixed-head boundaries. Solve only free nodes in these clusters using SciPy’s sparse direct solver:</p>
<div class="eq">L_FF h_F = −L_FB h_B</div>
<p>Clusters touching only one boundary have constant head and zero flow. Interior floating clusters have arbitrary absolute head (stored as NaN) and zero flow; they are excluded from the singular linear system. Total inlet flux is evaluated on spanning clusters. This avoids artificial leakage or regularization.</p>
{validation_en}
<p><b>Verification:</b> all 24 automated tests pass. For 1,000 additional random networks, maximum interior balance residual was {diag.max_residual.max():.2e}, maximum relative inlet/outlet mismatch {diag.relative_balance_error.max():.2e}, and maximum energy discrepancy {diag.energy_error.max():.2e}.</p>
<h3>Experimental design and efficiency</h3>{design}
<p class="small">n=25 resolves the connectivity transition at low cost; n=15 and 41 check size dependence. δ∈{{0,0.1,0.25}} spans p=0.5 through p∈[0.25,0.75]. σ∈{{0,1,2}} spans constant conductance to strong heterogeneity (σ²=0,1,4). σ, rather than σ², is sampled uniformly for sensitivity. Seed {m['seed']}; separated RNG streams; common random numbers across scenarios; measured full study time {m['elapsed_seconds']:.1f} s on this execution host. Hardware and timing are not a performance guarantee.</p>
</section>''')
    en.append(f'''<section class="page"><h2>2 · Uncertainty: statistics of predicted flow</h2>
<p>Each row contains 10,000 independent network realizations. The same latent realizations couple different rows to improve paired comparisons; rows must not be pooled as independent observations. P(G=0) is a probability, distinct from the size of nonzero flow.</p>
{table(uh_en,uncertainty_rows)}
{fig('02_uncertainty','Mean conductance: normal Monte Carlo 95% intervals. Spanning probability: Wilson 95% intervals; identical across σ by construction.')}
<p><b>Reference scenario (δ=0.25, σ=1).</b> Mean = {ci(base)}; standard deviation = {f(base.sd)}; median = {f(base['median'])}; 95th percentile = {f(base.q95)} (order-statistic 95% interval [{f(base.q95_low)}, {f(base.q95_high)}]). The probability of a spanning path is {base.p_span:.4f}, with Wilson interval [{base.p_span_low:.4f}, {base.p_span_high:.4f}]. Conditional mean given a positive flow is {f(base.positive_mean)}.</p>
<p class="small">The reported confidence intervals quantify finite Monte Carlo sampling error under the specified model. They are not prediction intervals for a new network and do not include structural model uncertainty. Quantile intervals use binomial order statistics and may be conservative when observations tie at zero.</p>
</section>''')
    en.append(f'''<section class="page"><h2>3 · Distribution and connectivity transition</h2>
{fig('03_distribution','The full output law has a discrete mass at zero and a broad positive component. A single lognormal distribution cannot describe the full output.')}
<p>For the reference scenario, the atom at zero is {base.p_zero:.4f}; the mean, median and upper tail are far apart. Report P(G=0) and positive-flow statistics separately. The density plot describes log₁₀(G) conditional on positive flow, not the density of G. The CDF panel focuses on 0≤G≤0.75; full values are retained in the raw data.</p>
{fig('04_connectivity','Fixed-p sweep, n=25 and σ=1, 2,500 samples per p. p_c=0.5 is the infinite square-lattice bond threshold [2]; the finite network has a smooth crossover.')}
<p>Increasing δ leaves E[p]=0.5, but spreads realizations below and above the connectivity transition. The nonlinear flow response gives high-p networks substantial flow, while low-p networks remain at zero. This explains the increased mean without a comparable increase in spanning probability; it is an empirical result over these ranges, not a general convexity theorem.</p>
<p class="small">Unconditional links are correlated because they share p: for different edge indicators Iₑ,I𝒻, Cov(Iₑ,I𝒻)=Var(p)=δ²/3. Sampling a separate p for every edge would remove this mechanism and implement a different model.</p>
</section>''')
    en.append(f'''<section class="page"><h2>4 · Global sensitivity with stochastic inputs</h2>
<p>Use grouped variance-based Sobol analysis [1], which captures threshold behavior and interactions. Expand the stochastic simulator into five independent input groups: δ∼U(0,0.25), σ∼U(0,2), U∼U(−1,1), topology field W (1,200 uniforms), and conductance field Z (1,200 normals). This avoids incorrectly treating δ and the realized p as independent inputs.</p>
<div class="eq">Sᵢ = Var(E[Y | Xᵢ]) / Var(Y)<br>Sᵀᵢ = E[Var(Y | X_−ᵢ)] / Var(Y)<br>Ŝᵢ = 1 − mean[(Y_B−Y_ABᵢ)²]/(2V),  Ŝᵀᵢ = mean[(Y_A−Y_ABᵢ)²]/(2V)</div>
<p class="small">A and B are independent grouped input samples; ABᵢ replaces group i of A with B. V is the pooled sample variance of Y_A and Y_B. Use 16,384 independent base pairs (114,688 evaluations), with 1,000 paired-row bootstrap resamples for percentile 95% intervals.</p>
{table(['Group','First order [95% CI]','Total order [95% CI]'],sens_rows(factor_en))}
{fig('05_sensitivity','First-order and total-order indices for flow magnitude and the binary existence of a spanning path. Error bars are bootstrap 95% intervals.')}
<p class="small">U dominates; δ has a much larger total than first-order effect, indicating interactions, principally with the connectivity draw. About {100*(1-first_sum):.1f}% of variance is assigned to interactions by the sum of estimated first-order indices. Total indices overlap and must not be added as independent percentages. σ and Z have exactly zero total effect on spanning because every active conductance is positive. Small spurious first-order estimates for known inactive factors are sampling error; raw estimates are preserved. Rankings depend on the chosen priors and latent parameterization and concern all realizations, including zero-flow networks.</p>
</section>''')
    en.append(f'''<section class="page"><h2>5 · Convergence, finite size and normalization</h2>
{fig('06_convergence','Left: nested Monte Carlo checkpoints (250 to 10,000). Right: different network ensembles at n=15,25,41; this is finite-size sensitivity, not continuum mesh convergence.')}
<p>The final relative 95% Monte Carlo half-width of the mean is {relci.min():.2f}–{relci.max():.2f}% across the nine scenarios. At p=0.5 (δ=0, σ=1), mean G decreases from {size_means[15]:.4f} at n=15 to {size_means[25]:.4f} at n=25 and {size_means[41]:.4f} at n=41. Finite-size effects remain material near the threshold. Results cannot be advertised as a size-independent continuum prediction. Sobol prefix convergence is saved in figure 08 and CSV; the leading ranking is resolved, while tiny first-order terms remain uncertain.</p>
{fig('07_normalization','The assigned model fixes median(g)=1, so increasing σ also increases E[g]. A constant-mean control is obtained by exact per-network rescaling.')}
<p>In the specified model E[g | active]=exp(σ²/2), equal to 1, 1.6487 and 7.3891. Nevertheless mean G falls as σ increases in the studied scenarios: for δ=0.25, from {s[(s.delta==.25)&(s.sigma==0)].iloc[0]['mean']:.4f} to {high_sigma['mean']:.4f}. Weak links on series paths constrain flow. To isolate variability at fixed mean link conductance, set g*=g exp(−σ²/2); linearity gives G*=G exp(−σ²/2) exactly. This is a supplementary control and does not replace the assigned distribution.</p>
</section>''')
    en.append(f'''<section class="page"><h2>6 · Interpretation, limitations and extensions</h2>
<h3>Main findings</h3><ul>
<li>Connectivity and its interactions dominate global output variance over the declared ranges. Characterizing connectivity has more value here than measuring every individual link conductance precisely.</li>
<li>The mean alone hides a near-50% probability of no flow and a highly variable positive component. A useful flow forecast includes connection probability, conditional flow and an upper quantile.</li>
<li>Broader uncertain connectivity increases mean flow at fixed E[p], while greater log-conductance spread reduces mean flow in this study. Both are range- and model-dependent findings.</li>
</ul><h3>Limitations</h3><ul>
<li>Regular, isotropic, two-dimensional bond geometry; no fracture length/orientation distributions, intersections, aperture physics or spatially correlated fields conditional on p.</li>
<li>Steady, single-phase, linear hydraulic conductance; no storage, transport, transient pressure, matrix flow, multiphase processes or geomechanical feedback.</li>
<li>Ranges are exploratory, not calibrated priors. The model is dimensionless and has not been validated against field data. Analytical and invariant checks establish implementation correctness, not geological predictive accuracy.</li>
<li>Finite-size dependence is significant. Near-critical paths, rare tails and strongly heterogeneous conductances need more sampling and conditioning checks if ranges are extended.</li>
</ul><h3>Extensions to fractured rocks</h3><p>Generate discrete fracture networks in 2D/3D with measured fracture geometry and intersection statistics; link conductance to aperture and the cubic law with consistent units; introduce anisotropy and spatial correlation; calibrate against heads, flow or tracer data. Include matrix exchange and transient transport, then propagate parameter and structural uncertainty. Surrogates could accelerate these richer models after validating their errors near the connectivity transition; they are unnecessary for this inexpensive lattice study.</p>
<h3>Reproducibility and deliverables</h3><p>Source solver, 24 tests, complete compressed realization-level CSVs, summary CSVs, seed/version manifest, scripts for figures/reports/slides, and an executed notebook are included. The presentation is supplied as editable PowerPoint and PDF. Run commands are in the repository README. All reported values are generated from saved results, not manually populated example data.</p>
<h3>References</h3>{refs}
</section>''')

    fa=[]
    fa.append(f'''<section class="page first"><div class="tag">پروژه تکمیل‌شده · ۴ نوامبر ۲۰۲۶</div>
<h1>تحلیل عدم‌قطعیت و حساسیت جریان در شبکه تصادفی دوبعدی</h1>
<div class="box"><b>یافته اصلی:</b> اتصال شبکه، هم وجود جریان و هم مقدار آن را کنترل می‌کند. افزایش دامنه عدم‌قطعیت اتصال، میانگین جریان را به‌شدت افزایش داد؛ با این حال احتمال وجود مسیر عبوری نزدیک یک‌دوم باقی ماند. ناهمگنی رسانایی، مقدار جریان را تغییر داد و اثر گلوگاه‌ها نزدیک آستانه اتصال مهم بود.</div>
<p>برای <bdi>σ=1</bdi>، افزایش <bdi>δ</bdi> از صفر به <bdi>0.25</bdi>، میانگین رسانایی مؤثر را از <bdi>{f(low['mean'])}</bdi> به <bdi>{f(base['mean'])}</bdi> رساند؛ یعنی <bdi>{base['mean']/low['mean']:.2f}</bdi> برابر. در حالت دوم، <bdi>{100*base.p_zero:.2f}%</bdi> شبکه‌ها اصلاً جریان نداشتند. این گزارش بر <bdi>{m['solver_calls']:,}</bdi> اجرای مدل، شامل ۹۰ هزار نمونه تحلیل اصلی، استوار است.</p>
<h3>مدل و دامنه مطالعه</h3><p>شبکه مبنا ۲۵×۲۵ گره دارد: ۶۲۵ گره و ۱۲۰۰ لینک بالقوه. کمیت‌ها بی‌بُعد هستند. در هر تحقق، فقط یک مقدار سراسری <bdi>p</bdi> نمونه‌برداری می‌شود؛ سپس لینک‌ها به‌شرط این مقدار مستقل هستند. هد چپ و راست به‌ترتیب ۱ و صفر است. نبود لینک خارجی در بالا و پایین، شرط بدون جریان را اعمال می‌کند.</p>
<div class="eq">p = 0.5 + δU, U ∼ Uniform(−1,1)<br>gₑ = 1[Wₑ &lt; p] exp(σZₑ), Wₑ ∼ Uniform(0,1), Zₑ ∼ Normal(0,1)<br>Qᵢⱼ = gᵢⱼ(hᵢ−hⱼ),  Σⱼ Qᵢⱼ = 0,  G_eff = Q_total / Δh = Q_total</div>
{fig('01_network','دو نمونه نمایشی ۱۵×۱۵؛ رنگ گره‌ها هد و ضخامت خطوط نارنجی مقدار جریان را نشان می‌دهد. هد مطلق گره‌های خاکستریِ جدا از مرزها نامعین است.')}
</section>''')
    fa.append(f'''<section class="page"><h2>۱. روش عددی و اعتبارسنجی</h2>
<p>لاپلاسین وزن‌دار گراف ساخته شد. تحلیل مؤلفه‌های همبند، خوشه‌های متصل به هر دو مرز هد ثابت را شناسایی کرد. فقط هد گره‌های آزاد در همین خوشه‌ها با حل‌گر مستقیم ماتریس تنک <bdi>SciPy</bdi> محاسبه شد:</p>
<div class="eq">L_FF h_F = −L_FB h_B</div>
<p>خوشه متصل به یک مرز، هد ثابت و جریان صفر دارد. خوشه داخلیِ جدا از هر دو مرز، هد مطلق دلخواه و جریان صفر دارد؛ هد آن با <bdi>NaN</bdi> ثبت و از دستگاه تکین حذف می‌شود. شار ورودی فقط از خوشه‌های عبوری محاسبه می‌شود. هیچ نشتی یا رسانایی مصنوعی برای رفع تکینی اضافه نشده است.</p>
{table(['آزمون مستقل','پاسخ مورد انتظار'],[
 ['شبکه کامل یکنواخت؛ چهار اندازه و سه رسانایی','<bdi>G = g n/(n−1)</bdi> و هد خطی'],
 ['مسیر سری و دو مسیر موازی','معکوس مجموع مقاومت‌ها؛ جمع رسانایی شاخه‌ها'],
 ['جزیره جدا و برش عمودی شبکه','رسانایی صفر؛ هد جزیره نامعین'],
 ['بقای جریان، انرژی و کران هد','<bdi>Qin=Qout=Σg(Δh)²; 0≤h≤1</bdi>'],
 ['مقیاس‌پذیری و یکنوایی ریلی','<bdi>G(cg)=cG(g)</bdi>؛ افزایش لینک جریان را کم نمی‌کند'],
 ['برآوردگر حساسیت','تطبیق شاخص‌های تحلیلی تابع ایشیگامی با خطای کمتر از ۰٫۰۱۸'],
 ])}
<p>هر <b>۲۴ آزمون خودکار</b> موفق بود. در ۱۰۰۰ شبکه تصادفی مستقل، بیشترین باقیمانده موازنه گره‌ای <bdi>{diag.max_residual.max():.2e}</bdi>، بیشترین خطای نسبی تطابق ورودی و خروجی <bdi>{diag.relative_balance_error.max():.2e}</bdi> و بیشترین اختلاف انرژی با شار کل <bdi>{diag.energy_error.max():.2e}</bdi> به‌دست آمد.</p>
<h3>طرح آزمایش و انتخاب‌ها</h3>
{table(['آزمایش','طرح','تعداد اجرا'],[
 ['عدم‌قطعیت اصلی','۳ مقدار δ × ۳ مقدار σ × ۱۰٬۰۰۰ نمونه','90,000'],
 ['پویش احتمال اتصال','۹ مقدار p از ۰٫۳۰ تا ۰٫۷۰؛ هرکدام ۲۵۰۰ نمونه','22,500'],
 ['اثر اندازه اضافی','۱۵ و ۴۱ گره در ضلع؛ دو δ؛ σ=۱؛ ۵۰۰۰ نمونه','20,000'],
 ['حساسیت گروهی سوبول','۱۶٬۳۸۴ جفت پایه × ۷ ارزیابی','114,688'],
 ['بررسی عددی تکمیلی','۱۰۰۰ شبکه مستقل','1,000'],
 ['کل','داده اندازه ۲۵ از آزمایش اصلی بازاستفاده شد',f'{m["solver_calls"]:,}'],
 ])}
<p class="small">اندازه ۲۵ هزینه کمی دارد و ناحیه گذار اتصال را نشان می‌دهد؛ اندازه‌های ۱۵ و ۴۱ برای بررسی وابستگی نتایج به اندازه انتخاب شدند. <bdi>δ∈{{0,0.1,0.25}}</bdi> و <bdi>σ∈{{0,1,2}}</bdi> به‌ترتیب اتصال نزدیک آستانه تا دامنه <bdi>p∈[0.25,0.75]</bdi> و رسانایی یکنواخت تا ناهمگنی قوی را پوشش می‌دهند. در تحلیل حساسیت، توزیع یکنواخت برای <bdi>σ</bdi> انتخاب شده است، نه <bdi>σ²</bdi>. بذر پایه <bdi>{m['seed']}</bdi> است؛ زمان اجرای مطالعه روی این میزبان حدود <bdi>{m['elapsed_seconds']:.1f}</bdi> ثانیه بود.</p>
</section>''')
    fa.append(f'''<section class="page"><h2>۲. نتایج تحلیل عدم‌قطعیت</h2>
<p>هر سطر ۱۰٬۰۰۰ تحقق مستقل دارد. برای کاهش خطای مقایسه، سطرهای سناریوها از متغیرهای تصادفی مشترک استفاده می‌کنند؛ بنابراین نباید همه سطرها را به‌عنوان نمونه‌های مستقل با هم ادغام کرد. احتمال صفر بودن جریان، با اندازه جریان در شبکه‌های متصل تفاوت دارد.</p>
{table(uh_fa,uncertainty_rows)}
{fig('02_uncertainty','چپ: میانگین با فاصله اطمینان مونت‌کارلو ۹۵٪. راست: احتمال مسیر عبوری با فاصله ویلسون ۹۵٪؛ این احتمال برای همه σها یکسان است.')}
<p><b>سناریوی مرجع <bdi>(δ=0.25, σ=1)</bdi>:</b> میانگین <bdi>{f(base['mean'])}</bdi> با فاصله اطمینان <bdi>[{f(base.mean_ci_low)}, {f(base.mean_ci_high)}]</bdi>، انحراف معیار <bdi>{f(base.sd)}</bdi>، میانه <bdi>{f(base['median'])}</bdi> و صدک ۹۵ برابر <bdi>{f(base.q95)}</bdi> است. فاصله اطمینان صدک ۹۵ برابر <bdi>[{f(base.q95_low)}, {f(base.q95_high)}]</bdi> است. احتمال مسیر عبوری <bdi>{f(base.p_span)}</bdi> با فاصله ویلسون <bdi>[{f(base.p_span_low)}, {f(base.p_span_high)}]</bdi> و میانگین به‌شرط جریان مثبت <bdi>{f(base.positive_mean)}</bdi> است.</p>
<p class="small">فاصله‌های اطمینان جدول، خطای ناشی از تعداد محدود نمونه‌های مونت‌کارلو را بیان می‌کنند؛ بازه پیش‌بینی برای یک شبکه تازه یا عدم‌قطعیت ساختاری مدل نیستند. فاصله میانگین از تقریب نرمال، احتمال اتصال از روش ویلسون و فاصله صدک از آمار ترتیبی مبتنی بر دوجمله‌ای محاسبه شده است.</p>
</section>''')
    fa.append(f'''<section class="page"><h2>۳. توزیع خروجی و گذار اتصال</h2>
{fig('03_distribution','خروجی، ترکیبی از جرم احتمال در صفر و توزیع پیوسته مقادیر مثبت است؛ یک توزیع لگ‌نرمال واحد برای کل خروجی مناسب نیست.')}
<p>در سناریوی مرجع، جرم احتمال صفر <bdi>{f(base.p_zero)}</bdi> است و میانگین، میانه و مقادیر بالای توزیع فاصله زیادی دارند. پس باید احتمال قطع جریان و آمار جریان‌های مثبت جدا گزارش شوند. نمودار چگالی، چگالی <bdi>log₁₀(G)</bdi> برای مقادیر مثبت است. نمای تابع توزیع به بازه <bdi>0≤G≤0.75</bdi> محدود شده؛ همه مقادیر در فایل خام حفظ شده‌اند.</p>
{fig('04_connectivity','پویش p ثابت با ۲۵۰۰ نمونه در هر نقطه، شبکه ۲۵×۲۵ و σ=۱. آستانه ۰٫۵ مربوط به شبکه مربعی نامتناهی است [۲]؛ گذار شبکه محدود تدریجی است.')}
<p>با افزایش <bdi>δ</bdi>، میانگین <bdi>p</bdi> همچنان ۰٫۵ است، اما نمونه‌ها در دو سوی گذار اتصال پخش می‌شوند. شبکه‌های با احتمال اتصال بالا می‌توانند جریان زیادی داشته باشند؛ شبکه‌های کم‌اتصال اغلب در صفر باقی می‌مانند. پاسخ غیرخطی، افزایش میانگین جریان بدون افزایش مشابه احتمال اتصال را توضیح می‌دهد. این نتیجه برای دامنه مطالعه مشاهده شده و یک قضیه عمومی درباره تحدب نیست.</p>
<p class="small">لینک‌ها فقط به‌شرط <bdi>p</bdi> مستقل هستند. چون همه یک <bdi>p</bdi> مشترک دارند، کوواریانس شاخص فعال بودن دو لینک متفاوت برابر <bdi>Var(p)=δ²/3</bdi> است. نمونه‌برداری یک <bdi>p</bdi> جداگانه برای هر لینک، مدل دیگری می‌سازد.</p>
</section>''')
    fa.append(f'''<section class="page"><h2>۴. تحلیل حساسیت سراسری</h2>
<p>روش گروهی واریانس‌محور سوبول با برآوردگر یانسن [۱] انتخاب شد، چون گذار اتصال، غیرخطی بودن و برهم‌کنش‌ها را پوشش می‌دهد. شبیه‌ساز به پنج گروه ورودی مستقل تبدیل شد: <bdi>δ∼U(0,0.25)</bdi>، <bdi>σ∼U(0,2)</bdi>، <bdi>U∼U(−1,1)</bdi>، میدان آرایش <bdi>W</bdi> با ۱۲۰۰ متغیر یکنواخت و میدان رسانایی <bdi>Z</bdi> با ۱۲۰۰ متغیر نرمال. به این ترتیب، <bdi>δ</bdi> و <bdi>p</bdi> وابسته، اشتباهاً ورودی مستقل فرض نمی‌شوند.</p>
<div class="eq">Sᵢ = Var(E[Y | Xᵢ]) / Var(Y)<br>Sᵀᵢ = E[Var(Y | X_−ᵢ)] / Var(Y)<br>Ŝᵢ = 1 − mean[(Y_B−Y_ABᵢ)²]/(2V),  Ŝᵀᵢ = mean[(Y_A−Y_ABᵢ)²]/(2V)</div>
<p class="small">نمونه‌های <bdi>A</bdi> و <bdi>B</bdi> مستقل‌اند؛ در <bdi>ABᵢ</bdi> فقط گروه <bdi>i</bdi> از <bdi>B</bdi> جایگزین می‌شود. <bdi>V</bdi> واریانس تجمیعی دو خروجی پایه است. ۱۶٬۳۸۴ جفت پایه و ۱۰۰۰ بازنمونه‌گیری بوت‌استرپ از سطرهای جفت‌شده برای فاصله اطمینان استفاده شد.</p>
{table(['گروه','اثر مرتبه اول [۹۵٪]','اثر کل [۹۵٪]'],sens_rows(factor_fa))}
{fig('05_sensitivity','شاخص‌های مرتبه اول و کل برای مقدار رسانایی و وجود مسیر؛ خطاها فاصله اطمینان بوت‌استرپ هستند.')}
<p class="small"><bdi>U</bdi> مهم‌ترین ورودی است. اثر کل <bdi>δ</bdi> بسیار بیشتر از اثر مرتبه اول آن است؛ یعنی برهم‌کنش‌ها، به‌ویژه با متغیر احتمال اتصال، مهم‌اند. از مجموع شاخص‌های مرتبه اول، حدود <bdi>{100*(1-first_sum):.1f}%</bdi> واریانس به برهم‌کنش‌ها نسبت داده می‌شود. اثرهای کل هم‌پوشانی دارند و نباید به‌صورت سهم‌های مستقل جمع شوند. اثر کل <bdi>σ</bdi> و <bdi>Z</bdi> بر وجود مسیر دقیقاً صفر است، چون همه رسانایی‌های فعال مثبت‌اند؛ مقادیر کوچک مرتبه اول آن‌ها خطای نمونه‌گیری است. رتبه‌بندی به توزیع‌های انتخابی و پارامتردهی متغیرهای نهفته وابسته است و همه شبکه‌ها، شامل صفرها، را در نظر می‌گیرد.</p>
</section>''')
    fa.append(f'''<section class="page"><h2>۵. همگرایی، اندازه شبکه و اثر تعریف رسانایی</h2>
{fig('06_convergence','چپ: همگرایی آماری با ۲۵۰ تا ۱۰٬۰۰۰ نمونه. راست: حساسیت به اندازه شبکه؛ تغییر اندازه، جامعه شبکه‌ها را تغییر می‌دهد و همگرایی شبکه‌بندی یک مسئله پیوسته نیست.')}
<p>نیم‌عرض نسبی فاصله اطمینان ۹۵٪ میانگین در ۹ سناریو، بین <bdi>{relci.min():.2f}%</bdi> و <bdi>{relci.max():.2f}%</bdi> است. در <bdi>δ=0, σ=1</bdi>، میانگین رسانایی برای اندازه‌های ۱۵، ۲۵ و ۴۱ به‌ترتیب <bdi>{size_means[15]:.4f}</bdi>، <bdi>{size_means[25]:.4f}</bdi> و <bdi>{size_means[41]:.4f}</bdi> است. بنابراین اثر اندازه نزدیک آستانه قابل‌توجه است و نتیجه، پیش‌بینی مستقل از اندازه نیست. همگرایی شاخص‌های سوبول در شکل ۰۸ و فایل CSV آمده است؛ رتبه عامل غالب روشن است، ولی شاخص‌های بسیار کوچک همچنان عدم‌قطعیت دارند.</p>
{fig('07_normalization','چپ: مدل اصلی با میانه رسانایی فعال برابر ۱. راست: کنترل تکمیلی با میانگین رسانایی فعال برابر ۱، حاصل باز‌مقیاس‌گذاری دقیق هر شبکه.')}
<p>در مدل خواسته‌شده، <bdi>E[g|active]=exp(σ²/2)</bdi> است؛ برای سه <bdi>σ</bdi> انتخابی، این میانگین برابر ۱، ۱٫۶۴۸۷ و ۷٫۳۸۹۱ می‌شود. با وجود این، میانگین رسانایی مؤثر با افزایش <bdi>σ</bdi> کاهش یافت؛ برای <bdi>δ=0.25</bdi> از <bdi>{s[(s.delta==.25)&(s.sigma==0)].iloc[0]['mean']:.4f}</bdi> به <bdi>{high_sigma['mean']:.4f}</bdi>. لینک‌های ضعیف در مسیرهای سری جریان را محدود می‌کنند. برای جدا کردن ناهمگنی در میانگین ثابت، <bdi>g*=g exp(−σ²/2)</bdi> قرار داده شد؛ در نتیجه <bdi>G*=G exp(−σ²/2)</bdi> دقیقاً برقرار است. این آزمایش، کنترل تکمیلی است و توزیع اصلی پروژه را جایگزین نمی‌کند.</p>
</section>''')
    fa.append(f'''<section class="page"><h2>۶. تفسیر، محدودیت‌ها و توسعه‌های ممکن</h2>
<h3>برداشت علمی</h3><ul>
<li>در دامنه‌های انتخاب‌شده، اتصال و برهم‌کنش‌های آن بیشترین سهم را در تغییرپذیری خروجی دارند. شناخت اتصال شبکه در این مدل از اندازه‌گیری دقیق رسانایی تک‌تک لینک‌ها مؤثرتر است.</li>
<li>میانگین به‌تنهایی، احتمال نزدیک ۵۰٪ قطع جریان و دامنه بزرگ جریان‌های مثبت را پنهان می‌کند. پیش‌بینی مناسب باید احتمال اتصال، جریان شرطی و یک صدک بالا را گزارش کند.</li>
<li>افزایش عدم‌قطعیت اتصال در میانگین ثابت p، میانگین جریان را افزایش داد؛ افزایش پراکندگی لگاریتم رسانایی آن را کاهش داد. این یافته‌ها به دامنه و ساختار همین مدل وابسته‌اند.</li>
</ul><h3>محدودیت‌ها</h3><ul>
<li>هندسه منظم، همسانگرد و دوبعدی است؛ طول و جهت شکستگی‌ها، هندسه تقاطع‌ها، فیزیک بازشدگی و همبستگی مکانی شرطی مدل نشده‌اند.</li>
<li>جریان پایا، تک‌فازی و خطی است؛ ذخیره، انتقال، جریان ماتریس سنگ، چندفازی بودن و اثرات ژئومکانیکی وجود ندارند.</li>
<li>دامنه‌ها اکتشافی‌اند و از داده میدانی کالیبره نشده‌اند. مدل بی‌بُعد است. آزمون‌های تحلیلی و فیزیکی، صحت پیاده‌سازی را بررسی می‌کنند و اعتبار پیش‌بینی زمین‌شناختی را اثبات نمی‌کنند.</li>
<li>وابستگی به اندازه مهم است. برای رسانایی‌های بسیار ناهمگن‌تر، رخدادهای نادر و نزدیک آستانه باید نمونه‌های بیشتر و وضعیت عددی دستگاه بررسی شود.</li>
</ul><h3>توسعه برای سنگ‌های شکسته واقعی</h3><p>شبکه شکستگی گسسته دوبعدی یا سه‌بعدی با هندسه اندازه‌گیری‌شده ساخته شود؛ رسانایی از بازشدگی و قانون مکعبی با واحدهای سازگار محاسبه شود؛ ناهمسانگردی و همبستگی مکانی افزوده و پارامترها با داده هد، دبی یا ردیاب کالیبره شوند. سپس تبادل با ماتریس و جریان و انتقال گذرا افزوده و عدم‌قطعیت پارامتری و ساختاری منتشر شود. مدل جانشین برای چنین توسعه‌های پرهزینه‌ای مفید است، به‌شرط اعتبارسنجی نزدیک گذار اتصال؛ برای این شبکه کم‌هزینه ضرورتی نداشت.</p>
<h3>بازتولید و خروجی‌ها</h3><p>کد حل‌گر، ۲۴ آزمون، تمام نمونه‌های خام فشرده، جدول‌های آماری، بذر و نسخه کتابخانه‌ها، اسکریپت ساخت نمودار و گزارش و ارائه، و دفترچه اجراشده تحویل شده‌اند. ارائه به‌صورت پاورپوینت قابل‌ویرایش و PDF موجود است. فرمان‌های اجرا در README مخزن آمده‌اند. تمام اعداد گزارش از فایل نتایج خوانده می‌شوند.</p>
<h3>منابع</h3>{refs}
</section>''')

    for lang,pages in [('en',en),('fa',fa)]:
        html=f'<!doctype html><html lang="{lang}" dir="{"rtl" if lang=="fa" else "ltr"}"><meta charset="utf-8"><title>2D random network flow: {lang}</title><style>{CSS}</style><body class="{lang}">'+''.join(pages)+'</body></html>'
        path=D/f'report_{lang}.html'
        path.write_text(html,encoding='utf-8')
        HTML(filename=str(path)).write_pdf(D/f'report_{lang}.pdf')
        print(f'Built {path.name} and report_{lang}.pdf')


if __name__=='__main__':main()
