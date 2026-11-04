#!/usr/bin/env python3
"""Build bilingual comparison reports with all numerical statements data-driven."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from weasyprint import HTML

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.make_reports import CSS,table,fig

B=ROOT/'parametric_pinn';R=B/'results';D=B/'reports'


def main():
    train=json.loads((R/'training_manifest.json').read_text())
    evaluation=json.loads((R/'evaluation_manifest.json').read_text())
    summary=pd.read_csv(R/'test_summary.csv')
    scenarios=pd.read_csv(R/'scenario_comparison.csv')
    history=pd.read_csv(R/'training_history.csv')
    depths=pd.read_csv(R/'validation_depths.csv')
    sensitivity=pd.read_csv(R/'sobol_comparison.csv')
    timing=pd.read_csv(R/'timing_benchmark.csv')
    params_only=pd.read_csv(R/'physical_parameters_only_baseline.csv').iloc[0]
    difficult=json.loads((R/'difficult_example.json').read_text())
    original=pd.read_csv(ROOT/'results/scenario_summary.csv')
    def row(split,method='pinn_accurate'):
        return summary[(summary.split==split)&(summary.method==method)].iloc[0]
    def pct(x,d=4):return f'{100*x:.{d}f}%'
    def num(x,d=5):return f'{x:.{d}f}'
    a=row('scenario');c=row('continuous');fast=row('scenario','pinn_fast');untrained=row('scenario','untrained_control')
    ood=row('ood_sigma3')
    tm=timing.groupby(['method','batch_size']).median(numeric_only=True).reset_index()
    fastest=tm.loc[tm.milliseconds_per_case.idxmin()]
    ref_ms=float(np.median(timing.sparse_g_seconds/timing.cases)*1000)
    ratio=fastest.milliseconds_per_case/ref_ms
    cost_en=f'The fastest measured PINN setting is {ratio:.2f}× the sparse G-only latency; no positive break-even is observed.' if ratio>=1 else f'The fastest batched PINN is {1/ratio:.2f}× faster in this benchmark; training amortization is in the timing CSV.'
    cost_fa=f'سریع‌ترین تنظیم PINN در این آزمون {ratio:.2f} برابر زمان حل‌گر مستقیمِ فقط رسانایی هزینه داشت؛ نقطه سربه‌سر مثبتی مشاهده نشد.' if ratio>=1 else f'سریع‌ترین تنظیم دسته‌ای PINN در این آزمون {1/ratio:.2f} برابر سریع‌تر بود؛ هزینه سربه‌سر آموزش در CSV آمده است.'
    comparison_en=table(['Property','Direct sparse method','Parametric graph PINN'],[
        ['Equations','Kirchhoff conservation','Same graph physics; energy loss'],
        ['Inputs','Full edge conductance field, heads','Full field + δ,p,σ + heads'],
        ['Offline fitting','None',f'{train["train_data"]["retained_spanning"]:,} spanning training fields; no solution labels'],
        ['New input','New sparse solve','Graph preprocessing + fixed K neural layers'],
        ['Effective conductance','Inlet flux / Δh','Energy-derived upper-bound estimate'],
        ['Conservation','Linear-solver accuracy','Approximate; tested separately'],
        ['Boundary / disconnected graph','Exact','Exact structural constraints'],
    ])
    comparison_fa=table(['ویژگی','حل مستقیم تنک','PINN پارامتری گرافی'],[
        ['معادلات','بقای جریان کیرشهف','همان فیزیک گراف؛ تابع هزینه انرژی'],
        ['ورودی','کل میدان رسانایی و هد مرزها','کل میدان، δ و p و σ، هد مرزها'],
        ['آموزش اولیه','ندارد','۴۰۹۶ شبکه عبوری؛ بدون برچسب جواب'],
        ['ورودی تازه','حل دستگاه تنک جدید','پیش‌پردازش گراف و K لایه ثابت'],
        ['رسانایی مؤثر','شار ورودی بر اختلاف هد','تخمین کران بالای مبتنی بر انرژی'],
        ['بقای جریان','تا دقت حل‌گر خطی','تقریبی؛ با آزمون جدا سنجیده شد'],
        ['شرط مرزی و جریان صفر','دقیق','با قید ساختاری دقیق'],
    ])
    methods=['geodesic','untrained_control','pinn_fast','pinn_accurate']
    names_en={'geodesic':'Geodesic only','untrained_control':'Untrained, K=256','pinn_fast':'Trained, K=256','pinn_accurate':'Trained, K=2048'}
    names_fa={'geodesic':'فقط فاصله مقاومتی','untrained_control':'بدون آموزش؛ ۲۵۶ لایه','pinn_fast':'آموزش‌دیده؛ ۲۵۶ لایه','pinn_accurate':'آموزش‌دیده؛ ۲۰۴۸ لایه'}
    def error_rows(names):
        return [[names[m],pct(row('scenario',m).positive_mean_abs_relative_error),
            pct(row('scenario',m).positive_p95_abs_relative_error),pct(row('scenario',m).positive_max_abs_relative_error),
            num(row('scenario',m).mean_free_node_rmse,6)] for m in methods]
    def uq_rows():
        rows=[]
        for _,r in scenarios[scenarios.method=='pinn_accurate'].sort_values(['delta','sigma']).iterrows():
            old=original[(original.delta==r.delta)&(original.sigma==r.sigma)].iloc[0]['mean']
            rows.append([num(r.delta,2),num(r.sigma,0),num(old),num(r.mean_reference),num(r.mean_predicted),pct(r.mean_relative_bias,5),num(r.p_zero_predicted,3)])
        return rows
    sob=sensitivity[(sensitivity.method=='pinn_accurate')&(sensitivity.order=='total')]
    sobref=sensitivity[(sensitivity.method=='reference')&(sensitivity.order=='total')].set_index('factor')
    labels_en={'delta':'δ','sigma':'σ','p_latent':'Global p latent U','topology':'Topology field W','conductance_field':'Conductance field Z'}
    labels_fa={'delta':'δ','sigma':'σ','p_latent':'متغیر نهفته اتصال U','topology':'میدان آرایش W','conductance_field':'میدان رسانایی Z'}
    def sobrows(labels):
        return [[labels[r.factor],num(sobref.loc[r.factor,'index'],4),num(r['index'],4),
                  f'<bdi>{r.difference_vs_reference:.2e} [{r.difference_ci_low:.2e}, {r.difference_ci_high:.2e}]</bdi>'] for _,r in sob.iterrows()]
    def timerows(names):
        return [[names[r.method],int(r.batch_size),f'{1000*r.preprocessing_seconds/r.cases:.3f}',
                 f'{1000*r.neural_core_seconds/r.cases:.3f}',f'{r.milliseconds_per_case:.3f}',f'{r.speedup_vs_sparse_g:.3f}×'] for _,r in tm.iterrows()]
    splits=['continuous','ood_sigma3','ood_fully_connected','ood_size15','ood_size41']
    def oodrows():
        return [[split,int(row(split).cases),int(row(split).positive_cases),pct(row(split).positive_mean_abs_relative_error),pct(row(split).positive_p95_abs_relative_error),pct(row(split).positive_max_abs_relative_error)] for split in splits]
    references='''<div class="refs"><p>[1] Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. <i>J. Computational Physics</i> 378, 686–707. doi:10.1016/j.jcp.2018.10.045.</p>
<p>[2] E, W., Yu, B. (2018). The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems. <i>Communications in Mathematics and Statistics</i> 6, 1–12. doi:10.1007/s40304-018-0127-z.</p>
<p>[3] Jansen, M. J. W. (1999). Analysis of variance designs for model output. <i>Computer Physics Communications</i> 117, 35–43. doi:10.1016/S0010-4655(98)00154-4.</p>
<p>[4] The supplied mini-project brief and the separately preserved direct-solver study in this repository. The present unrolled graph architecture is a custom discrete, parametric physics-informed model; it is not the original continuum architecture in [1].</p></div>'''
    en=[]
    en.append(f'''<section class="page first"><div class="tag">SEPARATE PARAMETRIC PINN STUDY · 04 NOVEMBER 2026</div><h1>Train once, reuse across random networks</h1>
<p><b>Parametric graph PINN versus the verified sparse solver.</b> A neural operator receives the full random conductance field and physical parameters, predicts hydraulic head, and estimates effective conductance. Trained weights are reused without a new sparse solve or per-input weight fitting. Every forward evaluation still performs K fixed iterative updates of head.</p>
<div class="box">On {int(a.cases):,} independent held-out scenario cases ({int(a.positive_cases):,} with positive flow), the K=2048 model has mean absolute relative conductance error <b>{pct(a.positive_mean_abs_relative_error)}</b>, 95th-percentile error <b>{pct(a.positive_p95_abs_relative_error)}</b>, and maximum error <b>{pct(a.positive_max_abs_relative_error)}</b>. The zero-flow decision is exact graph preprocessing, not a learned classifier.</div>
{comparison_en}
<p><b>Practical conclusion.</b> The reusable model works accurately for most in-domain conductance predictions, but errors are not uniformly small. It is an unrolled learned iterative architecture with explicit graph preprocessing, not a closed-form or a single dense-MLP evaluation. {cost_en} The sparse method remains the preferred precision reference for this inexpensive 25×25 problem.</p>
<h3>Why physical scalars alone are insufficient</h3><p>Two networks with identical δ,p,σ can have different topology and conductance fields, including one spanning and one disconnected. In {int(params_only.cases)} test cases with identical δ=0, p=0.5 and σ=1, standard deviation of G was {params_only.sd_g:.5f}; even the best empirical constant predictor had RMSE {params_only.constant_best_empirical_rmse:.5f}. The model therefore takes the full field, or a seed identifying it. With only δ and σ, the API reports ensemble statistics from sampled fields, not a unique realization.</p></section>''')
    en.append(f'''<section class="page"><h2>1 · Physics, parameters and architecture</h2>
<p>Use the same n=25 lattice, 625 nodes and 1,200 possible edges. For each realization p=0.5+δU, U∼Uniform(−1,1); active edges have g=exp(σZ). The neural conditioning scalars include δ,p,σ; node inputs include coordinates, conductance and topology descriptors, and shortest resistance distances to the two boundaries. Given the full g field, δ,p,σ are redundant distribution descriptors, not additional forces in Kirchhoff’s law.</p>
<div class="eq">E(h;g) = Σₑ gₑ(hᵤ−hᵥ)²,  h_left=1, h_right=0<br>∂E/∂hᵢ = 2 Σⱼgᵢⱼ(hᵢ−hⱼ) = 0<br>G_exact = min_h E(h;g),  G_est = E(h_θ;g) ≥ G_exact</div>
<p>The inequality holds for admissible fixed-boundary heads in exact arithmetic; it does not certify a tight error bound. Away from equilibrium, E(hθ) and the inlet flux from hθ need not agree, so both are reported separately.</p>
<h3>A discrete, parametric PINN</h3><ol>
<li>Connected components impose the exact head on one-boundary clusters and identify zero-flow networks. Floating clusters carry zero flow and have undefined absolute head.</li>
<li>Dijkstra distances use resistances 1/g. Initialize h₀=d_right/(d_left+d_right) on spanning components. Uniform conductance scaling is canonicalized by the active-link median.</li>
<li>A 21→32→32→1 node encoder predicts a bounded initial correction. A 6→16→2 parametric MLP predicts graph-level α,β. There are {train['parameters']:,} trainable parameters.</li>
<li>Apply a fixed number K of learned Kirchhoff-residual message-passing layers; clamp final heads to [0,1], which cannot increase graph energy.</li></ol>
<div class="eq">Pᵢⱼ = gᵢⱼ / Σⱼgᵢⱼ<br>hᵏ⁺¹ = hᵏ − α(hᵏ−Phᵏ) + β(hᵏ−hᵏ⁻¹), on free nodes<br>β = 0.999 sigmoid(b), α = (1+β) 0.999 sigmoid(a)</div>
<p>This is a physics-informed unrolled graph network related to variational PINN/Deep Ritz ideas [1,2]. It deliberately contains repeated graph operations. It is neither a continuum coordinate-derivative PINN nor a claim that all computation disappears after training. For arbitrary boundary heads, h=h_R+(h_L−h_R)h_unit and Q_est=G_est(h_L−h_R); scaling all g by c scales G by c exactly.</p></section>''')
    en.append(f'''<section class="page"><h2>2 · One-time training and model selection</h2>
<p>Train on {train['train_data']['retained_spanning']:,} spanning networks, retained from {train['train_data']['generated']:,} independent generated fields. Nonspanning cases need no optimization because their response is fixed structurally. Half of δ and σ draws use scenario values {{0,0.1,0.25}} and {{0,1,2}}, respectively; the other half use Uniform(0,0.25) and Uniform(0,2). A single global p is then sampled per network.</p>
<div class="eq">Loss(θ) = mean log[E(hθ;g) / E(h₀;g)]<br>+ 0.01 mean [r²], r=hθ−Phθ on free nodes and zero elsewhere</div>
<p>The logarithm is monotone; each connected network’s energy minimum is the Kirchhoff solution. Training uses <b>no reference head or conductance labels</b>. Sparse reference solutions are generated only for 512 independent validation networks and for later paired tests. Adam uses initial learning rate 0.0015, weight decay 10⁻⁷ and gradient clipping 1. Batch size 32; 16 epochs with depth curriculum 32→64→128→256. Training uses float32; evaluation uses float64, on CPU with two PyTorch threads.</p>
{fig('01_training','Training physics loss and validation errors. Best epoch is selected by validation mean relative conductance error at K=256.')}
<p>Training-loop time was {train['training_seconds']:.1f} s; total offline setup, validation and depth selection took {train['elapsed_seconds']:.1f} s. Epoch {train['best_epoch']} was selected. Fast mode uses K=256; accurate mode uses K={train['accurate_depth']}. Larger depth reuses the same learned weights; it is additional fixed message passing, not additional training.</p>
<div class="box"><b>Unmet validation target.</b> The desired validation 99th-percentile G error below 0.1% was not met within the tested depth cap 2048 (observed {pct(depths.iloc[-1].p99_relative_g_error)}). The most accurate tested depth was retained, and the failure is not hidden as a guarantee. No settings were tuned to improve the subsequent test results.</div>
<p class="small">Base SeedSequence 20261105; training stream 101, validation 102, test 201–205, sensitivity 301, bootstrap 302, timing 401. Input fingerprints and exact versions are saved. One training seed was run; training-to-training variation is not quantified.</p></section>''')
    en.append(f'''<section class="page"><h2>3 · Paired pointwise accuracy and ablation</h2>
<p>Freeze weights and depth settings before testing. The main paired test has nine δ/σ scenarios × 1,000 realizations, plus 1,000 continuous in-domain cases. Each reference and model sees the <b>same g array</b>. Relative errors exclude exactly zero reference values; disconnected cases are assessed separately.</p>
{table(['Method','Mean |relative G error|','95th pct.','Maximum','Mean free-head RMSE'],error_rows(names_en))}
{fig('02_accuracy','Log-log conductance comparison and cumulative error distributions on positive in-domain cases. Tiny errors are floored only for plotting.')}
<p>At K=256, training improves mean scenario G error by {untrained.positive_mean_abs_relative_error/fast.positive_mean_abs_relative_error:.2f}× versus the untrained physics network. Much of the final accuracy comes from the graph-physics layers; it cannot be attributed to a small dense encoder alone. Across continuous held-out cases, accurate-mode mean relative G error was {pct(c.positive_mean_abs_relative_error)}, with maximum {pct(c.positive_max_abs_relative_error)}.</p>
<p>For scenario tests in accurate mode, mean free-node head RMSE is {a.mean_free_node_rmse:.6f}; 95th-percentile inlet-flux relative error is {pct(a.p95_inlet_relative_error)} and 95th-percentile relative inlet/outlet imbalance is {pct(a.p95_relative_in_out_imbalance)}. G derived from energy is more accurate than directly using the approximate boundary flux. Residuals and all raw fluxes are retained; conservation is approximate.</p></section>''')
    en.append(f'''<section class="page"><h2>4 · Spatial predictions and stochastic statistics</h2>
{fig('03_head_fields','Deliberately difficult in-domain example: largest free-node head RMSE in the held-out scenario set. Floating nodes are blank; local head errors can persist near weakly connected branches.')}
<p>The displayed case has free-node RMSE {difficult['free_node_rmse']:.5f}. Energy-weighted output accuracy does not uniformly bound unweighted local head errors in ill-conditioned or weakly connected subgraphs.</p>
{table(['δ','σ','Old MC mean¹','Paired direct mean','PINN mean','Mean bias','P(G=0)'],uq_rows())}
<p class="small">¹The original study used 10,000 samples per scenario; this independent paired comparison uses 1,000 per scenario. Differences between old and new reference means are Monte Carlo variation. The model discrepancy is the difference within each paired row, not the old-to-new difference.</p>
<p>Means, standard deviations, medians, 95th percentiles, zero probabilities and paired-bias 95% intervals are saved for every scenario and method. Zero probabilities agree exactly by component preprocessing. The bias interval quantifies finite test-sample uncertainty in the mean discrepancy; it is neither a field prediction interval nor a training-uncertainty interval. The ensemble API still samples random fields, but performs no sparse solve for them.</p></section>''')
    en.append(f'''<section class="page"><h2>5 · Uncertainty and sensitivity comparison</h2>
{fig('04_uncertainty_comparison','Same-field Monte Carlo ensemble means and relative biases. Error bars use paired sample differences and normal 95% intervals.')}
<p>Global sensitivity repeats the grouped Jansen design on {evaluation['sensitivity']['base_pairs']:,} independent base pairs ({evaluation['sensitivity']['reference_calls']:,} reference fields), with 1,000 paired-row bootstrap resamples. The five independent groups remain δ,σ,U,W,Z. First-order and total indices, their confidence intervals, and paired model-minus-reference differences are all saved.</p>
{table(['Input group','Direct total index','PINN total index','Paired difference [95% interval]'],sobrows(labels_en))}
{fig('05_sensitivity_comparison','First and total sensitivity of energy-derived G. Sparse-reference bootstrap intervals are shown; full paired intervals for all methods are in CSV.')}
<p class="small">This comparison uses a new sensitivity sample, whereas the original study used 16,384 base pairs. Do not compare these point estimates as if their only difference were model error. The paired differences in this table isolate the approximation on identical fields. Rankings depend on priors and latent parameterization; total indices overlap. Topology controls path existence exactly, so σ and Z have no effect on the binary spanning event in either pipeline.</p></section>''')
    en.append(f'''<section class="page"><h2>6 · Measured computational cost</h2>
{fig('06_cost_accuracy','Warm CPU inference including graph preprocessing, tensor assembly and neural evaluation. Depth selection uses validation only. The offline training cost is additional.')}
{table(['Mode','Batch','Preprocess ms/case','Core ms/case','Total ms/case','Direct/PINN speedup'],timerows(names_en))}
<p>Values are medians of three repetitions on the same 64 new fields. Direct G-only latency was {ref_ms:.3f} ms/case; the CSV also includes full-head sparse timings. The PINN uses two CPU threads, float64 inference and warm loaded weights. Preprocessing includes components and resistance Dijkstra distances; graph generation is excluded for both methods. No GPU results or large-network speedups are inferred.</p>
<div class="box">{cost_en} Reusing weights avoids retraining and a sparse solve, but does not imply a runtime advantage. The original sparse method is already inexpensive; this neural architecture’s many fixed graph layers and preprocessing dominate cost.</div>
<p>The high-accuracy mode gives a smaller approximation error at greater inference cost. A positive break-even requires per-input neural cost lower than sparse cost; the measured break-even formula is T_offline/(t_sparse−t_neural) only when that denominator is positive. No training cost is hidden inside the reference timing. Model loading is excluded from warm inference and should be added for cold one-shot use.</p></section>''')
    en.append(f'''<section class="page"><h2>7 · Generalization and limits</h2>
{table(['Test family','Cases','Positive','Mean rel. G error','95th pct.','Maximum'],oodrows())}
{fig('07_generalization_physics','Out-of-domain tests and the distinction between energy and inlet-flux estimates. All comparisons use new physical fields; no test-time correction is applied.')}
<p>The trained domain is n=25, δ∈[0,0.25], p generated around 0.5, and σ∈[0,2]. The OOD tests use σ=3, fully connected p=1, and n=15 or 41. For other sizes, the same learned weights are reused with regenerated graph-index buffers; no retraining is done. This is exploratory transfer, not a claim of uniform size or parameter generalization.</p>
<p>At σ=3, accurate-mode mean relative G error rises to {pct(ood.positive_mean_abs_relative_error)} and the maximum to {pct(ood.positive_max_abs_relative_error)}. Strong contrast and weakly anchored clusters can converge slowly at fixed depth. The dense fully connected case can be easier despite lying outside the training distribution. Small G errors do not certify all heads or fluxes.</p>
<p>Exact boundaries and zero-flow routing are engineered physical constraints, not learned accomplishments. Component analysis and Dijkstra remain required per field. No corrected sparse solve is applied after the model. This architecture does not satisfy a stronger claim of instant scalar-only prediction for every possible random network. Model and sampling error must remain distinct.</p></section>''')
    en.append(f'''<section class="page"><h2>8 · Use the trained model and reproduce the study</h2>
<p>All new artifacts are separated under <code>parametric_pinn/</code>; the original solver, data and reports remain available. The repository contains trained and untrained checkpoints, training histories, complete paired prediction CSVs, sensitivity outputs, timing records, physical diagnostics, spatial examples, an executed notebook, report-generating scripts and editable presentation.</p>
<div class="eq" style="text-align:left">python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --p .6 --seed 42<br>python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --draws 1000<br>python parametric_pinn/scripts/predict.py --conductances field.npy --p .6 --sigma 1<br>make -C parametric_pinn all</div>
<p>The default CLI never calls the reference solver. Use <code>--compare-reference</code> explicitly for a check; <code>--save-heads</code> exports heads. Physical boundary heads and uniform conductance scale are supported. A .npy field has 1,200 entries, horizontal-then-vertical, row-major, as in the direct solver. API batch prediction reuses one loaded checkpoint. Different seeds can produce different results with the same δ,p,σ; with only distribution parameters, use multiple draws.</p>
<h3>Verification and integrity</h3><p>The 24 original tests plus 12 new PINN tests cover analytic grids, variational bounds, energy gradients, exact boundaries, zero flow, scale equivariance, batch equivalence, no-label training generation, and proof by monkeypatching that prediction does not call the sparse solver. A separate audit rechecks statistics, the zero-flow rule, field-energy consistency, checkpoint identity and the executed notebook. File checksums are in the root inventory.</p>
<h3>Scientific conclusion</h3><p>The study demonstrates a reusable physics-informed parametric graph map with accurate in-domain effective conductance for most cases. The direct solver gives stronger numerical precision and better measured CPU cost for this small lattice. Future work could remove dangling components analytically, learn multilevel preconditioners, optimize batched kernels or train neural operators for larger 3D fracture networks. Such changes require new validation and fair cost accounting, rather than assuming that a PINN is faster or uniformly accurate.</p>
<h3>References</h3>{references}</section>''')

    fa=[]
    fa.append(f'''<section class="page first"><div class="tag">مطالعه مستقل PINN پارامتری · ۴ نوامبر ۲۰۲۶</div><h1>آموزش یک‌باره و پیش‌بینی برای شبکه‌های تصادفی جدید</h1>
<p><b>مقایسه PINN پارامتری گرافی با حل‌گر مستقیم معتبر.</b> مدل، کل میدان رسانایی و پارامترهای فیزیکی را دریافت می‌کند، هد را پیش‌بینی و رسانایی مؤثر را تخمین می‌زند. وزن‌ها برای ورودی جدید دوباره استفاده می‌شوند؛ حل دستگاه تنک یا آموزش مجدد در زمان پیش‌بینی وجود ندارد.</p>
<div class="box">روی <bdi>{int(a.cases):,}</bdi> نمونه آزمون مستقل سناریویی، شامل <bdi>{int(a.positive_cases):,}</bdi> شبکه دارای جریان، حالت ۲۰۴۸ لایه میانگین خطای نسبی مطلق رسانایی <bdi>{pct(a.positive_mean_abs_relative_error)}</bdi>، صدک ۹۵ خطا <bdi>{pct(a.positive_p95_abs_relative_error)}</bdi> و بیشترین خطا <bdi>{pct(a.positive_max_abs_relative_error)}</bdi> داشت. تشخیص جریان صفر از تحلیل دقیق مؤلفه‌های گراف می‌آید و یک طبقه‌بند آموزش‌دیده نیست.</div>
{comparison_fa}
<p><b>نتیجه عملی:</b> مدل برای بیشتر پیش‌بینی‌های رسانایی در دامنه آموزش دقیق است، ولی خطا برای همه نمونه‌ها یکسان نیست. معماری، شبکه‌ای با بازکردن تعداد ثابتی گام فیزیکی و پیش‌پردازش گراف است؛ فرمول بسته یا تنها یک MLP نیست. {cost_fa} حل‌گر مستقیم برای این شبکه کوچک، مرجع مناسب دقت باقی می‌ماند.</p>
<h3>چرا فقط پارامترهای آماری کافی نیستند؟</h3><p>دو شبکه با δ، p و σ یکسان می‌توانند آرایش متفاوت و حتی وضعیت اتصال متفاوت داشته باشند. در ۱۰۰۰ نمونه با <bdi>δ=0, p=0.5, σ=1</bdi>، انحراف معیار رسانایی <bdi>{params_only.sd_g:.5f}</bdi> و خطای RMSE بهترین پیش‌بینی ثابت تجربی <bdi>{params_only.constant_best_empirical_rmse:.5f}</bdi> بود. بنابراین مدل باید میدان کامل یا بذر مشخص‌کننده آن را بگیرد. با ورودی صرفاً δ و σ، خروجی معنادار، آمار مجموعه‌ای از شبکه‌های نمونه‌برداری‌شده است.</p></section>''')
    fa.append(f'''<section class="page"><h2>۱. معادلات، پارامترها و معماری</h2>
<p>همان شبکه ۲۵×۲۵ با ۶۲۵ گره و ۱۲۰۰ لینک استفاده شد. در هر تحقق، <bdi>p=0.5+δU</bdi> و رسانایی لینک فعال <bdi>g=exp(σZ)</bdi> است. δ و p و σ ورودی شرطی شبکه‌اند؛ مختصات، توصیفگرهای رسانایی و آرایش و فاصله مقاومتی کوتاه‌ترین مسیر به دو مرز نیز ورودی گره‌ها هستند. وقتی میدان g معلوم باشد، پارامترهای آماری نقش توصیف توزیع را دارند و نیروی فیزیکی تازه‌ای به قانون کیرشهف اضافه نمی‌کنند.</p>
<div class="eq">E(h;g)=Σₑgₑ(hᵤ−hᵥ)², h_left=1, h_right=0<br>∂E/∂hᵢ=2Σⱼgᵢⱼ(hᵢ−hⱼ)=0<br>G_exact=min_h E(h;g), G_est=E(h_θ;g)≥G_exact</div>
<p>نامساوی در حساب دقیق برای هدهای مجاز برقرار است و به‌تنهایی کوچک‌بودن خطا را گواهی نمی‌کند. دور از تعادل، انرژی و شار ورودی از هد تقریبی برابر نیستند؛ هر دو جدا ثبت شده‌اند.</p>
<h3>PINN پارامتری روی گراف گسسته</h3><ol>
<li>مؤلفه‌های همبند، شبکه بدون جریان و هد ثابت خوشه‌های متصل به یک مرز را تعیین می‌کنند. هد مطلق جزیره‌های داخلی نامعین و جریان آن‌ها صفر است.</li>
<li>فاصله دایکسترا با مقاومت <bdi>1/g</bdi> محاسبه و <bdi>h₀=d_right/(d_left+d_right)</bdi> ساخته می‌شود. برای حفظ مقیاس‌پذیری دقیق، رسانایی با میانه لینک‌های فعال نرمال می‌شود.</li>
<li>یک رمزگذار گره‌ای <bdi>21→32→32→1</bdi> تصحیح محدود هد اولیه را می‌سازد. یک MLP پارامتری <bdi>6→16→2</bdi> ضرایب α و β را تعیین می‌کند. مدل <bdi>{train['parameters']:,}</bdi> پارامتر آموزش‌پذیر دارد.</li>
<li>تعداد ثابت K لایه پیام‌رسانی با باقیمانده کیرشهف اجرا می‌شود. محدودکردن هد نهایی به بازه صفر تا یک، انرژی را افزایش نمی‌دهد.</li></ol>
<div class="eq">Pᵢⱼ=gᵢⱼ/Σⱼgᵢⱼ<br>hᵏ⁺¹=hᵏ−α(hᵏ−Phᵏ)+β(hᵏ−hᵏ⁻¹)<br>β=0.999 sigmoid(b), α=(1+β)0.999 sigmoid(a)</div>
<p>این یک معماری گرافی فیزیک‌آگاه با گام‌های بازشده و تابع انرژی، مرتبط با ایده‌های PINN و Deep Ritz است [۱،۲]. مشتق مکانی یک PINN پیوسته در اینجا جایگزین معادلات گراف نشده است. برای هد دلخواه، <bdi>h=h_R+(h_L−h_R)h_unit</bdi> و <bdi>Q_est=G_est(h_L−h_R)</bdi> اعمال می‌شود؛ ضرب کل g در c نیز رسانایی مؤثر را دقیقاً در c ضرب می‌کند.</p></section>''')
    fa.append(f'''<section class="page"><h2>۲. آموزش یک‌باره و انتخاب مدل</h2>
<p>از <bdi>{train['train_data']['generated']:,}</bdi> میدان مستقل تولیدشده، <bdi>{train['train_data']['retained_spanning']:,}</bdi> شبکه عبوری برای آموزش نگه داشته شد. شبکه‌های غیرعبوری به آموزش نیاز ندارند، چون پاسخشان ساختاری معلوم است. نیمی از نمونه‌های δ و σ از مقادیر سناریوها و نیمی از توزیع‌های یکنواخت در <bdi>[0,0.25]</bdi> و <bdi>[0,2]</bdi> گرفته شدند؛ سپس یک p سراسری برای هر شبکه نمونه‌برداری شد.</p>
<div class="eq">Loss(θ)=mean log[E(hθ;g)/E(h₀;g)]<br>+0.01 mean[(hθ−Phθ)² on free nodes]</div>
<p>لگاریتم صعودی است و کمینه انرژی هر شبکه عبوری همان جواب کیرشهف است. <b>هیچ برچسب هد یا رسانایی مرجع در آموزش استفاده نشد.</b> حل‌گر مستقیم فقط برای ۵۱۲ شبکه اعتبارسنجی مستقل و آزمون‌های بعدی فراخوانی شد. بهینه‌ساز Adam، نرخ اولیه ۰٫۰۰۱۵، زوال وزن <bdi>10⁻⁷</bdi>، محدودسازی نرم گرادیان در ۱، دسته ۳۲ و ۱۶ دوره با افزایش عمق <bdi>32→64→128→256</bdi> استفاده شد. آموزش float32 و ارزیابی float64، روی CPU با دو رشته PyTorch بود.</p>
{fig('01_training','هزینه فیزیکی آموزش و خطاهای اعتبارسنجی؛ انتخاب وزن براساس میانگین خطای نسبی رسانایی اعتبارسنجی در عمق ۲۵۶ انجام شد.')}
<p>حلقه آموزش <bdi>{train['training_seconds']:.1f}</bdi> ثانیه و کل آماده‌سازی، اعتبارسنجی و انتخاب عمق <bdi>{train['elapsed_seconds']:.1f}</bdi> ثانیه طول کشید. دوره <bdi>{train['best_epoch']}</bdi> انتخاب شد. حالت سریع K=۲۵۶ و حالت دقیق K=۲۰۴۸ دارد؛ افزایش عمق با همان وزن‌هاست و آموزش تازه‌ای انجام نمی‌شود.</p>
<div class="box"><b>هدف اعتبارسنجی محقق‌نشده:</b> هدف صدک ۹۹ خطای رسانایی کمتر از ۰٫۱٪ تا سقف عمق ۲۰۴۸ محقق نشد؛ مقدار مشاهده‌شده <bdi>{pct(depths.iloc[-1].p99_relative_g_error)}</bdi> بود. دقیق‌ترین عمق آزموده‌شده نگه داشته شد و این هدف به‌صورت تضمین گزارش نمی‌شود. هیچ تنظیمی برای بهترشدن نتایج آزمون بعدی تغییر نکرد.</div>
<p class="small">بذر پایه <bdi>20261105</bdi>، جریان آموزش ۱۰۱، اعتبارسنجی ۱۰۲، آزمون ۲۰۱ تا ۲۰۵، حساسیت ۳۰۱، بوت‌استرپ ۳۰۲ و زمان‌سنجی ۴۰۱ است. فقط یک بذر آموزش اجرا شد؛ پراکندگی ناشی از تکرار آموزش سنجیده نشده است.</p></section>''')
    fa.append(f'''<section class="page"><h2>۳. دقت نقطه‌ای و سهم آموزش</h2>
<p>وزن‌ها و عمق‌ها پیش از آزمون ثابت شدند. آزمون اصلی شامل ۹ سناریوی δ و σ، هرکدام ۱۰۰۰ تحقق، به‌اضافه ۱۰۰۰ نمونه پیوسته در دامنه آموزش است. هر دو روش دقیقاً <b>همان آرایه g</b> را دریافت می‌کنند. خطای نسبی فقط برای مرجع مثبت محاسبه می‌شود؛ شبکه‌های با جریان صفر جدا سنجیده می‌شوند.</p>
{table(['روش','میانگین خطای نسبی G','صدک ۹۵','بیشترین','میانگین RMSE هد آزاد'],error_rows(names_fa))}
{fig('02_accuracy','مقایسه رسانایی در مقیاس لگاریتمی و تابع توزیع خطا در نمونه‌های مثبتِ درون دامنه؛ کف خطای بسیار کوچک فقط برای رسم اعمال شده است.')}
<p>در عمق ۲۵۶، آموزش، میانگین خطای رسانایی سناریوها را نسبت به معماری فیزیکی بدون آموزش حدود <bdi>{untrained.positive_mean_abs_relative_error/fast.positive_mean_abs_relative_error:.2f}</bdi> برابر بهتر کرد. بخش بزرگی از دقت نهایی از لایه‌های فیزیکی گراف می‌آید و نباید همه آن را به رمزگذار کوچک نسبت داد. در آزمون پیوسته مستقل، میانگین خطای نسبی حالت دقیق <bdi>{pct(c.positive_mean_abs_relative_error)}</bdi> و بیشترین خطا <bdi>{pct(c.positive_max_abs_relative_error)}</bdi> بود.</p>
<p>برای حالت دقیق در آزمون سناریویی، میانگین RMSE هد گره‌های آزاد <bdi>{a.mean_free_node_rmse:.6f}</bdi>، صدک ۹۵ خطای نسبی شار ورودی <bdi>{pct(a.p95_inlet_relative_error)}</bdi> و صدک ۹۵ عدم‌تطابق نسبی ورودی و خروجی <bdi>{pct(a.p95_relative_in_out_imbalance)}</bdi> است. تخمین انرژی رسانایی از شار مرزی هد تقریبی دقیق‌تر است؛ بقای جریان همچنان تقریبی است. همه شارها و باقیمانده‌ها در خروجی خام ثبت شده‌اند.</p></section>''')
    fa.append(f'''<section class="page"><h2>۴. میدان هد و آمار تصادفی</h2>
{fig('03_head_fields','نمونه دشوار عمدی: بیشترین RMSE هد آزاد در آزمون سناریویی. گره‌های با هد نامعین خالی‌اند. خطای موضعی در شاخه‌های ضعیف‌متصل می‌تواند باقی بماند.')}
<p>نمونه نمایش‌داده‌شده RMSE هد آزاد <bdi>{difficult['free_node_rmse']:.5f}</bdi> دارد. دقت خروجی وزن‌دار با انرژی، برای خطای موضعی بدون وزن در زیرگراف‌های بدشرط کران یکنواختی ایجاد نمی‌کند.</p>
{table(['δ','σ','میانگین قدیمی¹','مرجع جفت‌شده','میانگین PINN','اریبی میانگین','احتمال صفر'],uq_rows())}
<p class="small">¹مطالعه اولیه برای هر سناریو ۱۰٬۰۰۰ نمونه و مقایسه جفت‌شده حاضر ۱۰۰۰ نمونه مستقل تازه دارد. اختلاف دو میانگین مرجع می‌تواند از نمونه‌گیری مونت‌کارلو باشد؛ اختلاف مدل، فقط تفاضل همان سطر جفت‌شده است.</p>
<p>میانگین، انحراف معیار، میانه، صدک ۹۵، احتمال صفر و فاصله اطمینان ۹۵٪ اریبی جفت‌شده برای همه روش‌ها ذخیره شده است. احتمال صفر به‌دلیل پیش‌پردازش مؤلفه‌ها دقیقاً برابر است. فاصله اریبی، خطای تعداد محدود نمونه آزمون را بیان می‌کند؛ بازه پیش‌بینی میدانی یا عدم‌قطعیت تکرار آموزش نیست. رابط آماری همچنان میدان‌های تصادفی را نمونه‌برداری می‌کند، اما برای آن‌ها دستگاه تنک حل نمی‌کند.</p></section>''')
    fa.append(f'''<section class="page"><h2>۵. مقایسه عدم‌قطعیت و حساسیت</h2>
{fig('04_uncertainty_comparison','میانگین‌های مونت‌کارلو روی میدان‌های یکسان و اریبی نسبی آن‌ها؛ میله‌های خطا از تفاضل جفت‌شده و تقریب نرمال ۹۵٪ ساخته شده‌اند.')}
<p>طرح گروهی یانسن با <bdi>{evaluation['sensitivity']['base_pairs']:,}</bdi> جفت پایه مستقل، یعنی <bdi>{evaluation['sensitivity']['reference_calls']:,}</bdi> میدان مرجع، و ۱۰۰۰ بازنمونه‌گیری بوت‌استرپ از سطرهای جفت‌شده اجرا شد. پنج گروه مستقل همان δ، σ، U، W و Z هستند. شاخص‌های مرتبه اول و کل، فاصله‌های اطمینان و اختلاف جفت‌شده مدل با مرجع همگی ذخیره شده‌اند.</p>
{table(['گروه ورودی','اثر کل مستقیم','اثر کل PINN','اختلاف جفت‌شده [فاصله ۹۵٪]'],sobrows(labels_fa))}
{fig('05_sensitivity_comparison','مقایسه شاخص‌های مرتبه اول و کل برای G مبتنی بر انرژی؛ فواصل مرجع در شکل و فواصل کامل همه روش‌ها در CSV آمده‌اند.')}
<p class="small">این نمونه حساسیت تازه است؛ مطالعه اولیه ۱۶٬۳۸۴ جفت پایه داشت. اختلاف نقطه‌ای شاخص‌های دو مطالعه، صرفاً خطای مدل نیست. اختلاف‌های جفت‌شده جدول، تقریب شبکه را روی ورودی یکسان جدا می‌کنند. رتبه‌بندی به توزیع ورودی وابسته است و شاخص‌های کل هم‌پوشانی دارند. وجود مسیر در هر دو روش از آرایش تعیین می‌شود؛ σ و Z بر رخداد دودویی اتصال اثری ندارند.</p></section>''')
    fa.append(f'''<section class="page"><h2>۶. هزینه محاسباتی اندازه‌گیری‌شده</h2>
{fig('06_cost_accuracy','زمان پیش‌بینی گرم CPU شامل پیش‌پردازش، ساخت تنسور و اجرای مدل؛ هزینه آموزش اولیه جداست. منحنی عمق فقط از اعتبارسنجی آمده است.')}
{table(['حالت','دسته','پیش‌پردازش ms','هسته ms','کل ms/نمونه','مستقیم/PINN'],timerows(names_fa))}
<p>اعداد، میانه سه تکرار روی همان ۶۴ میدان تازه هستند. زمان حل مستقیم فقط رسانایی <bdi>{ref_ms:.3f}</bdi> میلی‌ثانیه برای هر نمونه بود؛ زمان حل میدان کامل هم در CSV آمده است. PINN با دو رشته CPU و float64، پس از بارگذاری وزن‌ها اجرا شد. مؤلفه‌ها، دایکسترا و ساخت تنسور در زمان مدل حساب شده‌اند؛ تولید میدان تصادفی برای هر دو روش حذف شده است. نتیجه‌ای درباره GPU یا شبکه بزرگ‌تر فرض نمی‌شود.</p>
<div class="box">{cost_fa} استفاده مجدد از وزن‌ها، نیاز به آموزش تازه و حل تنک را حذف می‌کند، اما برتری زمانی را تضمین نمی‌کند. حل مستقیم شبکه ۲۵×۲۵ ارزان است و در مدل حاضر، تعداد زیاد لایه‌ها و پیش‌پردازش هزینه غالب‌اند.</div>
<p>حالت دقیق، خطای کمتر و هزینه بیشتری دارد. نقطه سربه‌سر مثبت فقط زمانی تعریف می‌شود که هزینه هر پیش‌بینی شبکه کمتر از حل مستقیم باشد: <bdi>T_offline/(t_sparse−t_neural)</bdi>. زمان بارگذاری مدل در آزمون گرم نیست و برای استفاده تک‌نمونه سرد باید اضافه شود. هزینه آموزش نیز از مقایسه حذف یا در روش مرجع پنهان نشده است.</p></section>''')
    fa.append(f'''<section class="page"><h2>۷. تعمیم و محدودیت‌ها</h2>
{table(['خانواده آزمون','تعداد','مثبت','میانگین خطای G','صدک ۹۵','بیشترین'],oodrows())}
{fig('07_generalization_physics','آزمون خارج دامنه و تفاوت تخمین انرژی با شار ورودی؛ هیچ تصحیح عددی یا آموزش پس از پیش‌بینی انجام نشده است.')}
<p>دامنه آموزش، اندازه ۲۵، <bdi>δ∈[0,0.25]</bdi>، p تصادفی پیرامون ۰٫۵ و <bdi>σ∈[0,2]</bdi> است. آزمون خارج دامنه شامل σ=۳، اتصال کامل p=۱ و اندازه‌های ۱۵ و ۴۱ است. برای اندازه تازه، همان وزن‌ها با شاخص‌های گراف بازساخته‌شده استفاده شدند؛ آموزش مجدد انجام نشد. این بررسی، انتقال اکتشافی است و تضمین تعمیم یکنواخت برای همه اندازه‌ها نیست.</p>
<p>در σ=۳، میانگین خطای رسانایی حالت دقیق به <bdi>{pct(ood.positive_mean_abs_relative_error)}</bdi> و بیشترین خطا به <bdi>{pct(ood.positive_max_abs_relative_error)}</bdi> رسید. اختلاف شدید رسانایی و خوشه‌های ضعیف‌متصل می‌توانند در عمق ثابت کند همگرا شوند. شبکه کاملاً متصل، با وجود خارج‌دامنه‌بودن، ممکن است آسان‌تر باشد. خطای کوچک G، همه هدها و شارها را گواهی نمی‌کند.</p>
<p>شرط مرزی دقیق و تشخیص جریان صفر، قیود فیزیکی طراحی‌شده‌اند و موفقیت یادگیری محسوب نمی‌شوند. تحلیل مؤلفه و دایکسترا برای هر میدان هنوز لازم‌اند. هیچ حل تنک تصحیحی بعد از مدل اجرا نشده است. این معماری ادعای قوی‌ترِ پاسخ فوری از چند پارامتر اسکالر برای هر شبکه تصادفی ممکن را محقق نمی‌کند؛ خطای تقریب مدل و خطای نمونه‌گیری باید جدا بمانند.</p></section>''')
    fa.append(f'''<section class="page"><h2>۸. استفاده و بازتولید کامل</h2>
<p>تمام خروجی‌های این مرحله در پوشه مستقل <bdi>parametric_pinn/</bdi> قرار دارند؛ داده‌ها و گزارش‌های روش مستقیم حفظ شده‌اند. وزن آموزش‌دیده و کنترل بدون آموزش، تاریخچه آموزش، خروجی‌های جفت‌شده، تحلیل حساسیت، زمان‌سنجی، مثال‌های میدان، دفترچه اجراشده، اسکریپت‌ها و ارائه قابل‌ویرایش تحویل شده‌اند.</p>
<div class="eq" style="text-align:left">python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --p .6 --seed 42<br>python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --draws 1000<br>python parametric_pinn/scripts/predict.py --conductances field.npy --p .6 --sigma 1<br>make -C parametric_pinn all</div>
<p>فرمان پیش‌فرض هرگز حل‌گر مرجع را فراخوانی نمی‌کند. برای بررسی اختیاری از <bdi>--compare-reference</bdi> و برای ذخیره هد از <bdi>--save-heads</bdi> استفاده کنید. هد دو مرز و ضریب مقیاس رسانایی قابل‌تنظیم‌اند. فایل رسانایی، آرایه ۱۲۰۰تایی با ترتیب لینک‌های افقی و سپس عمودی، سطرمحور است. API دسته‌ای از یک وزن بارگذاری‌شده استفاده می‌کند. برای بذرهای متفاوت، پاسخ با همان پارامترهای آماری می‌تواند متفاوت باشد.</p>
<h3>اعتبارسنجی و صحت فایل‌ها</h3><p>۲۴ آزمون قبلی و ۱۲ آزمون جدید شامل شبکه تحلیلی، کران انرژی، گرادیان انرژی، شرط مرزی، جریان صفر، مقیاس‌پذیری، تطابق دسته و تک‌نمونه، تولید آموزش بدون برچسب و آزمون جلوگیری از فراخوانی حل‌گر مستقیم در پیش‌بینی است. ممیزی جداگانه، جدول‌ها، قاعده صفر، سازگاری انرژی و میدان، وزن‌ها و دفترچه اجراشده را بررسی می‌کند. اثر انگشت فایل‌ها در فهرست اصلی مخزن ثبت شده است.</p>
<h3>برداشت نهایی علمی</h3><p>یک نگاشت پارامتری فیزیک‌آگاه قابل‌استفاده مجدد ساخته شد که رسانایی بیشتر نمونه‌های درون دامنه را دقیق پیش‌بینی می‌کند. حل مستقیم برای این شبکه کوچک، دقت عددی قوی‌تر و زمان CPU بهتری دارد. حذف تحلیلی شاخه‌های آویزان، یادگیری پیش‌شرط‌ساز چندسطحی، بهینه‌سازی اجرای دسته‌ای و توسعه به شکستگی سه‌بعدی می‌توانند مراحل بعدی باشند؛ هرکدام به اعتبارسنجی و محاسبه منصفانه هزینه نیاز دارند.</p>
<h3>منابع</h3>{references}</section>''')
    for lang,pages in [('en',en),('fa',fa)]:
        html=f'<!doctype html><html lang="{lang}" dir="{"rtl" if lang=="fa" else "ltr"}"><meta charset="utf-8"><title>Parametric PINN comparison</title><style>{CSS}</style><body class="{lang}">'+''.join(pages)+'</body></html>'
        path=D/f'comparison_{lang}.html';path.write_text(html,encoding='utf-8')
        HTML(filename=str(path)).write_pdf(D/f'comparison_{lang}.pdf')
        print(f'Built comparison_{lang}.pdf')


if __name__=='__main__':main()
