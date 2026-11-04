#!/usr/bin/env python3
"""Build an editable 12-slide PowerPoint and a matching standalone PDF."""
import html
import json
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from weasyprint import HTML

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports'


def main():
    s=pd.read_csv(ROOT/'results/scenario_summary.csv')
    sen=pd.read_csv(ROOT/'results/sobol_summary.csv')
    diag=pd.read_csv(ROOT/'results/solver_diagnostics.csv')
    manifest=json.loads((ROOT/'results/manifest.json').read_text())
    base=s[(s.delta==.25)&(s.sigma==1)].iloc[0]
    low=s[(s.delta==0)&(s.sigma==1)].iloc[0]
    spec=[]
    def add(title,subtitle,bullets,image=None,rows=None,notes=''):
        spec.append(dict(title=title,subtitle=subtitle,bullets=bullets,image=image,rows=rows,notes=notes))
    add('Flow in a 2D random network','Uncertainty, global sensitivity and verified computation',[
        'Connectivity dominates flow variability over the selected ranges.',
        f'{manifest["solver_calls"]:,} model evaluations; 24 independent verification tests.',
        'Completed analysis • 04 November 2026'], '01_network',
        notes='Open by distinguishing the probability of any flow from the amount of flow. All results are dimensionless.')
    add('A hierarchical random-network model','625 nodes · 1,200 possible links · n = 25',[
        'Draw ONE p per realization: p = 0.5 + δU, U uniform on [−1, 1].',
        'Conditional on p: each link active independently with probability p.',
        'Active links: ln(g) ~ Normal(0, σ²); inactive links: g = 0.',
        'Left head = 1; right head = 0; no flow across top and bottom.'], '01_network',
        notes='The plotted examples use n=15 for legibility; the main study uses n=25. Unconditional edges share p and are correlated.')
    add('Solve only hydraulically relevant components','Sparse Kirchhoff equations with explicit disconnected-component treatment',[
        'Find components that touch both fixed-head boundaries.',
        'Solve L_FF h_F = −L_FB h_B on their free nodes.',
        'One-boundary clusters: constant head. Floating islands: arbitrary head, zero flow.',
        'G_eff = inlet flow / Δh; here Δh = 1.'],rows=[
        ['Verification','Evidence'],
        ['Uniform grid','G = g n/(n−1), exact linear head'],
        ['Series / parallel branches','Analytical resistance identities'],
        ['Conservation / energy','Qin = Qout = dissipated energy'],
        ['Scaling / monotonicity','G(cg)=cG(g); Rayleigh property'],
        ['Ishigami benchmark','Known first and total Sobol indices'],
        ['1,000 random diagnostics',f'Max residual {diag.max_residual.max():.2e}']],
        notes='No artificial conductivity is added to disconnected links. Verification establishes numerical correctness, not field validation.')
    add('An explicit, reproducible experiment','Separate RNG streams; common random numbers for paired scenario comparisons',[
        'δ = 0, 0.10, 0.25; σ = 0, 1, 2.',
        'Compare n = 15, 25 and 41 to expose finite-size dependence.',
        'Seed 20261104; all realization-level outputs are retained.'],rows=[
        ['Experiment','Design','Evaluations'],
        ['Main uncertainty','9 × 10,000','90,000'],
        ['Fixed-p sweep','9 × 2,500','22,500'],
        ['Extra sizes','4 × 5,000','20,000'],
        ['Grouped Sobol','16,384 × 7','114,688'],
        ['Numerical diagnostics','1,000','1,000'],
        ['Total','Size 25 reuses main data',f'{manifest["solver_calls"]:,}']],
        notes='Parameter ranges are exploratory. They are justified by covering the transition and strong heterogeneity, rather than claimed to be calibrated priors.')
    add('More uncertain connectivity raises mean flow','The probability of any flow remains close to one half',[
        f'At σ=1, mean G: {low["mean"]:.4f} → {base["mean"]:.4f} as δ: 0 → 0.25.',
        f'Mean ratio: {base["mean"]/low["mean"]:.2f}×, despite unchanged E[p]=0.5.',
        f'Reference spanning probability: {100*base.p_span:.2f}%.',
        'High-p realizations contribute substantial flow; low-p realizations often remain at zero.'], '02_uncertainty',
        notes='Error bars are sampling confidence intervals, not predictive intervals. The trend is empirical over these ranges.')
    add('A mean cannot describe the output distribution','Separate zero-flow probability from positive-flow statistics',[
        f'Reference: δ=0.25, σ=1; P(G=0) = {100*base.p_zero:.2f}%.',
        f'Mean {base["mean"]:.4f}; median {base["median"]:.4f}; 95th percentile {base.q95:.4f}.',
        f'Mean given G>0: {base.positive_mean:.4f}.',
        'A single lognormal law is unsuitable for a distribution with an atom at zero.'], '03_distribution',
        notes='The histogram shows density in log space, conditional on positive flow. Full raw samples are provided.')
    add('A transition controls the response','Fixed-p sweep: 2,500 realizations per p; σ=1; n=25',[
        'For an infinite square bond lattice, p_c = 0.5 (Kesten, 1980).',
        'A finite network has a smooth connectivity crossover.',
        'Nonlinear flow response explains the strong effect of the width δ.',
        'Shared p induces covariance δ²/3 between distinct link indicators.'], '04_connectivity',
        notes='Do not interpret p=0.5 as a deterministic no-flow condition in a finite network, or sample a different p for each link.')
    add('Sensitivity of a stochastic model','Grouped Sobol analysis on independent latent inputs',[
        'Use five independent groups; keep δ and realized p out of the same independent-input list.',
        'Jansen first and total indices: interactions are explicitly retained.',
        '16,384 base pairs; 1,000 paired-row bootstrap resamples.',
        'The estimator also passes the analytic Ishigami benchmark.'],rows=[
        ['Group','Distribution / role'],
        ['δ','Uniform(0, 0.25), connectivity width'],
        ['σ','Uniform(0, 2), log-conductance SD'],
        ['U','Uniform(−1, 1), p = 0.5 + δU'],
        ['Topology W','1,200 independent Uniform(0,1) values'],
        ['Conductance Z','1,200 independent standard normals']],
        notes='The expanded simulator is deterministic given these inputs. Importance depends on priors and latent parameterization. Sobol indices target unconditional G, including zeros.')
    totals=sen[sen.output=='geff'].set_index('factor')
    add('Global connectivity is the dominant influence','Total-order indices include every interaction involving each group',[
        f'U: total {totals.loc["p_latent","total"]:.3f}; first {totals.loc["p_latent","first"]:.3f}.',
        f'δ: total {totals.loc["delta","total"]:.3f}; first {totals.loc["delta","first"]:.3f}. Strong interactions.',
        f'Topology: {totals.loc["topology","total"]:.3f}; σ: {totals.loc["sigma","total"]:.3f}; Z: {totals.loc["conductance_field","total"]:.3f}.',
        'σ and Z have exactly zero total effect on path existence. Do not sum overlapping total indices.'], '05_sensitivity',
        notes='Small first-order estimates for known inactive factors reflect Monte Carlo error. Raw estimates and intervals are retained, including negatives.')
    half=(1.96*s.mcse/s['mean'])*100
    add('Sampling precision improves; size effects remain','Monte Carlo convergence is distinct from finite-size sensitivity',[
        f'Final relative 95% half-widths: {half.min():.2f}–{half.max():.2f}% for scenario means.',
        'At δ=0, σ=1: mean G decreases substantially with n.',
        'Changing n changes the random-network ensemble.',
        'A size-independent continuum limit has not been demonstrated.'], '06_convergence',
        notes='Prefix Sobol convergence is supplied as figure 08 and CSV. The leading rank is resolved; tiny indices remain uncertain.')
    add('Heterogeneity is not a pure variance change','The specified lognormal family fixes the median, not the mean',[
        'Active-link E[g] = exp(σ²/2): 1, 1.6487, 7.3891 for σ = 0, 1, 2.',
        'Mean network conductance still falls with σ over these scenarios.',
        'Weak links on series paths act as bottlenecks.',
        'Constant-mean control: g* = g exp(−σ²/2), so G* = G exp(−σ²/2) exactly.'], '07_normalization',
        notes='The constant-mean normalization is a supplementary control. It does not replace the assigned ln(g) ~ Normal(0, σ²) model.')
    add('Interpretation and next steps','Numerically verified behavior of a simplified dimensionless lattice',[
        'Measure connectivity carefully; report connection probability and flow magnitude separately.',
        'Limits: regular 2D geometry, exploratory priors, steady linear single-phase flow, finite size.',
        'Next: 3D fracture geometry, aperture-based conductance, correlation, matrix exchange and calibration.',
        'No field-data validation or geological prediction accuracy is claimed.'],rows=[
        ['Deliverable','Location'],
        ['Persian and English reports','reports/report_fa.pdf; report_en.pdf'],
        ['Code and 24 tests','src/flow_network; tests/'],
        ['Raw samples and statistics','results/'],
        ['Executed notebook','notebooks/analysis.ipynb'],
        ['One-command reproduction','README.md; scripts/'],
        ['GitHub','mohrezsayeh90-ai/Interview_upsla']],
        notes='References: Jansen (1999), CPC 117, 35–43, doi:10.1016/S0010-4655(98)00154-4; Kesten (1980), CMP 74, 41–59, doi:10.1007/BF01197577. Full references are in both reports.')

    prs=Presentation()
    prs.slide_width,prs.slide_height=Inches(13.333333),Inches(7.5)
    def text(slide,x,y,w,h,content,size=20,color='23394B',bold=False):
        tf=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame
        tf.word_wrap=True
        for i,line in enumerate(content.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
            p.text=line;p.font.size=Pt(size);p.font.name='DejaVu Sans';p.font.bold=bold
            p.font.color.rgb=RGBColor.from_string(color)
            p.space_after=Pt(12)
        return tf
    htmlslides=[]
    for idx,item in enumerate(spec,1):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        text(slide,.42,.28,12.4,.65,item['title'],28,'176B87',True)
        text(slide,.45,1.02,12.4,.5,item['subtitle'],16,'607681')
        image=item['image']
        rows=item['rows']
        if image:
            slide.shapes.add_picture(str(REPORT/'figures'/f'{image}.png'),Inches(.35),Inches(1.92),width=Inches(8.05))
        elif rows:
            shape=slide.shapes.add_table(len(rows),len(rows[0]),Inches(.45),Inches(1.85),Inches(7.7),Inches(4.4))
            for row_index,row in enumerate(rows):
                for col_index,value in enumerate(row):
                    cell=shape.table.cell(row_index,col_index);cell.text=value
                    cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('176B87' if row_index==0 else ('EDF5F7' if row_index%2 else 'FFFFFF'))
                    cell.margin_left=Inches(.1);cell.margin_right=Inches(.08)
                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.size=Pt(13.5 if len(row)>2 else 15)
                        paragraph.font.name='DejaVu Sans'
                        paragraph.font.color.rgb=RGBColor.from_string('FFFFFF' if row_index==0 else '23394B')
        text(slide,8.65,1.82,4.15,4.95,'\n'.join('• '+x for x in item['bullets']),18)
        text(slide,.45,7.05,11.9,.25,'RANDOM FLOW NETWORKS  •  04 NOV 2026  •  Source and full statistical intervals in the reports',9,'607681')
        text(slide,12.45,7.05,.5,.25,str(idx),10,'176B87')
        slide.notes_slide.notes_text_frame.text=item['notes']
        left=(f'<img src="figures/{image}.png">' if image else '<table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(c)}</{"th" if i==0 else "td"}>' for c in row)+'</tr>' for i,row in enumerate(rows))+'</table>')
        htmlslides.append(f'<section><h1>{html.escape(item["title"])}</h1><p class="subtitle">{html.escape(item["subtitle"])}</p><div class="layout"><div class="left">{left}</div><ul>'+''.join(f'<li>{html.escape(b)}</li>' for b in item['bullets'])+f'</ul></div><footer>RANDOM FLOW NETWORKS · 04 NOV 2026 <span>{idx}</span></footer></section>')
    prs.save(REPORT/'presentation_en.pptx')
    css='''@page{size:13.333333in 7.5in;margin:0;}body{margin:0;font-family:'DejaVu Sans';color:#23394b}section{height:7.5in;padding:.3in .45in;position:relative;box-sizing:border-box;break-after:page}section:last-child{break-after:auto}h1{font-size:28pt;color:#176b87;margin:0 0 .12in}p.subtitle{font-size:16pt;color:#607681;margin:0}.layout{display:flex;align-items:center;gap:.3in;margin-top:.4in;height:4.95in}.left{width:7.65in;flex-shrink:0}.left img{width:100%}ul{font-size:18pt;line-height:1.38;margin:0;padding-left:.28in}li{margin-bottom:.23in}table{border-collapse:collapse;width:100%;font-size:14pt}th,td{padding:.16in .1in;border-bottom:1px solid #c5dce4}th{background:#176b87;color:white;text-align:left}tr:nth-child(even){background:#edf5f7}footer{position:absolute;bottom:.2in;font-size:9pt;color:#607681;left:.45in;right:.45in}footer span{float:right;color:#176b87}'''
    path=REPORT/'presentation_en.html'
    path.write_text('<!doctype html><meta charset="utf-8"><title>Random-network flow: presentation</title><style>'+css+'</style>'+''.join(htmlslides))
    HTML(filename=str(path)).write_pdf(REPORT/'presentation_en.pdf')
    notes='\n\n'.join(f'## {i}. {x["title"]}\n\n'+ '\n'.join('- '+b for b in x['bullets'])+'\n\nPresenter notes: '+x['notes'] for i,x in enumerate(spec,1))
    (REPORT/'presentation_notes.md').write_text('# Presentation and speaker notes\n\n'+notes+'\n')
    print(f'Built {len(spec)} slides: PPTX, PDF, HTML and notes')


if __name__=='__main__':main()
