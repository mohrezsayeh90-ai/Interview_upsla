#!/usr/bin/env python3
"""Make a concise editable comparison presentation with matched PDF export."""
import html
import json
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from weasyprint import HTML

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'parametric_pinn';R=B/'results';D=B/'reports'


def main():
    m=json.loads((R/'training_manifest.json').read_text())
    s=pd.read_csv(R/'test_summary.csv')
    a=s[(s.split=='scenario')&(s.method=='pinn_accurate')].iloc[0]
    fast=s[(s.split=='scenario')&(s.method=='pinn_fast')].iloc[0]
    untrained=s[(s.split=='scenario')&(s.method=='untrained_control')].iloc[0]
    timing=pd.read_csv(R/'timing_benchmark.csv')
    time_summary=timing.groupby(['method','batch_size']).median(numeric_only=True)
    ref_ms=float((timing.sparse_g_seconds/timing.cases*1000).median())
    spec=[]
    def add(title,subtitle,bullets,image=None,rows=None,notes=''):
        spec.append(dict(title=title,subtitle=subtitle,bullets=bullets,image=image,rows=rows,notes=notes))
    add('A reusable parametric graph PINN','Independent comparison with the verified random-network solver',[
        f'Main test: {int(a.cases):,} unseen realizations; {int(a.positive_cases):,} carry positive flow.',
        f'K=2048: mean relative conductance error {100*a.positive_mean_abs_relative_error:.4f}%.',
        'Weights are reused. Each input still receives fixed iterative physics updates.',
        'Completed study • 04 November 2026'],image='02_accuracy')
    add('Physical scalars do not identify a random network','The full conductance field, or a seed, must be part of the input',[
        'The same δ,p,σ can produce either a spanning or disconnected network.',
        'For a particular realization, input all 1,200 edge conductances.',
        'For only δ and σ, sample fields and predict ensemble statistics.',
        'Boundary heads and uniform conductance scaling use exact physical factors.'],rows=[
        ['Input','Meaning'],['δ, p, σ','Distribution descriptors / neural conditioning'],
        ['g field or random seed','Identifies the actual physical realization'],
        ['h_left, h_right','Prescribed heads'],['Global conductance scale','Exact output scaling'],
        ['Output','Head field, energy G estimate, flux diagnostics']])
    add('A discrete physics-informed architecture','Learned encoder + parametric coefficients + fixed graph layers',[
        'Component analysis handles zero-flow and fixed-head components exactly.',
        'Resistance Dijkstra distances initialize head.',
        'Small neural networks learn initial correction and graph-level α,β.',
        'K residual layers update h; no new sparse solve or weight fitting.'],rows=[
        ['Block','Definition'],['Node encoder','21 → 32 → 32 → 1'],
        ['Coefficient MLP','6 → 16 → 2'],['Physics layer','h_next = h − α(h−Ph) + β(h−h_prev)'],
        ['Training objective','log Dirichlet energy ratio + residual term'],
        ['Trainable parameters',f'{m["parameters"]:,}']],
        notes='This is a custom unrolled graph PINN / learned iterative operator, not the original continuum coordinate PINN architecture. Fixed head updates are performed at every inference.')
    add('Train once without reference solution labels','4,096 spanning training fields; 512 independent validation fields',[
        'Reference solutions are used for validation and tests only.',
        f'16 epochs; selected epoch {m["best_epoch"]}; depth curriculum 32→64→128→256.',
        f'Training loop {m["training_seconds"]:.1f}s; total offline work {m["elapsed_seconds"]:.1f}s.',
        'Training, validation and test random streams are disjoint.'],image='01_training')
    add('Accuracy improves with trained physical layers','All relative conductance errors below exclude exact zero-flow cases',[
        f'K=256 trained mean error: {100*fast.positive_mean_abs_relative_error:.4f}%.',
        f'Untrained K=256 mean error: {100*untrained.positive_mean_abs_relative_error:.4f}%.',
        f'K=2048: mean {100*a.positive_mean_abs_relative_error:.4f}%, p95 {100*a.positive_p95_abs_relative_error:.4f}%.',
        f'Maximum K=2048 error: {100*a.positive_max_abs_relative_error:.3f}%; accuracy is not uniform.'],image='02_accuracy',
        notes='The validation target of a 99th-percentile error below 0.1% was not reached within the depth cap. The target failure is recorded; no test-based retuning was done.')
    add('Energy accuracy and field accuracy differ','A deliberately difficult held-out head-field example',[
        'G is estimated from Dirichlet energy, an admissible-head upper bound.',
        'Approximate inlet flux can differ from the energy estimate.',
        f'Mean free-node head RMSE: {a.mean_free_node_rmse:.6f}.',
        f'95th-percentile inlet/outlet imbalance: {100*a.p95_relative_in_out_imbalance:.2f}%.'],image='03_head_fields',
        notes='Local weakly connected branches can have significant head errors while contributing little energy. Raw flux and conservation diagnostics are retained.')
    add('Paired uncertainty comparison','Same physical fields for sparse reference and neural prediction',[
        'Nine scenarios × 1,000 new realizations each.',
        'Compare mean, standard deviation, median, upper quantile and conditional flow.',
        'P(G=0) agrees exactly through structural connectivity preprocessing.',
        'Old 10,000-sample study and new reference means differ by Monte Carlo variation.'],image='04_uncertainty_comparison')
    add('Sensitivity rankings are reproduced','2,048 independent base pairs; 14,336 matched physical fields',[
        'Five groups: δ, σ, global-p latent U, topology W and conductance Z.',
        '1,000 paired bootstrap resamples preserve input coupling.',
        'Compare model-minus-reference indices on the same fields.',
        'Total indices overlap; their sums are not independent variance shares.'],image='05_sensitivity_comparison')
    add('Reusable weights do not guarantee a speedup','Measured warm CPU cost includes graph preprocessing and tensor assembly',[
        f'Sparse G-only solver: {ref_ms:.3f} ms per realization.',
        f'Fast PINN, batch 64: {time_summary.loc[("pinn_fast",64),"milliseconds_per_case"]:.3f} ms per realization.',
        f'Accurate PINN, batch 64: {time_summary.loc[("pinn_accurate",64),"milliseconds_per_case"]:.3f} ms per realization.',
        'For this 25×25 problem, direct solving is faster; no positive break-even was observed.'],image='06_cost_accuracy',
        notes='Two CPU threads, float64 inference, three repetitions on the same 64 new fields. Model loading and field generation are excluded. Offline training is additional.')
    ood=s[(s.split=='ood_sigma3')&(s.method=='pinn_accurate')].iloc[0]
    add('Generalization has boundaries','Stress tests do not imply a guarantee outside the training domain',[
        f'At σ=3, mean relative G error rises to {100*ood.positive_mean_abs_relative_error:.2f}%.',
        f'The σ=3 maximum error is {100*ood.positive_max_abs_relative_error:.2f}%.',
        'Other tests: p=1 and n=15,41; same learned weights, no refitting.',
        'Strong contrast and weak anchoring can require much larger depth.'],image='07_generalization_physics')
    add('Use the trained model directly','The default prediction path does not call the sparse solver',[
        'Select fast K=256 or accurate K=2048.',
        'Provide a .npy conductance field, or distribution parameters plus a seed.',
        'Use multiple draws to obtain stochastic summaries.',
        'Keep inlet flux diagnostics separate from the energy-derived G estimate.'],rows=[
        ['Task','Command from repository root'],
        ['One realization','python parametric_pinn/scripts/predict.py --p .6 --sigma 1'],
        ['Ensemble','.../predict.py --delta .25 --sigma 1 --draws 1000'],
        ['Supplied field','.../predict.py --conductances field.npy'],
        ['Optional reference check','.../predict.py --compare-reference'],
        ['Complete reproduction','make -C parametric_pinn all']])
    add('Conclusions and separate deliverables','36 automated tests; reproducible fields, checkpoints and paired comparisons',[
        'The graph PINN reuses trained parameters across unseen random fields.',
        'High in-domain conductance accuracy does not imply exact local conservation.',
        'The direct solver retains the advantage in precision and measured CPU cost here.',
        'All new outputs are isolated under parametric_pinn/.'],rows=[
        ['Artifact','Location'],['Reports','reports/comparison_fa.pdf; comparison_en.pdf'],
        ['Trained weights','models/parametric_pinn.pt'],['Paired results and timings','results/'],
        ['Executed notebook','notebooks/comparison.ipynb'],['API and reproduction','README.md; scripts/']],
        notes='References: Raissi et al. (2019), doi:10.1016/j.jcp.2018.10.045; E and Yu (2018), doi:10.1007/s40304-018-0127-z; Jansen (1999), doi:10.1016/S0010-4655(98)00154-4. See reports for full context.')
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    def text(slide,x,y,w,h,content,size=19,color='23394B',bold=False):
        tf=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame;tf.word_wrap=True
        for i,line in enumerate(content.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
            p.font.name='DejaVu Sans';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(11)
    pages=[]
    for i,item in enumerate(spec,1):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        text(slide,.4,.25,12.5,.65,item['title'],28,'176B87',True)
        text(slide,.43,.99,12.5,.6,item['subtitle'],15,'607681')
        if item['image']:
            slide.shapes.add_picture(str(D/'figures'/(item['image']+'.png')),Inches(.35),Inches(1.95),width=Inches(8.05))
            left=f'<img src="figures/{item["image"]}.png">'
        else:
            rows=item['rows'];shape=slide.shapes.add_table(len(rows),len(rows[0]),Inches(.4),Inches(1.8),Inches(7.8),Inches(4.55))
            for ri,row_ in enumerate(rows):
                for ci,value in enumerate(row_):
                    cell=shape.table.cell(ri,ci);cell.text=str(value);cell.fill.solid()
                    cell.fill.fore_color.rgb=RGBColor.from_string('176B87' if ri==0 else ('EDF5F7' if ri%2 else 'FFFFFF'))
                    for p in cell.text_frame.paragraphs:
                        p.font.name='DejaVu Sans';p.font.size=Pt(14);p.font.color.rgb=RGBColor.from_string('FFFFFF' if ri==0 else '23394B')
            left='<table>'+''.join('<tr>'+''.join(f'<{"th" if ri==0 else "td"}>{html.escape(str(v))}</{"th" if ri==0 else "td"}>' for v in row_)+'</tr>' for ri,row_ in enumerate(rows))+'</table>'
        text(slide,8.6,1.75,4.2,5.05,'\n'.join('• '+b for b in item['bullets']),18)
        text(slide,.43,7.06,11.7,.25,'PARAMETRIC GRAPH PINN  •  04 NOV 2026  •  Full error distributions and physical limits in the reports',9,'607681')
        text(slide,12.4,7.06,.5,.25,str(i),10,'176B87')
        slide.notes_slide.notes_text_frame.text=item['notes']
        pages.append(f'<section><h1>{html.escape(item["title"])}</h1><p class="sub">{html.escape(item["subtitle"])}</p><div class="layout"><div class="left">{left}</div><ul>'+''.join(f'<li>{html.escape(b)}</li>' for b in item['bullets'])+f'</ul></div><footer>PARAMETRIC GRAPH PINN · 04 NOV 2026 <span>{i}</span></footer></section>')
    prs.save(D/'comparison_slides.pptx')
    css="""@page{size:13.333333in 7.5in;margin:0}body{margin:0;font-family:'DejaVu Sans';color:#23394b}section{height:7.5in;padding:.3in .45in;position:relative;box-sizing:border-box;break-after:page}section:last-child{break-after:auto}h1{font-size:28pt;color:#176b87;margin:0 0 .12in}.sub{font-size:15pt;color:#607681;margin:0}.layout{display:flex;align-items:center;gap:.3in;margin-top:.4in;height:4.95in}.left{width:7.65in;flex-shrink:0}.left img{width:100%}ul{font-size:18pt;line-height:1.35;margin:0;padding-left:.25in}li{margin-bottom:.22in}table{border-collapse:collapse;width:100%;font-size:13.5pt}th,td{padding:.15in .1in;border-bottom:1px solid #c5dce4}th{background:#176b87;color:white;text-align:left}tr:nth-child(even){background:#edf5f7}footer{position:absolute;bottom:.2in;font-size:9pt;color:#607681;left:.45in;right:.45in}footer span{float:right} """
    path=D/'comparison_slides.html';path.write_text('<!doctype html><meta charset="utf-8"><title>PINN comparison slides</title><style>'+css+'</style>'+''.join(pages))
    HTML(filename=str(path)).write_pdf(D/'comparison_slides.pdf')
    (D/'speaker_notes.md').write_text('# Parametric graph PINN comparison\n\n'+'\n\n'.join(f'## {i}. {item["title"]}\n\n'+ '\n'.join('- '+b for b in item['bullets'])+'\n\n'+item['notes'] for i,item in enumerate(spec,1))+'\n')
    print(f'Built {len(spec)} comparison slides, editable PPTX and PDF')


if __name__=='__main__':main()
