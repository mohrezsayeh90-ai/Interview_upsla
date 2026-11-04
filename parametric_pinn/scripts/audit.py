#!/usr/bin/env python3
"""Audit frozen checkpoints, paired comparisons, physics constraints and notebook."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nbformat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from parametric_pinn.model import Predictor
from scripts.run_study import jansen,FACTORS


def main():
    b=ROOT/'parametric_pinn';r=b/'results'
    train=json.loads((r/'training_manifest.json').read_text())
    evaluation=json.loads((r/'evaluation_manifest.json').read_text())
    assert train['training_reference_labels'] is False
    assert train['train_data']['stream']!=train['validation_data']['stream']
    assert train['train_data']['stream'] not in evaluation['test_streams'].values()
    assert train['validation_data']['stream'] not in evaluation['test_streams'].values()
    data=pd.read_csv(r/'paired_predictions.csv.gz')
    assert np.isfinite(data.g_pred).all() and (data.g_pred>=0).all()
    assert ((data.g_ref==0)==(data.g_pred==0)).all()
    # Variational upper bound up to reference/roundoff errors.
    assert (data.g_pred>=data.g_ref-1e-8).all()
    assert (data.relative_g_error[data.g_ref==0].isna()).all()
    assert len(data[(data.split=='scenario')&(data.method=='pinn_accurate')])==9*evaluation['arguments']['scenario_reps']
    summary=pd.read_csv(r/'test_summary.csv')
    for _,row in summary.iterrows():
        sub=data[(data.split==row.split)&(data.method==row.method)]
        np.testing.assert_allclose(sub.g_pred.mean(),row.mean_predicted,atol=1e-13)
        np.testing.assert_allclose((sub.g_pred-sub.g_ref).mean(),row.paired_bias,atol=1e-13)
    # Recompute all saved Sobol point estimates from paired output matrices.
    outputs=pd.read_csv(r/'sobol_paired_outputs.csv.gz')
    sensitivity=pd.read_csv(r/'sobol_comparison.csv')
    for method in ['reference','pinn_fast','pinn_accurate']:
        columns=[method+'__'+x for x in ['A','B']+['AB_'+f for f in FACTORS]]
        arr=outputs[columns].to_numpy()
        values=jansen(arr[:,0],arr[:,1],arr[:,2:])
        for order,x in zip(['first','total'],values):
            saved=sensitivity[(sensitivity.method==method)&(sensitivity.order==order)].set_index('factor').loc[FACTORS,'index'].to_numpy()
            np.testing.assert_allclose(x,saved,atol=2e-12)
    predictor=Predictor(b/'models/parametric_pinn.pt',depth=train['accurate_depth'])
    case=np.load(r/'difficult_example.npz')
    metadata=json.loads((r/'difficult_example.json').read_text())
    pred=predictor.predict(case['g'],metadata['delta'],metadata['p'],metadata['sigma'])
    np.testing.assert_allclose(pred['heads'],case['pinn_heads'],atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(pred['geff'],metadata['pinn_geff'],rtol=1e-12)
    nb=nbformat.read(b/'notebooks/comparison.ipynb',as_version=4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    from pypdf import PdfReader
    assert len(PdfReader(b/'reports/comparison_fa.pdf').pages)==9
    assert len(PdfReader(b/'reports/comparison_en.pdf').pages)==9
    assert len(PdfReader(b/'reports/comparison_slides.pdf').pages)==12
    verification=json.loads((r/'verification.json').read_text())
    assert verification['failures']==0 and verification['errors']==0
    report={'status':'passed','trainable_parameters':train['parameters'],
        'paired_rows':len(data),'unique_test_cases':int(data.case_id.nunique()),
        'tests':verification['tests'],'checkpoint_sha256':hashlib.sha256((b/'models/parametric_pinn.pt').read_bytes()).hexdigest(),
        'checks':['disjoint random streams','no training labels','exact zero-flow routing','variational energy bound',
                  'paired summary reproduction','Sobol output reproduction','frozen-checkpoint field reproduction',
                  'executed notebook','PDF page counts','automated test results']}
    (r/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    # Recheck the preserved direct study and refresh the complete file inventory.
    from scripts.audit_results import main as audit_original_and_checksums
    audit_original_and_checksums()


if __name__=='__main__':main()
