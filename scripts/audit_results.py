#!/usr/bin/env python3
"""Audit saved results, validate deliverables and write a SHA-256 inventory."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    res=ROOT/'results'
    m=json.loads((res/'manifest.json').read_text())
    samples=pd.read_csv(res/'scenario_samples.csv.gz')
    summary=pd.read_csv(res/'scenario_summary.csv')
    assert len(samples)==9*m['arguments']['scenario_reps']
    assert np.isfinite(samples.geff).all() and (samples.geff>=0).all()
    assert ((samples.geff>0)==samples.spanning).all()
    assert samples.groupby(['replicate','delta']).spanning.nunique().max()==1
    for _,row in summary.iterrows():
        sub=samples[(samples.delta==row.delta)&(samples.sigma==row.sigma)]
        np.testing.assert_allclose(sub.geff.mean(),row['mean'],rtol=1e-12)
        np.testing.assert_allclose(sub.spanning.mean(),row.p_span,atol=1e-14)
        np.testing.assert_allclose(sub.geff_unit_mean,sub.geff*np.exp(-row.sigma**2/2),atol=1e-14)
        assert row.mean_ci_low<=row['mean']<=row.mean_ci_high
        assert row.q95_low<=row.q95<=row.q95_high
    span=pd.read_csv(res/'sobol_spanning_samples.csv.gz')
    assert (span.A==span.AB_sigma).all()
    assert (span.A==span.AB_conductance_field).all()
    diag=pd.read_csv(res/'solver_diagnostics.csv')
    assert diag.max_residual.max()<1e-9
    assert diag.relative_balance_error.max()<1e-7
    assert diag.energy_error.max()<1e-9
    assert diag.min_head.min()>-1e-9 and diag.max_head.max()<1+1e-9
    # PDF and PPTX validation is optional for users running model-only installs.
    try:
        from pypdf import PdfReader
        assert len(PdfReader(ROOT/'reports/report_fa.pdf').pages)==7
        assert len(PdfReader(ROOT/'reports/report_en.pdf').pages)==7
        assert len(PdfReader(ROOT/'reports/presentation_en.pdf').pages)==12
    except ImportError:
        pass
    import nbformat
    nb=nbformat.read(ROOT/'notebooks/analysis.ipynb',as_version=4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    report={"status":"passed","checks":["raw sample counts and nonnegativity","spanning/zero-flow consistency", "summary reproduction", "exact constant-mean scaling", "conductance-independent spanning", "solver conservation and energy", "head bounds", "executed notebook"],"solver_calls":m['solver_calls']}
    (res/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    excluded={'.git','__pycache__','.pytest_cache','.venv'}
    inventory=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name=='SHA256SUMS' or any(part in excluded or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts):
            continue
        inventory.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(ROOT).as_posix())
    (ROOT/'SHA256SUMS').write_text('\n'.join(inventory)+'\n')
    print(f'Audit passed; checksums written for {len(inventory)} files')


if __name__=='__main__':main()
