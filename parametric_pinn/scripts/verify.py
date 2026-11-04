#!/usr/bin/env python3
"""Run both test suites and preserve machine-readable actual test counts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='pinn-verification-') as temp:
        path=Path(temp)/'tests.xml'
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','parametric_pinn/tests',f'--junitxml={path}'],cwd=ROOT)
        if result.returncode:sys.exit(result.returncode)
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        counts={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','errors','failures','skipped']}
        target=ROOT/'parametric_pinn/results/verification.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(counts,indent=2)+'\n')
        print(json.dumps(counts))
