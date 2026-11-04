#!/usr/bin/env python3
"""Rebuild saved-data figures, bilingual reports and comparison slides."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from parametric_pinn.scripts.make_figures import make_figures
from parametric_pinn.scripts.make_reports import main as reports
from parametric_pinn.scripts.make_presentation import main as slides

if __name__=='__main__':
    make_figures();reports();slides()
