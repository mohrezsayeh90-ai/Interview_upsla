.NOTPARALLEL:
.PHONY: all test train evaluate report notebook audit
ROOT = ..
RUN = PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python

all: test train evaluate report notebook audit

test:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/verify.py

train:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/train.py

evaluate:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/evaluate.py

report:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/make_deliverables.py

notebook:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/make_notebook.py

audit:
	cd $(ROOT) && $(RUN) parametric_pinn/scripts/audit.py
