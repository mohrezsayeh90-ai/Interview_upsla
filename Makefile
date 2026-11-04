.NOTPARALLEL:
.PHONY: all test study figures reports slides notebook audit
RUN = PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python

all: test study figures reports slides notebook audit

test:
	$(RUN) -m pytest -q

study:
	$(RUN) scripts/run_study.py

figures:
	$(RUN) scripts/make_figures.py

reports:
	$(RUN) scripts/make_reports.py

slides:
	$(RUN) scripts/make_presentation.py

notebook:
	$(RUN) scripts/make_notebook.py

audit:
	$(RUN) scripts/audit_results.py
