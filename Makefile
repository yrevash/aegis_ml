# aegis_ml — developer entry points.
#
# Every target here is also a documented command in docs/. If you change one, change the
# doc: on hackathon day the docs are what an agent reads, and a command that does not run
# is worse than no command at all.

PY      := .venv/bin/python
UV      := uv
VENV    := .venv
VENV_ML := .venv-ml

.PHONY: help venv venv-ml install install-strong doctor audit lint test demo clean

help:
	@echo "aegis_ml targets"
	@echo "  make install        serving venv: everything that co-installs with Aegis"
	@echo "  make install-strong trainer venv: AutoGluon + TabPFN + torch, isolated"
	@echo "  make doctor         environment report — run this FIRST on hackathon day"
	@echo "  make audit          fail if src/ contains mocks, stubs or swallowed imports"
	@echo "  make lint           ruff, same config as Aegis"
	@echo "  make test           pytest"
	@echo "  make demo           full pipeline on the reference domain, end to end"

venv:
	$(UV) venv $(VENV) --python 3.11

install: venv
	$(UV) pip install --python $(VENV) -e '.[dev]'

# The heavy half, deliberately isolated. AutoGluon/TabPFN/torch will not resolve under the
# backend's pandas<2.4 / numpy<2.5 / numba==0.67.0 caps — see decision D1 in finalplan.md.
venv-ml:
	$(UV) venv $(VENV_ML) --python 3.11

install-strong: venv-ml
	$(UV) pip install --python $(VENV_ML) -e '.[strong,serve]'

doctor:
	$(PY) -m aegis_ml.cli doctor

audit:
	$(PY) scripts/audit_no_mocks.py

lint:
	$(UV) run --python $(VENV) ruff check src scripts tests reference

test:
	$(PY) -m pytest tests -q

demo:
	$(PY) -m aegis_ml.cli doctor
	$(PY) scripts/run_demo.py

clean:
	rm -rf registry_store/runs registry_store/index.json registry_store/reports
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
