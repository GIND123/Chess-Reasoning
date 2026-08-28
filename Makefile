.PHONY: install test audit tables generate filter sft grpo eval smoke clean

install:
	pip install -e ".[train,dev]"

test:
	pytest -q

audit:                       ## regression-test the verifier against the legacy corpus
	python scripts/00_validate_extractor.py --audit

label:                       ## emit 200 claims for hand-labelling (gate 2)
	python scripts/00_validate_extractor.py --sample data/interim/labels.tsv

tables:
	python scripts/01_engine_tables.py --workers $${WORKERS:-8}

generate:
	python scripts/02_generate.py --n 1

retry:                       ## pass 2: only the rejects, higher temperature
	python scripts/02_generate.py --n 4 --temperature 1.0 --only-fens data/interim/rejects.jsonl

filter:
	python scripts/03_filter.py

sft:
	python scripts/04_sft.py --config configs/sft.yaml

grpo-all:                    ## every arm of the comparison programme
	for c in m6 m3 m4 m2 a3; do python scripts/05_grpo.py --config configs/grpo_$$c.yaml; done

eval:
	python scripts/06_evaluate.py --preds runs/preds_m6.jsonl

smoke:                       ## week-1 gate on Modal
	modal run modal_app.py::smoke

clean:
	rm -rf data/interim/* runs/*
