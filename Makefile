.PHONY: install test check

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

check:
	python -m compileall -q src train evaluation audit robustness data feature_extraction semantic_anchors figures tools
	pytest -q
