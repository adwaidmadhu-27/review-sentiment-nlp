.PHONY: install data test train predict clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

test:
	pytest -q

train: data
	PYTHONPATH=src python -m sentiment.pipeline

predict:
	PYTHONPATH=src python -m sentiment.predict "$(TEXT)"

clean:
	rm -rf models .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
