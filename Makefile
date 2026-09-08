.PHONY: install setup test lint format clean train infer app demo

install:
	pip install -e .

setup:
	python3 -m satellite_srm setup

test:
	python3 -m unittest discover tests

lint:
	python3 -m flake8 src tests || true

format:
	python3 -m black src tests || true

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache outputs/sr/* outputs/uncertainty/* outputs/validation/*

train:
	python3 -m satellite_srm train --config configs/training.yaml

infer:
	python3 -m satellite_srm infer --input data/example/input.tif --output outputs/sr/SR_product.tif

app:
	python3 -m satellite_srm app

demo:
	python3 -m satellite_srm demo
