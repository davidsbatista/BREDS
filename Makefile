.PHONY:	test lint virtualenv dist

lint:
	black -t py39 -l 120 src tests
	pycln -a src tests
	isort --profile black src tests
	pylint --rcfile=pylint.cfg src
	flake8 --config=setup.cfg src


typing:
	MYPYPATH=src/ mypy --config mypy.ini src


test:
	PYTHONPATH=src/ coverage run --rcfile=setup.cfg --source=./src -m pytest
	PYTHONPATH=src/ coverage report --rcfile=setup.cfg


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info


all:
	make clean
	make lint
	make typing
	make test