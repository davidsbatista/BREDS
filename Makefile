.PHONY:	test lint virtualenv dist

lint:
	black -t py39 -l 120 breds tests
	pycln -a breds tests
	isort --profile black breds tests
	PYTHONPATH=. pylint --rcfile=pylint.cfg breds
	PYTHONPATH=. flake8 --config=setup.cfg breds


typing:
	mypy --config mypy.ini -p breds


test:
	PYTHONPATH=. coverage run --rcfile=setup.cfg --source=./breds -m pytest
	PYTHONPATH=. coverage report --rcfile=setup.cfg


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info


publish:
	make all
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*


all:
	make clean
	make lint
	make typing
	make test