.PHONY:	test lint typing clean publish all

lint:
	ruff check breds tests
	ruff format breds tests


typing:
	mypy -p breds


test:
	PYTHONPATH=. coverage run --source=./breds -m pytest
	PYTHONPATH=. coverage report


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache src/*.egg-info


publish:
	make clean
	python -m pip install --upgrade build twine
	python -m build
	python -m twine upload dist/*


all:
	make clean
	make lint
	make typing
	make test
