.PHONY:	test lint

lint:
	black --check -t py39 -l 120 src
	pylint --rcfile=pylint.cfg src
	flake8 --config=setup.cfg src
	MYPYPATH=src mypy --config mypy.ini src

test:
	coverage run --rcfile=setup.cfg --source=./src -m pytest
	coverage report --rcfile=setup.cfg
