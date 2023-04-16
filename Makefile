.PHONY:	test lint virtualenv dist

PYTHON_VERSION = python3.9
VIRTUALENV := .venv

dev:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements_dev.txt
	source ${VIRTUALENV}/bin/activate && pip3 install --editable .


lint:
	black --check -t py39 -l 120 src
	pylint --rcfile=pylint.cfg src
	flake8 --config=setup.cfg src
	MYPYPATH=src mypy --config mypy.ini src


test:
	coverage run --rcfile=setup.cfg --source=./src -m pytest
	coverage report --rcfile=setup.cfg


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache


dist:
	-rm -r dist
	python -m pip install --upgrade build
	python -m build
