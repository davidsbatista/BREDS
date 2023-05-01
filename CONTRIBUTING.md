# Contributing to BREDS

If you are using BREDS probably your experience and what you can contribute are important to the project's success.


```sh
git clone https://github.com/breds/
cd breds
python3.9 -m virtualenv venv  # make sure you have python 3.9 installed
source venv/bin/activate  # install BREDS in edit mode and with development dependencies
pip install -e .[dev]
```


## The contribution process at a glance

1. [Prepare your environment](#preparing-the-environment).
2. Find out [where to make your changes](#where-to-make-changes).
3. [Prepare your changes](#preparing-changes):
    * Small fixes and additions can be submitted directly as pull requests,
      but [contact us](README.md#discussion) before starting significant work.
    * Create your stubs, considering [what to include](#what-to-include) and
      conforming to the [coding style](#stub-file-coding-style).
4. Optionally [format and check your stubs](#code-formatting).
5. Optionally [run the tests](tests/README.md).
6. [Submit your changes](#submitting-changes) by opening a pull request.

You can expect a reply within a few days, but please be patient if it takes a bit longer.

## Preparing the environment

### Code away!

BREDS runs a continuous integration (CI) on all pull requests. This means that  if you file a pull request (PR), a full 
test suite is run on your PR: 

-- including our linter
`flake8` -- . 

It also means that bots will automatically apply  changes to your PR (using `pycln`, `black` and `isort`) to fix any formatting issues.
This frees you up to ignore all local setup on your side, focus on the  code and rely on the CI to fix everything, or point you to the places that  need fixing.

### ... Or create a local development environment

If you prefer to run the tests & formatting locally, it's  possible too. Follow platform-specific instructions below.
For more information about our available tests, see
[tests/README.md](tests/README.md).

Whichever platform you're using, you will need a
virtual environment. If you're not familiar with what it is and how it works,
please refer to this
[documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Note that some tests require extra setup steps to install the required dependencies.



## Code formatting

The code is formatted using `black` and `isort`. Unused imports are also auto-removed using `pycln`.


if you *want* to run the checks locally when you commit  you're free to do so. Either run `pycln`, `black` and `isort` manually...

```bash
$ pycln --config=pyproject.toml .
$ isort .
$ black .
```

...Or install the pre-commit hooks: please refer to the
[pre-commit](https://pre-commit.com/) documentation.

Our code is also linted using `flake8`, with plugins `flake8-pyi`,
`flake8-bugbear`, and `flake8-noqa`. As with our other checks, running
flake8 before filing a PR is not required. However, if you wish to run flake8
locally, install the test dependencies as outlined above, and then run: