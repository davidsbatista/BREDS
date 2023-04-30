# Contributing to BREDS

If you are using BREDS probably your experience and what you can contribute are important to the project's success.


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

You can expect a reply within a few days, but please be patient when
it takes a bit longer. For more details, read below.

## Preparing the environment

### Code away!

Typeshed runs continuous integration (CI) on all pull requests. This means that
if you file a pull request (PR), our full test suite -- including our linter,
`flake8` -- is run on your PR. It also means that bots will automatically apply
changes to your PR (using `pycln`, `black` and `isort`) to fix any formatting issues.
This frees you up to ignore all local setup on your side, focus on the
code and rely on the CI to fix everything, or point you to the places that
need fixing.

### ... Or create a local development environment

If you prefer to run the tests & formatting locally, it's
possible too. Follow platform-specific instructions below.
For more information about our available tests, see
[tests/README.md](tests/README.md).

Whichever platform you're using, you will need a
virtual environment. If you're not familiar with what it is and how it works,
please refer to this
[documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Note that some tests require extra setup steps to install the required dependencies.

### Linux/Mac OS

On Linux and Mac OS, you will be able to run the full test suite on Python
3.9 or 3.10.
To install the necessary requirements, run the following commands from a
terminal window:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements-tests.txt
```

### Windows

If you are using a Windows operating system, you will not be able to run the pytype
tests, as pytype
[does not currently support running on Windows](https://github.com/google/pytype#requirements).
One option is to install
[Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/faq),
which will allow you to run the full suite of tests. If you choose to install
WSL, follow the Linux/Mac OS instructions above.

If you do not wish to install WSL, run the following commands from a Windows
terminal to install all non-pytype requirements:

```powershell
> python -m venv .venv
> .venv\scripts\activate
(.venv) > pip install -U pip
(.venv) > pip install -r "requirements-tests.txt"
```

## Code formatting

The code is formatted using `black` and `isort`. Unused imports are also auto-removed using `pycln`.

The repository is equipped with a [`pre-commit.ci`](https://pre-commit.ci/)
configuration file. This means that you don't *need* to do anything yourself to
run the code formatters. When you push a commit, a bot will run those for you
right away and add a commit to your PR.

That being said, if you *want* to run the checks locally when you commit,
you're free to do so. Either run `pycln`, `black` and `isort` manually...

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

```bash
(.venv3)$ flake8 .
```

## Where to make changes


### Third-party library stubs

We accept stubs for third-party packages into typeshed as long as:
* the package is publicly available on the [Python Package Index](https://pypi.org/);
* the package supports any Python version supported by typeshed; and
* the package does not ship with its own stubs or type annotations.

The fastest way to generate new stubs is to use `scripts/create_baseline_stubs.py` (see below).

Stubs for third-party packages
go into `stubs`. Each subdirectory there represents a PyPI distribution, and
contains the following:
* `METADATA.toml`, describing the package. See below for details.
* Stubs (i.e. `*.pyi` files) for packages and modules that are shipped in the
  source distribution.
* (Rarely) some docs specific to a given type stub package in `README` file.

When a third party stub is added or
modified, an updated version of the corresponding distribution will be
automatically uploaded to PyPI within a few hours.
Each time this happens the least significant
version level is incremented. For example, if `stubs/foo/METADATA.toml` has
`version = "x.y"` the package on PyPI will be updated from `types-foo-x.y.n`
to `types-foo-x.y.n+1`.

*Note:* In its current implementation, typeshed cannot contain stubs for
multiple versions of the same third-party library.  Prefer to generate
stubs for the latest version released on PyPI at the time of your
stubbing.
