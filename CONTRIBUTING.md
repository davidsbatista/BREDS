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

#### The `METADATA.toml` file

The metadata file describes the stubs package using the
[TOML file format](https://toml.io/en/). Currently, the following keys are
supported:

* `version`: The versions of the library that the stubs support. Two
  formats are supported:
    - A concrete version. This is especially suited for libraries that
      use [Calendar Versioning](https://calver.org/).
    - A version range ending in `.*`. This is suited for libraries that
      reflect API changes in the version number only, where the API-independent
      part is represented by the asterisk. In the case
      of [Semantic Versioning](https://semver.org/), this version could look
      like this: `2.7.*`.
  When the stubs are updated to a newer version
  of the library, the version of the stub should be bumped (note that
  previous versions are still available on PyPI).
* `requires` (optional): A list of other stub packages or packages with type
  information that are imported by the stubs in this package. Only packages
  generated by typeshed or required by the upstream package are allowed to
  be listed here, for security reasons.
* `extra_description` (optional): Can be used to add a custom description to
  the package's long description. It should be a multi-line string in
  Markdown format.
* `stub_distribution` (optional): Distribution name to be uploaded to PyPI.
  This defaults to `types-<distribution>` and should only be set in special
  cases.
* `obsolete_since` (optional): This field is part of our process for
  [removing obsolete third-party libraries](#third-party-library-removal-policy).
  It contains the first version of the corresponding library that ships
  its own `py.typed` file.
* `no_longer_updated` (optional): This field is set to `true` before removing
  stubs for other reasons than the upstream library shipping with type
  information.
* `upload` (optional): This field is set to `false` to prevent automatic
  uploads to PyPI. This should only used in special cases, e.g. when the stubs
  break the upload.

In addition, we specify configuration for stubtest in the `tool.stubtest` table.
This has the following keys:
* `skip` (default: `false`): Whether stubtest should be run against this
  package. Please avoid setting this to `true`, and add a comment if you have
  to.
* `apt_dependencies` (default: `[]`): A list of Ubuntu APT packages
  that need to be installed for stubtest to run successfully.
* `brew_dependencies` (default: `[]`): A list of MacOS Homebrew packages
  that need to be installed for stubtest to run successfully
* `choco_dependencies` (default: `[]`): A list of Windows Chocolatey packages
  that need to be installed for stubtest to run successfully
* `platforms` (default: `["linux"]`): A list of OSes on which to run stubtest.
  Can contain `win32`, `linux`, and `darwin` values.
  If not specified, stubtest is run only on `linux`.
  Only add extra OSes to the test
  if there are platform-specific branches in a stubs package.

`*_dependencies` are usually packages needed to `pip install` the implementation
distribution.

The format of all `METADATA.toml` files can be checked by running
`python3 ./tests/check_consistent.py`.


## Preparing Changes

### Before you begin

If your change will be a significant amount of work to write, we highly
recommend starting by opening an issue laying out what you want to do.
That lets a conversation happen early in case other contributors disagree
with what you'd like to do or have ideas that will help you do it.

### Format

Each Python module is represented by a `.pyi` "stub file".  This is a
syntactically valid Python file, although it usually cannot be run by
Python 3 (since forward references don't require string quotes).  All
the methods are empty.

Python function annotations ([PEP 3107](https://www.python.org/dev/peps/pep-3107/))
are used to describe the signature of each function or method.

See [PEP 484](http://www.python.org/dev/peps/pep-0484/) for the exact
syntax of the stub files and [below](#stub-file-coding-style) for the
coding style used in typeshed.

### Auto-generating stub files

Typeshed includes `scripts/create_baseline_stubs.py`.
It generates stubs automatically using a tool called
[stubgen](https://mypy.readthedocs.io/en/latest/stubgen.html) that comes with mypy.

To get started, fork typeshed, clone your fork, and then
[create a virtualenv](#-or-create-a-local-development-environment).
You can then install the library with `pip` into the virtualenv and run the script,
replacing `libraryname` with the name of the library below:

```bash
(.venv3)$ pip install libraryname
(.venv3)$ python3 scripts/create_baseline_stubs.py libraryname
```

When the script has finished running, it will print instructions telling you what to do next.

If it has been a while since you set up the virtualenv, make sure you have
the latest mypy (`pip install -r requirements-tests.txt`) before running the script.
