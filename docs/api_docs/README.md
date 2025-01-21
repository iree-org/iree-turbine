# API Docs

This directory uses Sphinx to build API documentation.

## Building the API documentation locally

### Setup virtual environment with requirements

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Build docs

```shell
sphinx-build -b html . _build
```

### Serve locally locally with autoreload

```shell
sphinx-autobuild . _build
```

Then open http://127.0.0.1:8000 as instructed by the logs and make changes to
the files in this directory as needed to update the documentation.

### Clean to show all warnings

A clean rebuild will show all warnings again:

```shell
make clean
sphinx-build -b html . _build
```
