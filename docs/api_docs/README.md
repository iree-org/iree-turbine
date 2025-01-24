# API Docs

This directory uses Sphinx to build API documentation.

## Building the API documentation locally

### Setup virtual environment with requirements

From this docs/api_docs/ directory:

```shell
python -m venv .venv
source .venv/bin/activate

# Install sphinx website generator requirements and PyTorch dep.
python -m pip install -r requirements.txt

# Install iree-turbine itself.
# Editable so you can make local changes and preview them easily
python -m pip install -e ../..
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
