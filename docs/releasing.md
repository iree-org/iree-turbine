# Releasing iree-turbine

This project hosts the https://pypi.org/project/iree-turbine/ package, which
depends on the https://pypi.org/project/iree-base-compiler/ and
https://pypi.org/project/iree-base-runtime/ packages. Releases can either be
conducted independently, or they can be coordinated across projects by
initiating a release here.

## Building Artifacts

Build a pre-release:

```bash
./build_tools/build_release.py --package-version 2.5.0 --package-pre-version=rcYYYYMMDD
```

Build an official release:

```bash
./build_tools/build_release.py --package-version 2.5.0
```

This will download all deps, including wheels for all supported platforms and
Python versions for iree-base-compiler and iree-base-runtime. All wheels will
be placed in the `wheelhouse/` directory.

## Testing

```bash
./build_tools/post_build_release_test.sh
```

This will

1. Set up a python virtual environment using your default `python` version
2. Install wheels from the `wheelhouse/` directory
3. Run `pytest` tests

## Push

From the testing venv, verify that everything is sane:

```bash
pip freeze
```

Push IREE deps (if needed/updated):

```bash
twine upload wheelhouse/iree_base_compiler-* wheelhouse/iree_base_runtime-*
```

Push built wheels:

```bash
twine upload wheelhouse/iree_turbine-*
```

## Install from PyPI and Sanity Check

From the testing venv:

```bash
./build_tools/post_pypi_release_test.sh
```
