# Releasing iree-turbine

This project hosts the https://pypi.org/project/iree-turbine/ package, which
depends on the https://pypi.org/project/iree-base-compiler/ and
https://pypi.org/project/iree-base-runtime/ packages. Releases can either be
conducted independently, or they can be coordinated across projects by
initiating a release here.

## Start with a clean test directory

```bash
rm -rf wheelhouse/
```

## Building Artifacts

Build a dev release (e.g. `3.1.0.dev+6879a433eecc1e0b2cdf6c6dbcad901c77d97ac8`):

```bash
python3.11 ./build_tools/compute_local_version.py -dev --write-json
python3.11 ./build_tools/build_release.py
```

Build a release candidate (e.g. `3.1.0rc20241204`):

```bash
python3.11 ./build_tools/compute_local_version.py -rc --write-json
python3.11 ./build_tools/build_release.py
```

Build an official release (e.g. `3.1.0`):

```bash
python3.11 ./build_tools/compute_local_version.py -stable --write-json
python3.11 ./build_tools/build_release.py
```

This will download all deps, including wheels for all supported platforms and
Python versions for iree-base-compiler and iree-base-runtime. All wheels will
be placed in the `wheelhouse/` directory.

If you just want to build without downloading wheels, run

```bash
python3.11 ./build_tools/build_release.py --no-download
# Note that the test scripts referenced below won't work with this.
```

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
