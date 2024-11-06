
# Releasing SHARK-Turbine/core

There are multiple release artifacts that are deployed from this project:

* shark-turbine wheel (transitional while switching to iree-turbine)
* iree-turbine wheel
* iree-compiler wheels
* iree-runtime wheels

Typically we deploy IREE compiler and runtime wheels along with a turbine
release, effectively promoting a nightly.

## Building Artifacts

Build a pre-release:

```
./build_tools/build_release.py --core-version 2.3.0 --core-pre-version=rcYYYYMMDD
```

Build an official release:

```
./build_tools/build_release.py --core-version 2.3.0
```

This will download all deps, including wheels for all supported platforms and
Python versions for iree-compiler and iree-runtime. All wheels will be placed
in the `wheelhouse/` directory.


## Testing

TODO: Write a script for this.

```
python -m venv wheelhouse/test.venv
source wheelhouse/test.venv/bin/activate
pip install -f wheelhouse iree-turbine[testing]
# Temp: tests require torchvision.
pip install -f wheelhouse torchvision
pytest core/tests
```

## Push

From the testing venv, verify that everything is sane:

```
pip freeze
```

Push IREE deps (if needed/updated):

```
twine upload wheelhouse/iree_compiler-* wheelhouse/iree_runtime-*
```

Push built wheels:

```
twine upload wheelhouse/iree_turbine-* wheelhouse/shark_turbine-*
```

## Install from PyPI and Sanity Check

TODO: Script this

From the testing venv:

```
pip uninstall -y shark-turbine iree-turbine iree-compiler iree-runtime
pip install iree-turbine
pytest core/tests
```
