# Setting up TK-Wave for development

Welcome! This page outlines a guide to get started with developing for Turbine Kernel (TK) Wave.

## Setting up

For setting up necessary environments and tools, first follow the instructions under "Developers" in the main [README](../README.md). Specifically, while in the root `iree-turbine` directory, do the following:

### Create a virtual environment

This will consolidate all necessary python packages for your project in one folder. GitHub will ignore the virtual environment if you name it `.venv` (see `.gitignore`).

```
python -m venv --prompt iree-turbine .venv #Create the virtual environment
source .venv/bin/activate #Activate the virtual environment. Must run any time you start a new terminal.
```

### Installing PyTorch

PyTorch is our friendly frontend for AI work, and we'll be using it to compare our computational results with their "golden standard".

*For CPU:*

```
pip install -r pytorch-cpu-requirements.txt
```

*For ROCM:*

```
pip install -r pytorch-rocm-requirements.txt
```

### Install Development Packages

Finally, install all other required Python packages.

```
pip install -r requirements.txt -e .
```

# Testing

Try running `tests/kernel/wave/wave_e2e_test.py`. Even with venv activated and shark_turbine installed, you might still get ModuleNotFoundError: No module named 'shark_turbine'. Me too! But why?

You even try running:

* `kernel/wave/wave_e2e_test.py` from `iree-turbine/sharktank`
* `wave/wave_e2e_test.py` from `iree-turbine/sharktank/kernel`
* `wave_e2e_test.py` from `iree-turbine/sharktank/kernel/wave`

... I need some help
