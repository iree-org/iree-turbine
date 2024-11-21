# AOT MLP Example

This example demonstrates export, compilation, and inference of 
a simple Multi-Layer Perceptron (MLP) model. 
The model is a four-layer neural network.

To run this example, you should clone the repository to your local device and
install the requirements in a virtual environment:

```bash
git clone https://github.com/iree-org/iree-turbine.git
cd iree-turbine/examples/aot_mlp
python -m venv mlp.venv
source ./mlp.venv/bin/activate
pip install -r requirements.txt
```

Once the requirements are installed, you should be able to run the example.

```bash
python mlp_export_simple.py
```

