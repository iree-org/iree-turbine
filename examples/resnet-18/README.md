# Dynamic AOT Resnet-18 Example

This example AOT-compiles a Resnet-18 module for performing inference on a
dynamic number of input images.

To run this example, you should clone the repository to your local device and
install the requirements in a virtual environment:

```bash
git clone https://github.com/iree-org/iree-turbine.git
cd iree-turbine/examples/resnet-18
python -m venv rn18.venv
source ./rn18.venv/bin/activate
pip install -r requirements.txt
```

Once the requirements are installed, you should be able to run the example.

```bash
python resnet-18.py
```
