from stats import ErrorAggregatorDict
import logging

from iree.compiler.extras.fx_importer import FxImporter
from shark_turbine.dynamo.passes import turbine_cpu_pass_pipeline
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)


def create_backend():
    imp = FxImporter()

    def import_compiler(gm: GraphModule, example_inputs):
        gm = turbine_cpu_pass_pipeline(gm, example_inputs)

        try:
            imp.import_graph_module(gm)
        finally:
            pass
        imp.module.operation.verify()
        return gm

    backend = import_compiler
    backend = aot_autograd(fw_compiler=backend)
    return backend


def evaluate_importer(nn_cls, get_init_args, get_forward_args, test_identifier):
    log = logging.getLogger("turbine-test")
    try:
        args, kwargs = get_init_args()
        nn_module = nn_cls(*args, **kwargs)
        opt_mod = torch.compile(nn_module, backend=create_backend())

        fargs, fkwargs = get_forward_args()
        opt_mod(*fargs, **fkwargs)
    except Exception as e:
        err = ErrorAggregatorDict.single(str(e), test_identifier)
        return err
