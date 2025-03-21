import os

# os.environ["TURBINE_DEBUG"] = "runtime_trace_dir=/home/zjgar/code/iree-turbine/trace_dir/, log_level=DEBUG"

from iree.turbine.ops._jinja_test_ops import test_add

from iree.turbine.runtime.op_reg import impl_helper
from iree.turbine.runtime.op_reg.base import KernelSelection, CustomOp, def_library, KernelBuilder, TensorArg, ALL_CUSTOM_OP_REGS
from iree.turbine.support.ir_imports import Operation, MLIRError, Attribute
from iree.turbine.support.logging import runtime_logger as logger
from iree.turbine.transforms.merger import Merger
import torch

import logging

x = torch.ones([2,2],dtype=torch.float32, device="cuda:0")
y = torch.ones([2,2],dtype=torch.float32, device="cuda:0")

# print(y.cpu())
z = test_add(x,y)

LIBRARY = def_library("fused_turbine")

from iree.turbine.aot import export, CompiledModule

@CustomOp.register(library=LIBRARY)
class test_matmul_add_relu(CustomOp):
    signature = "test_matmul_relu(Tensor t1, Tensor t2) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor([t1_desc.t.shape[0], t2_desc.t.shape[1]], t1_desc.t.dtype)
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        class M(torch.nn.Module):
            def forward(self, x, w):
                mm = torch.matmul(x,w)
                mm = test_add(mm,mm)
                return torch.nn.functional.relu(mm)
        args = []
        for arg_desc in ksel.arg_descs:
            if isinstance(arg_desc, TensorArg):
                args.append(arg_desc.t)


        function_name = "test_matmul_add_relu"
        def get_func():
            try:
                return kb.symbol_table[function_name]
            except KeyError:
                pass

            e = export(M(), args=tuple(args), function_name=function_name)
            e.import_to("import")
            print(e.mlir_module)
            CompiledModule.run_pass_pipeline(e.compiled_module, "builtin.module(inline, torch-func-backend-type-conversion)")
            try:
                func_op = e.mlir_module.regions[0].blocks[0].operations[0]
                with e.mlir_module.context as ctx:
                    pipeline_attr = Attribute.parse(
                        '#util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">'
                    )
                    func_op.attributes["preprocessing_pipeline"] = pipeline_attr
                asm = str(e.mlir_module)
            except MLIRError:
                logger.warning("Could not attach util.preprocessing_pipeline attr to func %s.", function_name)
            source_module_op = Operation.parse(asm, context=kb.context)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Generated kernel IR %s:\n%s", function_name, str(source_module_op)
                )
            merger = Merger(
                source_module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
            )
            merger.merge()
            func_op = kb.symbol_table[function_name]
            return func_op
        func_op = get_func()
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))
        print(kb.module_body)

z = test_matmul_add_relu(x,y)

print(z.cpu())


