import argparse
import torch
from ..support.logging import get_logger
import re

logger = get_logger("turbine.wave.interpreter")


from ..kernel.compiler.ir import (
    amdgpu_d,
    builtin_d,
    Context,
    IndexType,
    VectorType,
    Module,
    Operation,
    flow_d,
    func_d,
    gpu_d,
    llvm_d,
    scf_d,
    vector_d,
    memref_d,
    IntegerAttr,
    IndexType,
    arith_d,
    stream_d,
    F32Type,
    F16Type,
)


class Interpreter:
    """
    Python interpreter for MLIR.
    Uses torch for tensor operations.

    """

    def __init__(self, workgroup_ids: list[int], thread_ids: list[int]) -> None:
        self.workgroup_ids = workgroup_ids
        self.thread_ids = thread_ids
        self.symbol_table = {}
        # Reference to parent scf.for operation when walking the body
        # of the scf.for.
        self.for_op = None

    def get_dtype(self, dtype):
        if type(dtype) == F32Type:
            return torch.float32
        if type(dtype) == F16Type:
            return torch.float16
        raise NotImplementedError(f"Unsupported dtype: {dtype}")

    def create_tensor(self, shape: list[int], dtype, value) -> torch.Tensor:
        """
        Creates a constant tensor with the given shape, dtype and value.
        The tensor is filled with ones.
        """
        if type(dtype) == F32Type or type(dtype) == F16Type:
            value = float(value)
        return torch.ones(*shape, dtype=self.get_dtype(dtype)) * value

    def callback(self, op: Operation) -> None:
        if (
            op.operation.parent.name == "func.func"
            or op.operation.parent.name == "scf.for"
        ):

            logger.debug(f"Processing operation: {op}")
            value = None
            match type(op):
                case arith_d.ConstantOp:
                    vtype = type(op.value.type)
                    if vtype == IndexType:
                        value = torch.Tensor([int(IntegerAttr(op.value))])
                    elif vtype == VectorType:
                        shape = op.value.type.shape
                        dtype = op.value.type.element_type
                        value = self.create_tensor(
                            shape,
                            dtype,
                            op.attributes["value"].get_splat_value(),
                        )
                    else:
                        raise NotImplementedError(f"Unsupported constant type: {vtype}")
                case arith_d.MulIOp:
                    value = (
                        self.symbol_table[op.operands[0]]
                        * self.symbol_table[op.operands[1]]
                    )
                case arith_d.RemSIOp:
                    value = (
                        self.symbol_table[op.operands[0]]
                        % self.symbol_table[op.operands[1]]
                    )
                case arith_d.AddIOp:
                    value = (
                        self.symbol_table[op.operands[0]]
                        + self.symbol_table[op.operands[1]]
                    )
                case arith_d.SubIOp:
                    value = (
                        self.symbol_table[op.operands[0]]
                        - self.symbol_table[op.operands[1]]
                    )
                case arith_d.DivSIOp:
                    value = (
                        self.symbol_table[op.operands[0]]
                        // self.symbol_table[op.operands[1]]
                    )
                case amdgpu_d.LDSBarrierOp:
                    return
                case amdgpu_d.MFMAOp:
                    lhs = self.symbol_table[op.operands[0]]
                    rhs = self.symbol_table[op.operands[1]]
                    acc = self.symbol_table[op.operands[2]]
                    # TODO: Just use first row for now (which works for constant matrices)
                    # But figure out what to do in the general case
                    tmp = torch.outer(lhs, rhs)[0]
                    value = tmp + acc
                case vector_d.LoadOp:
                    load_indices = []
                    for index in op.indices:
                        load_indices.append(self.symbol_table[index])
                    memref = self.symbol_table[op.base]
                    result_type = op.result.type
                    result_shape = result_type.shape
                    result_dtype = result_type.element_type
                    value = torch.zeros(
                        *result_shape, dtype=self.get_dtype(result_dtype)
                    )
                    # Row-major load
                    load_indices = [int(x) for x in load_indices]
                    logger.debug("Load indices:", load_indices)
                    offset = [0 for _ in range(len(load_indices))]
                    offset[-1] += 1
                    for i in range(*result_shape):
                        value[i] = memref[
                            *[x + y for x, y in zip(load_indices, offset)]
                        ]
                case vector_d.ExtractStridedSliceOp:
                    vector = self.symbol_table[op.vector]
                    value = vector[[int(x) for x in op.offsets]]
                case vector_d.StoreOp:
                    store_indices = []
                    for index in op.indices:
                        store_indices.append(self.symbol_table[index])
                    vector = self.symbol_table[op.valueToStore]
                    memref = self.symbol_table[op.base]
                    result_type = vector.type
                    result_shape = vector.shape
                    # Row-major store
                    store_indices = [int(x) for x in store_indices]
                    offset = [0 for _ in range(len(store_indices))]
                    offset[-1] += 1
                    for i in range(*result_shape):
                        memref[
                            *[x + y for x, y in zip(store_indices, offset)]
                        ] = vector[i]
                case stream_d.DispatchWorkgroupIDOp:
                    index = int(op.attributes["dimension"])
                    value = self.workgroup_ids[index]
                    value = torch.Tensor([value])
                case stream_d.BindingSubspanOp:
                    mtype = op.result.type
                    shape = mtype.shape
                    dtype = mtype.element_type
                    value = torch.ones(
                        shape, dtype=self.get_dtype(dtype)
                    ) * torch.randn((1,))
                case gpu_d.ThreadIdOp:
                    dim = re.findall(r"^#gpu<dim (.*)>", str(op.dimension))[0]
                    if dim == "x":
                        value = self.thread_ids[0]
                    if dim == "y":
                        value = self.thread_ids[1]
                    if dim == "z":
                        value = self.thread_ids[2]
                    value = torch.Tensor([value])
                case memref_d.AllocOp:
                    mtype = op.memref.type
                    shape = mtype.shape
                    dtype = mtype.element_type
                    value = torch.zeros(shape, dtype=self.get_dtype(dtype))
                case scf_d.ForOp:
                    lb = int(self.symbol_table[op.lowerBound])
                    ub = int(self.symbol_table[op.upperBound])
                    step = int(self.symbol_table[op.step])
                    self.for_op = op
                    for init_arg, iter_arg in zip(op.initArgs, op.inner_iter_args):
                        self.symbol_table[iter_arg] = self.symbol_table[init_arg]
                    for i in range(lb, ub, step):
                        self.symbol_table[op.induction_variable] = i
                        for k in range(len(op.body.operations)):
                            self.callback(op.body.operations[k])
                    for result, iter_arg in zip(op.results, op.inner_iter_args):
                        self.symbol_table[result] = self.symbol_table[iter_arg]
                    return
                case scf_d.YieldOp:
                    for result, iter_arg in zip(
                        op.operands, self.for_op.inner_iter_args
                    ):
                        self.symbol_table[iter_arg] = self.symbol_table[result]
                    return
                case func_d.ReturnOp:
                    return
                case flow_d.DispatchOp:
                    return
                case llvm_d.CallIntrinsicOp:
                    return
                case _:
                    raise NotImplementedError(f"Unsupported operation: {op}")

        if type(op) != vector_d.StoreOp:
            self.symbol_table[op.result] = value

    def walk_operations(self, operation: Operation, callback: callable) -> None:
        for region in operation.regions:
            for block in region.blocks:
                for op in block.operations:
                    if isinstance(
                        op, (stream_d.ExecutableOp, builtin_d.ModuleOp, func_d.FuncOp)
                    ):
                        self.walk_operations(op, callback)
                        return
                    if isinstance(op, stream_d.ExecutableExportOp):
                        continue
                    callback(op)
                    self.walk_operations(op, callback)

    def interpret(self, asm: str) -> None:
        with Context() as _:
            module = Module.parse(asm)
            operation = module.operation
            self.walk_operations(operation, self.callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLIR Interpreter")
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--workgroup_ids", nargs="+", type=int, help="Workgroup ids")
    parser.add_argument("--thread_ids", nargs="+", type=int, help="Thread ids")
    args = parser.parse_args()
    with open(args.file, "r") as f:
        asm = f.read()
    interpreter = Interpreter(parser.workgroup_ids, parser.thread_ids)
    interpreter.interpret(asm)
