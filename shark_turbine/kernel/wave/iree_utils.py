import torch
from .utils import compile_and_invoke
from ...support.conversions import TORCH_DTYPE_TO_MLIR_TYPE_ASM


def get_mmt_asm(lhs_type: str, rhs_type: str, acc_type: str) -> str:
    acc_dtype = acc_type.split("x")[-1]
    matmul_function = f"""
    func.func @mmt(%lhs: tensor<{lhs_type}>, %rhs: tensor<{rhs_type}>) -> tensor<{acc_type}> {{
      %c0 = arith.constant 0.0 : {acc_dtype}
      %init = tensor.empty() : tensor<{acc_type}>
      %inital_result = linalg.fill ins(%c0 : {acc_dtype}) outs(%init : tensor<{acc_type}>) -> tensor<{acc_type}>
      %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<{lhs_type}>, tensor<{rhs_type}>)
                 outs(%inital_result: tensor<{acc_type}>) -> tensor<{acc_type}>
      return %result : tensor<{acc_type}>
    }}"""
    return matmul_function


def dtype_str(dtype: torch.dtype) -> str:
    dtype_str = TORCH_DTYPE_TO_MLIR_TYPE_ASM.get(dtype, None)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_str


def get_type_str(shape: tuple[int], dtype: torch.dtype) -> str:
    return "x".join([str(x) for x in shape] + [dtype_str(dtype)])


def generate_iree_ref(
    kernel_type: str,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    config: dict[str, str],
):
    """
    Generate a reference output for the given kernel type and arguments.
    """

    asm = None
    if kernel_type == "mmt":
        lhs_type = get_type_str(kernel_inputs[0].shape, kernel_inputs[0].dtype)
        rhs_type = get_type_str(kernel_inputs[1].shape, kernel_inputs[1].dtype)
        acc_type = get_type_str(kernel_outputs[0].shape, kernel_outputs[0].dtype)
        asm = get_mmt_asm(lhs_type, rhs_type, acc_type)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    compile_and_invoke(
        asm, kernel_type, config, kernel_inputs, kernel_outputs, True, False
    )
