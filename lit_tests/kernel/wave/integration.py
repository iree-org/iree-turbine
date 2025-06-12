# RUN: python %s | FileCheck %s

import iree.turbine.aot as aot
import torch
import textwrap
from jinja2 import Environment, BaseLoader
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_bhsd_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.compile_utils import (
    get_wave_module_body_asm,
    get_kernel_name,
)
from iree.turbine.runtime.op_reg.base import (
    CustomOp,
    KernelSelection,
    KernelBuilder,
)
from iree.turbine.support.ir_imports import (
    PassManager,
)
from iree.turbine.runtime.op_reg.impl_helper import (
    call_function,
)
from iree.compiler.ir import (
    Module,
    Context,
    RankedTensorType,
    Operation,
    MLIRError,
)
from iree.turbine.transforms.merger import Merger
from typing import Optional

_JINJA2_ENVIRONMENT: Optional[Environment] = None


def _get_jinja2_env() -> Environment:
    global _JINJA2_ENVIRONMENT
    if _JINJA2_ENVIRONMENT is None:
        _JINJA2_ENVIRONMENT = Environment(loader=BaseLoader())
    return _JINJA2_ENVIRONMENT


@CustomOp.register()
class WaveBhsdFlashAttention(CustomOp):

    signature = "wave_bhsd_flash_attention(Tensor q, Tensor k, Tensor v, Tensor output) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        q_desc = ksel.arg_tensor(0)  # Shape b, h, qs, qd
        k_desc = ksel.arg_tensor(1)  # Shape b, h, ks, kd
        v_desc = ksel.arg_tensor(2)  # Shape b, h, vs, vd
        o_desc = ksel.arg_tensor(3)  # Shape b, h, qs, vd

        q_bs = q_desc.t.shape[:-2]
        k_bs = k_desc.t.shape[:-2]
        v_bs = v_desc.t.shape[:-2]

        bs = len(q_bs)

        # Note: kernel does collapse dims to get to a single batch/head dim
        torch._check(len(q_bs) == 2, lambda: f"TODO: batch dims {bs} not supported")

        q_s, q_d = q_desc.t.shape[-2:]
        k_s, k_d = k_desc.t.shape[-2:]
        v_s, v_d = v_desc.t.shape[-2:]

        torch._check(
            q_desc.t.dtype.is_floating_point
            and k_desc.t.dtype.is_floating_point
            and v_desc.t.dtype.is_floating_point,
            lambda: f"wave_flash_attention: Expected floating point",
        )
        torch._check(
            q_desc.t.dtype == k_desc.t.dtype == v_desc.t.dtype,
            lambda: f"wave_flash_attention: Expected matching dtypes",
        )

        for q_b, k_b, v_b in zip(q_bs, k_bs, v_bs):
            torch._check(
                q_b == k_b and q_b == v_b,
                lambda: f"expected matching batch dims: {q_b}, {k_b}, {v_b}",
            )

        torch._check(q_d == k_d, lambda: f"expected matching qk features: {q_d}, {k_d}")

        torch._check(k_s == v_s, lambda: f"expected matching kv length: {q_d}, {k_d}")

        q_desc.specialize_dims(0, 1, 2, -1)
        k_desc.specialize_dims(0, 1, 2, -1)
        v_desc.specialize_dims(0, 1, 2, -1)
        o_desc.specialize_dims(0, 1, 2, -1)

        # Result 0: Shape batch, num_heads, m, n
        ksel.return_new_tensor((*q_bs, q_s, v_d), dtype=torch.float32).specialize_dims(
            0, 1, 2, -1
        )

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        q = kb.arg_value(0)
        k = kb.arg_value(1)
        v = kb.arg_value(2)
        output = kb.arg_value(3)

        q_tensor_type = RankedTensorType(q.type)
        v_tensor_type = RankedTensorType(v.type)

        batch_size, num_heads, q_s, q_d = q_tensor_type.shape
        v_batch_size, num_heads_kv, v_s, v_d = v_tensor_type.shape

        # Unspecialized dims will be negative
        i_type_str = str(q_tensor_type.element_type)
        # TODO: enable f16 output type via arg
        o_type_str = "f32"

        wave_kernel_name = f"wave_flash_attention_{batch_size}_{num_heads}_{q_s}_{v_d}_{i_type_str}_{o_type_str}"

        shape = AttentionShape(
            batch_size=batch_size,
            num_query_heads=num_heads,
            num_kv_heads=num_heads_kv,
            query_seq_len=q_s,
            head_size_kv=v_d,
            head_size=q_d,
            kv_seq_len=v_s,
        )

        mfma_variant = (tkw.MMAType.F32_32x32x8_F16, tkw.MMAType.F32_32x32x8_F16)
        dynamic_dims = False
        is_causal = True
        is_custom_mask = False

        (
            base_attention_func,
            hyperparams,
            dynamic_symbols,
            dynamic_symbols_map,
        ) = get_bhsd_attention_kernel(
            shape,
            mfma_variant,
            dynamic_dims,
            is_causal=is_causal,
            is_custom_mask=is_custom_mask,
        )
        hyperparams.update(get_default_scheduling_params())
        options = WaveCompileOptions(
            subs=hyperparams,
            schedule=SchedulingType.NONE,
            dynamic_symbols=dynamic_symbols,
            dynamic_symbols_map=dynamic_symbols_map,
            waves_per_eu=2,
            denorm_fp_math_f32="preserve-sign",
            func_name=wave_kernel_name,
            compile_to_mlir=True,
        )
        options = set_default_run_config(options)
        with Context() as ctx:
            base_attention = wave_compile(options, base_attention_func)

        asm = base_attention.asm

        asm_module = Module.parse(asm)
        asm_body = get_wave_module_body_asm(asm_module)

        mlir_wave_kernel = (
            asm_body
            + f"""
        util.func private @{{{{kernel_name}}}}(%arg0: tensor<4x32x128x128xf16>, %arg1: tensor<4x32x128x128xf16>, %arg2: tensor<4x32x128x128xf16>, %arg3: tensor<4x32x128x128xf32>) -> tensor<4x32x128x128xf32> {{
            %result = func.call @{wave_kernel_name}(%arg0, %arg1, %arg2, %arg3) : (tensor<4x32x128x128xf16>, tensor<4x32x128x128xf16>, tensor<4x32x128x128xf16>, tensor<4x32x128x128xf32>) -> tensor<4x32x128x128xf32>
            util.return %result : tensor<4x32x128x128xf32>
        }}
        """
        )
        mlir = "module {" + mlir_wave_kernel + "}"

        dims = {
            "B": batch_size,
            "H": num_heads,
            "M": q_s,
            "K1": q_d,
            "K2": v_s,
            "N": v_d,
        }

        dtypes = {
            "q": "f16",
            "k": "f16",
            "v": "f16",
            "o": "f32",
            "result": "f32",
        }

        tensor_dim_orders = {
            "q": ["B", "H", "M", "K1"],
            "k": ["B", "H", "K2", "K1"],
            "v": ["B", "H", "K2", "N"],
            "o": ["B", "H", "M", "N"],
            "result": ["B", "H", "M", "N"],
        }

        kernel_name = get_kernel_name(
            "wave_bhsd_flash_attention", dims, dtypes, tensor_dim_orders
        )

        # Try to check if the symbol table already has a generated
        # kernel for this specialization.
        symbol_name = None
        try:
            symbol_name = kb.symbol_table[kernel_name]
        except KeyError:
            pass

        # If this kernel is not already generated, generate it using
        # the mlir spec.
        if symbol_name is None:
            asm = (
                _get_jinja2_env()
                .from_string(mlir)
                .render(
                    {
                        "kernel_name": kernel_name,
                    }
                )
            )
            try:
                module_op = Operation.parse(asm, context=kb.context)
            except MLIRError as e:
                lines = asm.splitlines()
                lines_numbered = "\n".join(
                    [f"      {str(i+1):>5}: {l}" for i, l in enumerate(lines)]
                )
                raise RuntimeError(
                    f"Error parsing generated op template:"
                    f"\n{textwrap.indent(str(e), '  ')}"
                    f"\n{lines_numbered}"
                )
            op = module_op.operation

            merger = Merger(
                op, kb.module_body.owner, target_symbol_table=kb.symbol_table
            )
            merger.merge()

            symbol_name = kb.symbol_table[kernel_name]

        kb.yield_results(*call_function(symbol_name, *kb.arg_bindings))


@run_test
def test_aot_wave_integration():
    class WaveBhsdModule(torch.nn.Module):
        def forward(self, q, k, v, output):
            return WaveBhsdFlashAttention(q, k, v, output)

    e = aot.export(
        WaveBhsdModule(),
        args=(
            torch.empty((4, 32, 128, 128), dtype=torch.float16),
            torch.empty((4, 32, 128, 128), dtype=torch.float16),
            torch.empty((4, 32, 128, 128), dtype=torch.float16),
            torch.empty((4, 32, 128, 128), dtype=torch.float32),
        ),
    )
    mlir_asm = str(e.mlir_module)
    print(mlir_asm)

    # CHECK-LABEL:       func.func @main

    # CHECK-LABEL:       stream.executable private @base_attention

    # CHECK-LABEL:       stream.executable.export public @base_attention

    # CHECK-LABEL:       func.func @base_attention
    # CHECK:                %[[ZERO_1:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK:                %[[ZERO_2:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK:                %[[ZERO_3:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK:                %[[NEG_INF:.+]] = arith.constant dense<-1.000000e+06> : vector<16xf32>
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-32:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-2:            {{.*}} = arith.cmpi slt, {{.*}} : vector<16xindex>
    # CHECK-COUNT-2:            {{.*}} = arith.cmpi sge, {{.*}} : vector<16xi32>
    # CHECK-COUNT-2:            {{.*}} = arith.andi {{.*}} : vector<16xi1>
    # CHECK-COUNT-2:            {{.*}} = arith.select %{{.*}}, %[[ZERO_3]], %[[NEG_INF]] : vector<16xi1>, vector<16xf32>
    # CHECK-COUNT-2:            {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<16xf32>
    # CHECK-COUNT-2:            {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-16:            {{.*}} = amdgpu.mfma

    # CHECK-LABEL:       func.func private @wave_flash_attention_4_32_128_128_f16_f32

    # CHECK-LABEL:       util.func private @wave_bhsd_flash_attention_B_4_H_32_M_128_K1_128_f16_B_4_H_32_K2_128_K1_128_f16_B_4_H_32_K2_128_N_128_f16_B_4_H_32_M_128_N_128_f32_B_4_H_32_M_128_N_128_f32
