import iree.turbine.kernel.lang as lang
import iree.turbine.kernel.wave as wave
import iree.turbine.kernel.wave.compile as wave_compile
import iree.turbine.kernel.wave.utils.torch_utils as torch_utils

import torch


def main():
    constraints: List[iree.turbine.kernel.lang.Constraint] = [
        wave.WorkgroupConstraint(lang.sym.N, lang.sym.BLOCK_N, 0),
        wave.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(
                16,
                1,
                1,
            ),
            vector_shapes={lang.sym.N: 128},
        ),
    ]

    @wave.wave(constraints)
    def kernel_func(
        in0: lang.Memory[lang.sym.N, lang.sym.ADDRESS_SPACE, lang.f16],
        in1: lang.Memory[lang.sym.N, lang.sym.ADDRESS_SPACE, lang.f16],
        out: lang.Memory[
            lang.sym.N, lang.global_symbols.GLOBAL_ADDRESS_SPACE, lang.f16
        ],
    ):
        in0_reg = wave.read(in0, elements_per_thread=2)
        in1_reg = wave.read(in1, elements_per_thread=2)
        wave.write(in0_reg + in1_reg, out, elements_per_thread=2)

    in0 = torch_utils.device_randn(1024, dtype=torch.float16)
    in1 = torch_utils.device_randn(1024, dtype=torch.float16)
    out = torch_utils.device_zeros(1024, dtype=torch.float16)

    hyper_params = {
        lang.sym.ADDRESS_SPACE: lang.global_symbols.SHARED_ADDRESS_SPACE,
        lang.sym.BLOCK_N: 128,
        lang.sym.N: 1024,
    }

    default_params = wave.utils.general_utils.get_default_scheduling_params()
    hyper_params.update(default_params)

    options = wave_compile.WaveCompileOptions(
        subs=hyper_params,
        canonicalize=True,
    )
    options = wave.utils.run_utils.set_default_run_config(options)

    kernel = wave_compile.wave_compile(options, kernel_func)
    kernel(in0, in1, out)

    torch.testing.assert_close(in0 + in1, out)
    print("ok")


if __name__ == "__main__":
    main()
