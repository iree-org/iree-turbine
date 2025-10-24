import torch
from iree.turbine.kernel.boo.op_exports.aten import AtenParser


def test_aten_parser():
    sig = AtenParser.command_to_signature(
        [
            "aten::test",
            "[[2048, 4096], [4096, 8192]]",
            "['c10::BFloat16', 'float']",
            "[[1, 2048], [8192, 1]]",
            "['', '']",
        ]
    )
    [arg_0, arg_1] = sig.get_sample_args()
    assert list(arg_0.shape) == [2048, 4096]
    assert list(arg_0.stride()) == [1, 2048]
    assert arg_0.dtype == torch.bfloat16

    assert list(arg_1.shape) == [4096, 8192]
    assert list(arg_1.stride()) == [8192, 1]
    assert arg_1.dtype == torch.float64
