import torch
from iree.turbine.kernel.boo.op_exports.aten import AtenParser, AtenSignature


def test_aten_parser():
    sig = AtenParser.command_to_signature(
        [
            "aten::addmm",
            "[[512], [24576, 512], [512, 512], [], []]",
            "['c10::BFloat16', 'c10::BFloat16', 'float', 'Scalar', 'Scalar']",
            "[[1], [512, 1], [1, 512], [], []]",
            "['', '', '', '10', '20']",
        ]
    )

    [arg_0, arg_1, arg_2] = sig.get_sample_args()
    assert list(arg_0.shape) == [512]
    assert list(arg_0.stride()) == [1]
    assert arg_0.dtype == torch.bfloat16

    assert list(arg_1.shape) == [24576, 512]
    assert list(arg_1.stride()) == [512, 1]
    assert arg_1.dtype == torch.bfloat16

    assert list(arg_2.shape) == [512, 512]
    assert list(arg_2.stride()) == [1, 512]
    assert arg_2.dtype == torch.float64

    assert isinstance(sig, AtenSignature)
    [(beta_desc, beta_val), (alpha_desc, alpha_val)] = sig.get_concrete_args()
    assert beta_desc.name == "beta"
    assert beta_val == 10
    assert alpha_desc.name == "alpha"
    assert alpha_val == 20
