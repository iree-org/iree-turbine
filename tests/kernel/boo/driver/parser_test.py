import torch
from iree.turbine.kernel.boo.op_exports.aten import AtenParser, AtenSignature
import pytest


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


@pytest.mark.parametrize(
    "aten_dtype, expected_dtype",
    [("c10::Half", torch.float16), ("c10::Float", torch.float32)],
)
def test_aten_parser_sdpa(aten_dtype: str, expected_dtype: torch.dtype):
    sig = AtenParser.command_to_signature(
        [
            "aten::scaled_dot_product_attention",
            "[[32, 8, 128, 64], [32, 8, 128, 64], [32, 8, 128, 64], [], [], [], [], []]",
            f"['{aten_dtype}', '{aten_dtype}', '{aten_dtype}', '', 'Scalar', 'Scalar', '', 'Scalar']",
            "[[65536, 8192, 64, 1], [65536, 8192, 64, 1], [65536, 8192, 64, 1], [], [], [], [], []]",
            "['', '', '', 'None', '0.0', 'False', 'None', 'False']",
        ]
    )

    [arg_0, arg_1, arg_2] = sig.get_sample_args()
    assert list(arg_0.shape) == [32, 8, 128, 64]
    assert list(arg_0.stride()) == [65536, 8192, 64, 1]
    assert arg_0.dtype == expected_dtype


def test_aten_parser_non_contiguous_strides():
    # Non-contiguous strides may address more elements than prod(dims).
    # e.g. dims=[672,3,3], strides=[16,1,4]:
    #   prod(dims) = 6048, but sum((d-1)*s)+1 = 671*16+2*1+2*4+1 = 10747.
    sig = AtenParser.command_to_signature(
        [
            "aten::bmm",
            "[[672, 3, 3], [672, 3, 25000]]",
            "['float', 'float']",
            "[[16, 1, 4], [75000, 25000, 1]]",
            "['', '']",
        ]
    )

    [arg_0, arg_1] = sig.get_sample_args()

    assert list(arg_0.shape) == [672, 3, 3]
    assert list(arg_0.stride()) == [16, 1, 4]
    assert arg_0.dtype == torch.float64

    assert list(arg_1.shape) == [672, 3, 25000]
    assert list(arg_1.stride()) == [75000, 25000, 1]
    assert arg_1.dtype == torch.float64
