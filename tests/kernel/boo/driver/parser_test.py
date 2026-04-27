import torch
from iree.turbine.kernel.boo.op_exports.aten import AtenParser, AtenSignature
from iree.turbine.kernel.boo.op_exports.layer_norm import LayerNormParser
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
    assert arg_2.dtype == torch.float32

    assert isinstance(sig, AtenSignature)
    [(beta_desc, beta_val), (alpha_desc, alpha_val)] = sig.get_concrete_args()
    assert beta_desc.name == "beta"
    assert beta_val == 10
    assert alpha_desc.name == "alpha"
    assert alpha_val == 20


@pytest.mark.parametrize(
    "aten_dtype, expected_dtype",
    [
        ("float", torch.float32),
        ("double", torch.float64),
        ("c10::Half", torch.float16),
        ("c10::BFloat16", torch.bfloat16),
    ],
)
def test_aten_dtype_mapping(aten_dtype: str, expected_dtype: torch.dtype):
    """Test that profiler dtype strings map to the correct torch dtypes."""
    sig = AtenParser.command_to_signature(
        [
            "aten::bmm",
            "[[2, 3, 4], [2, 4, 5]]",
            f"['{aten_dtype}', '{aten_dtype}']",
            "[[12, 4, 1], [20, 5, 1]]",
            "['', '']",
        ]
    )
    [arg_0, arg_1] = sig.get_sample_args()
    assert arg_0.dtype == expected_dtype
    assert arg_1.dtype == expected_dtype


@pytest.mark.parametrize(
    "aten_dtype, expected_dtype",
    [("c10::Half", torch.float16), ("float", torch.float32)],
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
    assert arg_0.dtype == torch.float32

    assert list(arg_1.shape) == [672, 3, 25000]
    assert list(arg_1.stride()) == [75000, 25000, 1]
    assert arg_1.dtype == torch.float32


def test_layer_norm_parser_mode_0():
    """mode 0 (MIOPEN_ELEMENTWISE_AFFINE) should produce input-only signature."""
    parser = LayerNormParser.get_miopen_parser()
    args = parser.parse_args(["layernorm", "--input", "2x3x4x5", "--mode=0"])
    sig = LayerNormParser.get_signature(args)

    assert not sig.elementwise_affine
    assert not sig.bias
    sample_args = sig.get_sample_args(seed=0)
    assert len(sample_args) == 1, "mode 0 should only have input"


def test_layer_norm_parser_mode_1():
    """mode 1 (MIOPEN_WEIGHT_BIAS) should produce input/weight/bias signature."""
    parser = LayerNormParser.get_miopen_parser()
    args = parser.parse_args(["layernorm", "--input", "2x3x4x5", "--mode=1"])
    sig = LayerNormParser.get_signature(args)

    assert sig.elementwise_affine
    assert sig.bias
    sample_args = sig.get_sample_args(seed=0)
    assert len(sample_args) == 3, "mode 1 should have input, weight, bias"


@pytest.mark.parametrize("op_name", ["aten::argmax", "aten::argmin"])
def test_arg_compare_parser(op_name: str):
    sig = AtenParser.command_to_signature(
        [
            op_name,
            "[[4, 8, 16], [], []]",
            "['float', 'Scalar', 'Scalar']",
            "[[128, 16, 1], [], []]",
            "['', '1', 'False']",
        ]
    )
    assert isinstance(sig, AtenSignature)
    assert sig.name == op_name
    (input_tensor,) = sig.get_sample_args()
    assert list(input_tensor.shape) == [4, 8, 16]
    assert input_tensor.dtype == torch.float32

    concrete = list(sig.get_concrete_args())
    assert len(concrete) == 2
    assert concrete[0][0].name == "dim"
    assert concrete[0][1] == 1
    assert concrete[1][0].name == "keepdim"
    assert concrete[1][1] == False
