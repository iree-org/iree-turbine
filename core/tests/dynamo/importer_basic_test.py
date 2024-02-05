# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch

from testutils import *


class ImportTests(unittest.TestCase):
    def testImportStateless(self):
        a = torch.randn(3, 4)
        backend = create_backend()

        @dynamo.optimize(backend)
        def basic(x):
            return torch.tanh(x) * a

        basic(torch.randn(3, 4))

    def testImportDtype(self):
        def foo(x):
            o = x.to(torch.complex32)
            o = o.to(torch.float32)
            o = o.to(torch.float64)
            o = o.to(torch.float16)
            o = o.to(torch.int64)
            o = o.to(torch.int32)
            o = o.to(torch.int16)
            o = o.to(torch.int8)
            o = o.to(torch.uint8)
            o = o.to(torch.complex64)
            o = o.to(torch.bool)
            # o = o.to(torch.qint8) # we do not currently support quantized dtypes
            # o = o.to(torch.quint8)
            o = o.to(torch.bfloat16)
            return o

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.ones(10))

    def testImportDevice(self):
        def foo(x):
            return torch.arange(x, device="cpu")

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(10)

    def testImportLayout(self):
        def foo(x):
            # sparse layouts are not currently supported as they can not be created on the 'meta' device
            return torch.ones_like(x, layout=torch.strided)

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.randn(10))

    def testImportMemoryFormat(self):
        def foo():
            x = torch.ones_like(torch.randn(10), memory_format=torch.contiguous_format)
            x = torch.ones_like(torch.randn(10), memory_format=torch.preserve_format)
            x = torch.ones_like(
                torch.randn(1, 1, 1, 1), memory_format=torch.channels_last
            )
            x = torch.ones_like(
                torch.randn(1, 1, 1, 1, 1), memory_format=torch.channels_last_3d
            )

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testImportListArgs(self):
        def foo():
            return torch.randn((4, 5, 6))

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testImportListNodeArgs(self):
        def foo(x, y):
            return torch.cat((x, y), 0)

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.randn(10), torch.randn(10))

    def testImportOptionalListArgs(self):
        """
        Upsample triggers aten.index.Tensor with an 'indices' argument of the form List[Optional[Tensor]], this case tests
        whether we handle these cases properly in _import_list_argument
        """

        def foo(x):
            up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            return up(x)

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.randn(4, 4, 4, 4))

    def testPromoteScalarTensor(self):
        """
        Test whether scalar arguments are properly promoted to 0-rank Tensors for torch ops with no Scalar equivalent
        """

        def foo(x):
            return torch.ops.aten.div.Tensor_mode(x, 14, rounding_mode="floor")

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo(torch.randn(4, 4, 4, 4))

    def testImportDecomposeChunk(self):
        def foo_chunk(x):
            return torch.chunk(x, 2, dim=-1)

        opt = torch.compile(
            foo_chunk,
            backend=create_backend(
                decompose_ops=[
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes,
                ]
            ),
        )
        t = torch.randn([4, 4, 4, 4])
        opt(t)

    def testImportDecomposeBatchNorm2D(self):
        def foo_bn(x):
            return torch.nn.BatchNorm2d(4)(x)

        opt = torch.compile(
            foo_bn,
            backend=create_backend(
                decompose_ops=[
                    torch.ops.aten._native_batch_norm_legit_functional,
                    torch.ops.aten.squeeze.dims,
                ]
            ),
        )
        t = torch.randn([4, 4, 4, 4])
        opt(t)

    def testLiftFreshCopy(self):
        def foo():
            w = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
            x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
            y = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
            # TODO: Figure out why f64 is throwing a verification error
            # z = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
            return w, x, y

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testLiftFreshCopyComplex(self):
        def foo():
            x = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex64)
            y = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex128)
            return x, y

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testDenseResourceIntegerTypes(self):
        def foo():
            b = torch.tensor([True, False], dtype=torch.bool)
            ui8 = torch.tensor([[1, 2], [3, -4]], dtype=torch.uint8)
            i16 = torch.tensor([[1, 2], [-3, 4]], dtype=torch.int16)
            i32 = torch.tensor([[1, -2], [3, 4]], dtype=torch.int32)
            i64 = torch.tensor([[-1, 2], [3, 4]], dtype=torch.int64)
            return b, ui8, i16, i32, i64

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testDenseResourceFloatTypes(self):
        def foo():
            f16 = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.float16)
            f32 = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.float32)
            return f16, f32

        opt_foo = torch.compile(foo, backend=create_backend())
        opt_foo()

    def testImportVisionModule(self):
        from torch import nn
        import torch.nn.functional as F

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
                super(ConvBlock, self).__init__()
                self.stride = stride
                self.channel_pad = out_channels - in_channels
                padding = (kernel_size - 1) // 2
                self.convs = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=in_channels,
                        bias=True,
                    ),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                h = x
                if self.channel_pad > 0:
                    x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
                return self.act(self.convs(h) + x)

        mod = ConvBlock(3, 5)
        opt_mod = torch.compile(mod, backend=create_backend())
        opt_mod(torch.randn(1, 3, 256, 256))

    def testMultiHeadAttentionModule(self):
        import torch.nn as nn
        import torch.nn.functional as F

        class ScaledDotProductAttention(nn.Module):
            def __init__(self):
                super(ScaledDotProductAttention, self).__init__()

            def forward(self, Q, K, V, scale=None):
                attention = torch.matmul(Q, K.permute(0, 2, 1))
                if scale:
                    attention = attention * scale
                attention = F.softmax(attention, dim=-1)
                context = torch.matmul(attention, V)
                return context

        class MultiHeadAttention(nn.Module):
            def __init__(self, dim_model, num_head, dropout=0.0):
                super(MultiHeadAttention, self).__init__()
                self.num_head = num_head
                assert dim_model % num_head == 0
                self.dim_head = dim_model // self.num_head
                self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
                self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
                self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
                self.attention = ScaledDotProductAttention()
                self.fc = nn.Linear(num_head * self.dim_head, dim_model)
                self.dropout = nn.Dropout(dropout)
                self.layer_norm = nn.LayerNorm(dim_model)

            def forward(self, x):
                batch_size = x.size(0)
                Q = self.fc_Q(x)
                K = self.fc_K(x)
                V = self.fc_V(x)
                Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
                K = K.view(batch_size * self.num_head, -1, self.dim_head)
                V = V.view(batch_size * self.num_head, -1, self.dim_head)
                scale = K.size(-1) ** -0.5
                context = self.attention(Q, K, V, scale)
                context = context.view(batch_size, -1, self.dim_head * self.num_head)
                out = self.fc(context)
                out = self.dropout(out)
                out = out + x
                out = self.layer_norm(out)
                return out

        mod = MultiHeadAttention(256, 4)
        opt = torch.compile(mod, backend=create_backend())
        opt(torch.randn(1, 1, 256, 256))

    def testImportAtenFull(self):
        def foo(x):
            return torch.full(x.size(), fill_value=float("-inf"))

        opt_foo = torch.compile(foo, backend="turbine_cpu")
        opt_foo(torch.randn(2, 3))

    def _create_model(self, bias):
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self, input_size, output_size, bias=False):
                super().__init__()
                self.classifier = torch.nn.Linear(input_size, output_size, bias=bias)

            def forward(self, x):
                return self.classifier(x)

        return SimpleModel(20, 30, bias)

    def test_model_no_bias(self):
        model_no_bias = self._create_model(bias=False)
        output_no_bias = model_no_bias(torch.randn(128, 20))
        print("\nOutput without bias:")
        print(output_no_bias)
        opt_foo = torch.compile(model_no_bias, backend="turbine_cpu")
        opt_foo(torch.randn(128, 20))

    def test_model_with_bias(self):
        model_with_bias = self._create_model(bias=True)
        output_with_bias = model_with_bias(torch.randn(128, 20))
        print("\nOutput with bias:")
        print(output_with_bias)
        opt_foo = torch.compile(model_with_bias, backend="turbine_cpu")
        opt_foo(torch.randn(128, 20))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
