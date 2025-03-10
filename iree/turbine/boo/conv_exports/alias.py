s = """"# --batchsize          -n        Mini-batch size (Default=100)
# --in_channels        -c        Number of Input Channels (Default=3)
# --in_d               -!        Input Depth (Default=32)
# --in_h               -H        Input Height (Default=32)
# --in_w               -W        Input Width (Default=32)
# --in_layout          -I        Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --in_cast_type       -U        Cast type for input tensor, default to not set
# --out_channels       -k        Number of Output Channels (Default=32)
# --fil_d              -@        Filter Depth (Default=3)
# --fil_h              -y        Filter Height (Default=3)
# --fil_w              -x        Filter Width (Default=3)
# --fil_layout         -f        Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --wei_cast_type      -R        Cast type for weight tensor, default to not set
# --bias               -b        Use Bias (Default=0)
# --out_layout         -O        Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
# --out_cast_type      -T        Cast type for output tensor, default to not set
# --forw               -F        Flag enables fwd, bwd, wrw convolutions
# --mode               -m        Convolution Mode (conv, trans) (Default=conv)
# --conv_stride_d      -#        Convolution Stride for Depth (Default=1)
# --conv_stride_h      -u        Convolution Stride for Height (Default=1)
# --conv_stride_w      -v        Convolution Stride for Width (Default=1)
# --pad_d              -$        Zero Padding for Depth (Default=0)
# --pad_h              -p        Zero Padding for Height (Default=0)
# --pad_w              -q        Zero Padding for Width (Default=0)
# --pad_val            -r        Padding Value (Default=0)
# --pad_mode           -z        Padding Mode (same, valid, default) (Default=default)
# --dilation_d         -^        Dilation of Filter Depth (Default=1)
# --dilation_h         -l        Dilation of Filter Height (Default=1)
# --dilation_w         -j        Dilation of Filter Width (Default=1)
# --trans_output_pad_d -%        Zero Padding Output for Depth (Default=0)
# --trans_output_pad_h -Y        Zero Padding Output for Height (Default=0)
# --trans_output_pad_w -X        Zero Padding Output for Width (Default=0)
# --group_count        -g        Number of Groups (Default=1)
"""


def get_aliases_and_defaults():
    lines = s.splitlines()
    alias_dict = {}
    default_dict = {
        "-F": "0",
        "-T": None,
        "-U": None,
        "-R": None,
        "-I": None,
        "-f": None,
        "-O": None,
    }
    for l in lines:
        items = l.split()
        if len(items) < 4:
            continue
        short = items[2]
        long = items[1]
        alias_dict[short] = long
        if items[-1].startswith("(Default="):
            default = items[-1].removeprefix("(Default").removesuffix(")")
            default_dict[short] = default

    return alias_dict, default_dict
