# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class GeneralError(Exception):
    ...


class MismatchedDeviceSetClearError(AssertionError):
    def __init__(self):
        super().__init__("Calls to Device.set()/clear() are mismatched or unbalanced.")


class NoCurrentDeviceError(Exception):
    def __init__(self):
        super().__init__(
            "You accessed a method which requires a current device but none was set on this thread. "
            "Either pass an explicit 'device=' or set a current device via "
            "`with device:`"
        )


class UnsupportedTorchDeviceError(Exception):
    def __init__(self, torch_device):
        super().__init__(
            f"Attempt to use turbine with a torch.device that is not supported by this build: {torch_device}"
        )


class UnsupportedTypeError(Exception):
    def __init__(self, t: type, usage: str):
        super().__init__(f"Python type {t} is not supported for {usage}")


class ApiSequencingError(Exception):
    ...


class UnknownDTypeError(ValueError):
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(f"Unable to map torch dtype {dtype} to Turbine")
