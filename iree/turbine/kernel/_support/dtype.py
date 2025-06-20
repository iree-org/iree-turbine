__all__ = [
    "DataType",
    "bf16",
    "bool",
    "i4",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "f32",
    "f64",
    "f8E5M2",
    "f8E5M2FNUZ",
    "f8E4M3FN",
    "f8E4M3FNUZ",
    "f8E8M0FNU",
    "f6E2M3FN",
    "f4E2M1FN",
    "index",
]

_INT_TYPES = ["i1", "i4", "i8", "i16", "i32", "i64"]
_FLOAT_TYPES = [
    "bf16",
    "f16",
    "f32",
    "f64",
    "f8E5M2",
    "f8E5M2FNUZ",
    "f8E4M3FN",
    "f8E4M3FNUZ",
    "f8E8M0FNU",
    "f6E2M3FN",
    "f4E2M1FN",
]
_INDEX_TYPES = ["index"]


# TODO: this should really be a type.
class DataType:
    _name: str
    _ir_type_asm: str

    def __init__(self, name, ir_type_asm=None):
        self._name = name
        self._ir_type_asm = ir_type_asm if ir_type_asm else name
        self._symbolic_shape = ()

    def ir_type_asm(self):
        return self._ir_type_asm

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"DataType({self._ir_type_asm})"

    def is_int_asm(self):
        return self._name in _INT_TYPES

    def is_float_asm(self):
        return self._name in _FLOAT_TYPES

    def is_index_asm(self):
        return self._name in _INDEX_TYPES

    def bitwidth(self):
        if self._name == "bool":
            return 1
        if self._name == "index":
            return 64
        if "f4" in self._name:
            return 4
        if "f8" in self._name:
            return 8
        if "bf16" in self._name:
            return 16
        return int(self._name[1:])

    @property
    def dtype(self):
        # This cls is already dtype, hence can return self.
        # dtype() function is useful here for code reuse between
        # scalar and vector/register variables.
        return self

    @property
    def symbolic_shape(self):
        return self._symbolic_shape

    @symbolic_shape.setter
    def symbolic_shape(self, value):
        if not value:
            self._symbolic_shape = ()
        else:
            self.symbolic_shape = value


bf16 = DataType("bf16")
bool = DataType("bool", "i1")
i1 = bool
i4 = DataType("i4")
i8 = DataType("i8")
i16 = DataType("i16")
i32 = DataType("i32")
i64 = DataType("i64")
f32 = DataType("f32")
f64 = DataType("f64")
f16 = DataType("f16")
f32 = DataType("f32")
f64 = DataType("f64")
f8e5m2 = DataType("f8E5M2")
f8e5m2fnuz = DataType("f8E5M2FNUZ")
f8e4m3fn = DataType("f8E4M3FN")
f8e4m3fnuz = DataType("f8E4M3FNUZ")
f8e8m0fnu = DataType("f8E8M0FNU")
f6e2m3fn = DataType("f6E2M3FN")
f4e2m1fn = DataType("f4E2M1FN")
index = DataType("index")
