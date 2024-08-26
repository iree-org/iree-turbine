__all__ = [
    "DataType",
    "bool",
    "i4",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "f32",
    "f64",
    "index",
]

_INT_TYPES = ["i1", "i4", "i8", "i16", "i32", "i64"]
_FLOAT_TYPES = ["f16", "f32", "f64"]
_INDEX_TYPES = ["index"]


# TODO: this should really be a type.
class DataType:
    _name: str
    _ir_type_asm: str

    def __init__(self, name, ir_type_asm=None):
        self._name = name
        self._ir_type_asm = ir_type_asm if ir_type_asm else name

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
        return int(self._name[1:])


bool = DataType("bool", "i1")
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
index = DataType("index")
