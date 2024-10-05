from typing import Any, Optional, Union

from .._support.indexing import (
    IndexExpr,
    SymIndex,
)

from .base import (
    CodegenError,
    NDEBUG,
)

from .ir import (
    Attribute,
    Context,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    DenseElementsAttr,
    IrType,
    Location,
    Operation,
    SymbolTable,
    Value,
    VectorType,
    arith_d,
    math_d,
    builtin_d,
    F16Type,
    F32Type,
    F64Type,
)

# TODO: Use FloatType from upstream when available.
FLOAT_BITWIDTHS = {
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    # TODO: FP8 types.
}


class IRProxyValue:
    """Wrapper around an (ir.Value, py_value) for handling notionally python
    proxies that are associated with an IR Value.
    """

    __slots__ = [
        "ir_value",
        "py_value",
    ]

    def __init__(self, ir_value: Value, py_value: Any = None):
        self.ir_value = ir_value
        self.py_value = py_value
        assert NDEBUG or self.validate()

    def validate(self):
        assert isinstance(self.ir_value, Value), f"Got {type(self.ir_value)}"
        return True

    def __repr__(self):
        return f"<IRProxyValue({self.ir_value}):{self.py_value}>"


class ModuleBuilder:
    def __init__(
        self,
        *,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ):
        if module_op:
            self.module_op = module_op
            self.body_block = module_op.regions[0].blocks[0]
        else:
            if not context:
                context = Context()
            self.module_op = builtin_d.ModuleOp(loc=Location.unknown(context))
            self.body_block = self.module_op.body
        self.context = self.module_op.context
        self.unknown_loc = Location.unknown(self.context)
        self.symbol_table = SymbolTable(self.module_op)


class _ScalarBuilder:
    def is_floating_point_type(self, t: IrType) -> bool:
        # TODO: Use FloatType from upstream when available.
        return str(t) in FLOAT_BITWIDTHS

    def is_integer_type(self, t: IrType) -> bool:
        return IntegerType.isinstance(t)

    def is_index_type(self, t: IrType) -> bool:
        return IndexType.isinstance(t)

    def get_typeclass(self, t: IrType, index_same_as_integer=False) -> str:
        # If this is a vector type, get the element type.
        if isinstance(t, VectorType):
            t = t.element_type
        if self.is_floating_point_type(t):
            return "float"
        if self.is_integer_type(t):
            return "integer"
        if self.is_index_type(t):
            return "integer" if index_same_as_integer else "index"
        raise CodegenError(f"Unknown typeclass for type `{t}`")

    def get_float_bitwidth(self, t: IrType) -> int:
        # If this is a vector type, get the element type.
        if isinstance(t, VectorType):
            t = t.element_type
        return FLOAT_BITWIDTHS[str(t)]

    def to_dtype(self, value: IRProxyValue, dtype: IrType) -> IRProxyValue:
        value_type = value.ir_value.type
        # Create a vector type for dtype if value is a vector.
        to_type = dtype
        if isinstance(value_type, VectorType):
            to_type = VectorType.get(value_type.shape, dtype)

        # Short-circuit if already the right type.
        if value_type == to_type:
            return value

        value_typeclass = self.get_typeclass(value_type)
        to_typeclass = self.get_typeclass(dtype)
        attr_name = f"to_dtype_{value_typeclass}_to_{to_typeclass}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"No implemented path to implicitly promote scalar `{value_type}` to `{to_type}` (tried '{attr_name}')"
            )
        return IRProxyValue(handler(value.ir_value, to_type))

    def constant_attr(self, val: int | float, element_type: IrType) -> Attribute:
        if self.is_integer_type(element_type) or self.is_index_type(element_type):
            if not isinstance(val, int):
                raise TypeError(f"Expected an integer value, got {val}")
            return IntegerAttr.get(element_type, val)

        if self.is_floating_point_type(element_type):
            if not isinstance(val, float):
                raise TypeError(f"Expected a float value, got {val}")
            return FloatAttr.get(element_type, val)

        raise CodegenError(
            f"Cannot create a constant attribute for type `{element_type}`"
        )

    def zero_attr(self, t: IrType) -> Attribute:
        if self.is_integer_type(t) or self.is_index_type(t):
            return self.constant_attr(0, t)
        if self.is_floating_point_type(t):
            return self.constant_attr(0.0, t)
        raise CodegenError(f"Cannot create a zero attribute for type `{t}`")

    def constant(self, py_value, element_type: IrType) -> IRProxyValue:
        attr = self.constant_attr(py_value, element_type)
        return IRProxyValue(arith_d.constant(element_type, attr))

    def constant_vector(self, py_value, shape, element_type: IrType) -> IRProxyValue:
        attr = self.constant_attr(py_value, element_type)
        vector_type = VectorType.get(shape, element_type)
        splat = DenseElementsAttr.get_splat(vector_type, attr)
        return IRProxyValue(arith_d.constant(vector_type, splat))

    def binary_arithmetic(
        self, op: str, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        lhs_ir_type = lhs.ir_value.type
        rhs_ir_type = rhs.ir_value.type

        if lhs_ir_type != rhs_ir_type:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir_type} and {rhs_ir_type} due to element type mismatch"
            )

        typeclass = self.get_typeclass(lhs_ir_type, True)
        attr_name = f"binary_{op}_{typeclass}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir_type} and {rhs_ir_type} (tried '{attr_name}')"
            )
        return handler(lhs, rhs)

    def binary_vector_arithmetic(
        self, op: str, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        lhs_ir = lhs.ir_value
        rhs_ir = rhs.ir_value
        lhs_element_type = VectorType(lhs_ir.type).element_type
        rhs_element_type = VectorType(rhs_ir.type).element_type

        if lhs_element_type != rhs_element_type:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir.type} and {rhs_ir.type} due to element type mismatch"
            )

        typeclass = self.get_typeclass(lhs_element_type, True)
        attr_name = f"binary_{op}_{typeclass}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform binary arithmetic operation '{op}' between {lhs_ir.type} and {rhs_ir.type} (tried '{attr_name}')"
            )
        return handler(lhs, rhs)

    def unary_arithmetic(self, op: str, val: IRProxyValue) -> IRProxyValue:
        val_ir_type = val.ir_value.type
        typeclass = self.get_typeclass(val_ir_type, True)
        attr_name = f"unary_{op}_{typeclass}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform unary arithmetic operation '{op}' on {val_ir_type} (tried '{attr_name}')"
            )
        return handler(val)

    def unary_vector_arithmetic(self, op: str, val: IRProxyValue) -> IRProxyValue:
        val_ir = val.ir_value
        val_element_type = VectorType(val_ir.type).element_type
        typeclass = self.get_typeclass(val_element_type, True)
        attr_name = f"unary_{op}_{typeclass}"
        try:
            handler = getattr(self, attr_name)
        except AttributeError:
            raise CodegenError(
                f"Cannot perform unary arithmetic operation '{op}' on {val_ir.type} (tried '{attr_name}')"
            )
        return handler(val)

    ### Specializations

    # Casting
    def to_dtype_index_to_integer(self, value: Value, to_type: IrType) -> Value:
        return arith_d.index_cast(to_type, value)

    def to_dtype_index_to_float(self, value: Value, to_type: IrType) -> Value:
        # Cast index to integer, and then ask for a integer to float cast.
        # TODO: I don't really know how to query the machine bitwidth here,
        # so using 64.
        casted_to_int = arith_d.index_cast(IntegerType.get_signless(64), value)
        return self.to_dtype(IRProxyValue(casted_to_int), to_type).ir_value

    def to_dtype_integer_to_float(self, value: Value, to_type: IrType) -> Value:
        # sitofp
        casted_to_float = arith_d.sitofp(to_type, value)
        return self.to_dtype(IRProxyValue(casted_to_float), to_type).ir_value

    def to_dtype_float_to_float(self, value: Value, to_type: IrType) -> Value:
        # Check bitwidth to determine if we need to extend or narrow
        from_type = value.type
        from_bitwidth = self.get_float_bitwidth(from_type)
        to_bitwidth = self.get_float_bitwidth(to_type)
        if from_bitwidth < to_bitwidth:
            return arith_d.extf(to_type, value)
        elif from_bitwidth > to_bitwidth:
            return arith_d.truncf(to_type, value)
        else:
            raise CodegenError(f"NYI: Cast from {from_type} to {to_type}")

    # Binary integer/integer arithmetic.
    def binary_add_integer(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.addi(lhs.ir_value, rhs.ir_value))

    def binary_mul_integer(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.muli(lhs.ir_value, rhs.ir_value))

    def binary_sub_integer(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.subi(lhs.ir_value, rhs.ir_value))

    def binary_mod_integer(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.remsi(lhs.ir_value, rhs.ir_value))

    def binary_floordiv_integer(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.floordivsi(lhs.ir_value, rhs.ir_value))

    # Binary float arithmetic
    def binary_add_float(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.addf(lhs.ir_value, rhs.ir_value))

    def binary_mul_float(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.mulf(lhs.ir_value, rhs.ir_value))

    def binary_sub_float(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.subf(lhs.ir_value, rhs.ir_value))

    def binary_mod_float(self, lhs: IRProxyValue, rhs: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(arith_d.remf(lhs.ir_value, rhs.ir_value))

    def binary_truediv_float(
        self, lhs: IRProxyValue, rhs: IRProxyValue
    ) -> IRProxyValue:
        return IRProxyValue(arith_d.divf(lhs.ir_value, rhs.ir_value))

    def unary_exp2_float(self, val: IRProxyValue) -> IRProxyValue:
        return IRProxyValue(math_d.exp2(val.ir_value))


ScalarBuilder = _ScalarBuilder()
