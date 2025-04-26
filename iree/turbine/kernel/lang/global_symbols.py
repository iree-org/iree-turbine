from .._support.indexing import index_symbol
import sympy

# Global symbols used throughout the code.

# Address spaces.
GLOBAL_ADDRESS_SPACE = index_symbol("$GLOBAL_ADDRESS_SPACE")
SHARED_ADDRESS_SPACE = index_symbol("$SHARED_ADDRESS_SPACE")


# Distribution symbols.
WORKGROUP_0 = index_symbol("$WG0")
WORKGROUP_1 = index_symbol("$WG1")
WORKGROUP_2 = index_symbol("$WG2")


def get_workgroup_symbol(i: int):
    assert i >= 0, "Workgroup index must be non-negative."
    symbol_name = f"WORKGROUP_{i}"
    if symbol_name not in globals():
        globals()[symbol_name] = index_symbol(f"$WG{i}")
    return index_symbol(f"$WG{i}")


THREAD_0 = index_symbol("$T0")
THREAD_1 = index_symbol("$T1")
THREAD_2 = index_symbol("$T2")

# MMA symbols.
MMA_LHS = index_symbol("$MMA_LHS")
MMA_RHS = index_symbol("$MMA_RHS")
MMA_ACC = index_symbol("$MMA_ACC")
GPR_NUM = index_symbol("$GPR_NUM")

# Scheduling symbols.
READ_SHARED_DELAY = index_symbol("$READ_SHARED_DELAY")
WRITE_SHARED_DELAY = index_symbol("$WRITE_SHARED_DELAY")
READ_GLOBAL_DELAY = index_symbol("$READ_GLOBAL_DELAY")
WRITE_GLOBAL_DELAY = index_symbol("$WRITE_GLOBAL_DELAY")
MMA_DELAY = index_symbol("$MMA_DELAY")
VALU_DELAY = index_symbol("$VALU_DELAY")
SHUFFLE_DELAY = index_symbol("$SHUFFLE_DELAY")
SHARED_MEMORY_UNITS = index_symbol("$SHARED_MEMORY_UNITS")
GLOBAL_MEMORY_UNITS = index_symbol("$GLOBAL_MEMORY_UNITS")
MMA_UNITS = index_symbol("$MMA_UNITS")
VALU_UNITS = index_symbol("$VALU_UNITS")
SHUFFLE_UNITS = index_symbol("$SHUFFLE_UNITS")


# Iteration symbols.
def GET_ITER_ARG(i: int):
    return sympy.Symbol(f"$GET_ITER_ARG_{i}")
