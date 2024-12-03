from .._support.indexing import index_symbol
import sys

# Global symbols used throughout the code.

# Address spaces.
GLOBAL_ADDRESS_SPACE = index_symbol("$GLOBAL_ADDRESS_SPACE")
SHARED_ADDRESS_SPACE = index_symbol("$SHARED_ADDRESS_SPACE")


# Distribution symbols.
WORKGROUP_0 = index_symbol("$WG0")
WORKGROUP_1 = index_symbol("$WG1")
WORKGROUP_2 = index_symbol("$WG2")


def create_additional_workgroup_symbols():
    """
    Since we can have a large number of workgroups, we create
    symbols for them dynamically. However, only WORKGROUP_0,
    WORKGROUP_1, and WORKGROUP_2 will persist during code generation,
    so we generate those symbols statically.
    """
    max_workgroups = 5
    for i in range(3, max_workgroups):
        globals()[f"WORKGROUP_{i}"] = index_symbol(f"$WG{i}")


def get_workgroup_symbol(i: int):
    assert i >= 0, "Workgroup index must be non-negative."
    return index_symbol(f"$WG{i}")


create_additional_workgroup_symbols()

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
