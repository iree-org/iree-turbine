from .._support.indexing import index_symbol

# Global symbols used throughout the code.

# Address spaces.
GLOBAL_ADDRESS_SPACE = index_symbol("$GLOBAL_ADDRESS_SPACE")
SHARED_ADDRESS_SPACE = index_symbol("$SHARED_ADDRESS_SPACE")

# Distribution symbols.
WORKGROUP_0 = index_symbol("$WG0")
WORKGROUP_1 = index_symbol("$WG1")
WORKGROUP_2 = index_symbol("$WG2")

THREAD_0 = index_symbol("$T0")
THREAD_1 = index_symbol("$T1")
THREAD_2 = index_symbol("$T2")

# MMA symbols
MMA_LHS = index_symbol("$MMA_LHS")
MMA_RHS = index_symbol("$MMA_RHS")
MMA_ACC = index_symbol("$MMA_ACC")
