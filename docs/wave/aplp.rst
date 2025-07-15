.. default-role:: code

All Pairs Longest Paths for Software Pipelining
===============================================

This document explains the Rust library designed for All-Pairs Longest Path (APLP) computation, specifically tailored for software pipelining. The library consists of two main modules: `prune.rs` for optimizing path representations and `lib.rs` for the core APLP logic and Python FFI (Foreign Function Interface) using PyO3.

The "paths" are represented as pairs `(delay, iter_diff)`, which define a line :math:`L(S) = \text{delay} - \text{iter_diff} \cdot S`, where :math:`S` is the symbolic initiation interval. The goal of pruning is to find the upper envelope of these lines.

prune.rs: Path Pruning Logic
----------------------------

The `prune.rs` module contains the `prune_envelope` function, which is responsible for taking a list of candidate paths (each represented as a `(delay: f32, iter_diff: f32)` tuple) for a single pair of nodes and returning a minimal list of paths that form the upper envelope. This means that for any given initiation interval `S`, one of the paths in the pruned list will provide the maximum (longest) path value.

**Purpose:**
To reduce the number of paths that need to be considered for each pair of nodes by eliminating paths that are "dominated" by others across all possible non-negative values of the initiation interval `S`.

**Algorithm (`prune_envelope`):**
The function implements a variation of Andrew's monotone chain algorithm, which is commonly used for finding convex hulls. The steps are:

1.  **Handle Empty Input:** If the input list of paths is empty, return an empty list.
2.  **Initial Filter & Sort:**

    * Filter out exact duplicate paths (considering floating point precision).
    * Sort paths primarily by delay (descending) and then by iteration difference (ascending) for a canonical order. This helps in the subsequent steps.
3.  **Separate Path Types:**

    * Paths with `-infinity` delay are handled separately. If only such paths exist, those with the numerically smallest `iter_diff` are kept (as :math:`L(S) = -\infty - \text{iter_diff} \cdot S` means a smaller `iter_diff` is "less negative" or "longer" when :math:`S > 0`).
    * Paths with finite delays are processed further. If no finite paths exist, the result from infinite paths is returned.
4.  **Transform to Lines:** Finite paths `(delay, iter_diff)` are transformed into lines represented as `(slope, intercept)`, where:

    * `slope (m) = -iter_diff`
    * `intercept (c) = delay`

      So, the line equation becomes :math:`L(S) = c + m \cdot S`.
5.  **Filter Unique Slopes:** For lines with the same slope, only the one with the highest intercept is kept, as it will always be above or equal to others with the same slope. The lines are then sorted by slope `m` in ascending order.
6.  **Build Upper Envelope (Monotone Chain Scan):** This is the core adaptation of Andrew's algorithm.

    * Iterate through the sorted lines `(m, c)`.
    * Maintain a list (`envelope_lines_mc`) of lines currently forming the upper envelope.
    * For each `current_line`, check if adding it to `envelope_lines_mc` would make the previously second-to-last line in the envelope redundant. This is done by checking the "turn" direction formed by the last three lines (the two in the envelope and the current one). If they don't form a "right turn" (i.e., the middle line is below or on the segment formed by the other two, indicating it's not part of the convex upper envelope), the last line is popped from `envelope_lines_mc`.
    * The `current_line` is then added to `envelope_lines_mc`.
7.  **Convert Back:** The lines in `envelope_lines_mc` are converted back from `(slope, intercept)` to `(delay, iter_diff)` format. `iter_diff` is rounded as it typically represents an integer count.

**Detailed Explanation: Andrew's Monotone Chain for Upper Envelope**

Andrew's monotone chain algorithm is typically used to find the convex hull of a set of 2D points. It works by first sorting the points (usually by x-coordinate, then y-coordinate) and then constructing the upper and lower hulls in separate passes. For our purpose of finding the upper envelope of lines :math:`L(S) = c + mS` (where :math:`c=\text{delay}`, :math:`m=-\text{iter_diff}`), we adapt this:

1.  **Point Representation:** We consider the lines in their dual form or by their parameters. In our case, we sort lines by their slopes `m` (which is `-iter_diff`). If slopes are equal, we only keep the line with the highest intercept `c` (delay), as it dominates others with the same slope.
2.  **Monotonicity:** The algorithm relies on processing points (or in our case, lines sorted by slope) in a specific order.
3.  **Building the Hull (Upper Envelope):**

    * We iterate through the unique-slope lines, sorted by slope `m`.
    * We maintain a candidate list for the upper envelope (e.g., `envelope_lines_mc`).
    * When considering adding a new line (`current_line`) to the envelope:

        * Let the last two lines in the envelope be `L1` (second to last) and `L2` (last).
        * We check if the sequence `L1, L2, current_line` maintains the convexity required for an upper envelope. For an upper envelope of lines :math:`y = mx + c` where lines are sorted by increasing slope :math:`m`, we need the intersection point of :math:`(L1, L2)` to be to the left of the intersection point of :math:`(L2, \text{current_line})`.
        * This is equivalent to checking the "turn" direction. If adding `current_line` causes a "non-right turn" (i.e., a left turn or collinearity that makes `L2` redundant), `L2` is removed from the envelope. This check is repeated until the condition is met or the envelope has fewer than two lines.
        * The geometric check can be performed using a cross-product like condition without explicitly calculating intersection points:
          For lines :math:`L_1=(m_1, c_1)`, :math:`L_2=(m_2, c_2)`, and :math:`L_3=(m_3, c_3)` with :math:`m_1 < m_2 < m_3`:
          :math:`L_2` is part of the upper envelope if the intersection of :math:`L_1, L_2` occurs at an :math:`S`-value less than the intersection of :math:`L_2, L_3`.
          The intersection :math:`S`-value for :math:`L_a, L_b` is :math:`S_{ab} = (c_b - c_a) / (m_a - m_b)`.
          So, we need :math:`(c_2 - c_1) / (m_1 - m_2) < (c_3 - c_2) / (m_2 - m_3)`.
          Rearranging to avoid division (and being careful with signs and slope ordering, :math:`m_1 < m_2 < m_3` implies :math:`m_1-m_2 < 0` and :math:`m_2-m_3 < 0`):
          :math:`(c_2 - c_1)(m_2 - m_3) > (c_3 - c_2)(m_1 - m_2)` (for strictly convex).
          The implementation uses `(c1 - c2)*(current_m - m2) >= (c2 - current_c)*(m2 - m1)` to pop `L2` if it's redundant. This formulation correctly identifies when `L2` is "below" the segment formed by `L1` and `current_line` or collinear in a way that makes it non-essential for the upper envelope.
    * After the check, `current_line` is added to the envelope.
4.  **Result:** The final list `envelope_lines_mc` contains the lines that constitute the upper envelope.

This process ensures that only the lines that are maximal for some range of `S` are kept. The overall time complexity is dominated by the initial sort, making it :math:`O(N \log N)` where :math:`N` is the number of initial lines.

**Visualization of Upper Envelope Construction (Monotone Chain Idea):**

.. mermaid::

    graph TD
        Start["Start with lines sorted by slope m: L1, L2, L3, ..."] --> P1["Initialize Envelope_List = []"];
        P1 --> ForEach["For each Line_current (Lc) in sorted lines:"];
        ForEach --> CheckSize{"len(Envelope_List) < 2?"};
        CheckSize -- Yes --> AddLc1["Add Lc to Envelope_List"];
        AddLc1 --> ForEach;
        CheckSize -- No --> GetPrevLines["L2 = Envelope_List.last()\nL1 = Envelope_List.second_last()"];
        GetPrevLines --> TurnCheck{"Is L1-L2-Lc a 'right turn' (maintains upper convexity)?"};
        TurnCheck -- No (L2 is redundant) --> PopL2["Pop L2 from Envelope_List"];
        PopL2 --> CheckSize2{"len(Envelope_List) < 2?"};
        CheckSize2 -- Yes --> AddLc2["Add Lc to Envelope_List"];
        AddLc2 --> ForEach;
        CheckSize2 -- No --> GetPrevLines;
        TurnCheck -- Yes --> AddLc3["Add Lc to Envelope_List"];
        AddLc3 --> ForEach;
        ForEach -- All lines processed --> End["End: Envelope_List contains the upper envelope lines"];

**References for Convex Hull Algorithms:**

* A.M. Andrew, "Another efficient algorithm for convex hulls in two dimensions", Info. Proc. Letters 9, 216-219 (1979).
* Joseph O'Rourke, "Computational Geometry in C", 2nd Edition, Cambridge University Press (1998). (Chapter on Convex Hulls)
* Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein, "Introduction to Algorithms", 3rd Edition, MIT Press (2009). (Chapter 33: Computational Geometry)

**Mermaid Diagram for `prune_envelope` (Overall Flow):**

.. mermaid::

    graph TD
        A["Input: List of (delay, iter_diff) paths"] --> B{Handle -INF paths};
        B -- Finite Paths --> C["Transform to lines: (m=-iter_diff, c=delay)"];
        B -- Only -INF Paths --> D["Keep paths with min iter_diff"];
        C --> E["Filter unique slopes, keeping max intercept"];
        E --> F["Sort lines by slope 'm'"];
        F --> G["Build upper envelope (Monotone Chain Scan - see detailed diagram above)"];
        G --> H["Convert envelope lines back to (delay, iter_diff)"];
        H --> Z["Output: Pruned list of paths"];
        D --> Z;

lib.rs: APLP Computation and Python Interface
---------------------------------------------

The `lib.rs` module orchestrates the All-Pairs Longest Path computation and exposes the functionality to Python using PyO3.

**Data Structures:**

* **`PyRawEdge(u32, u32, f32, f32)`:** A Rust tuple struct that maps directly to Python tuples `(from_node_idx, to_node_idx, delay, iter_diff)` passed from Python. It uses `#[derive(FromPyObject)]` for automatic conversion.
* **Internal Path Representation:** Within Rust, paths for each pair of nodes `(u,v)` are stored as `Vec<(f32, f32)>`, representing the list of `(delay, iter_diff)` tuples that form the upper envelope for that pair.

**Core Logic (`compute_aplp_internal`):**

This function implements the Floyd-Warshall algorithm to compute APLP.

1.  **Initialization:**
    * A 3D vector `d_current_vec_vec[i][j]` is initialized. Each element `d_current_vec_vec[u_idx][v_idx]` stores a `Vec<(f32, f32)>` representing the pruned paths from node `u` to node `v`.
    * For self-paths: `d_current_vec_vec[i][i]` is initialized to `[(0.0, 0.0)]` (a zero-delay, zero-iteration-difference path from a node to itself) after pruning.
    * For direct edges `(u,v)` from the input `raw_edges`: the tuple `(edge.delay, edge.iter_diff)` is added to the list in `d_current_vec_vec[u_idx][v_idx]`, which is then pruned.
2.  **Floyd-Warshall Iteration:**
    * The algorithm iterates `k_idx` from `0` to `node_count - 1` (representing the intermediate node).
    * **Parallelization (Rayon):** For each `k_idx`, the computation of rows `i_idx` is parallelized using `rayon::into_par_iter()`.

    * An `Arc` (Atomically Reference Counted pointer) is used to safely share the `d_current_vec_vec` matrix (from the previous `k` iteration) among worker threads.
    * Each worker thread processes one or more rows `i_idx`.

    * **Inner Loops (Worker Thread):** For each pair of nodes `(i_idx, j_idx)`:
        * It considers paths from `i_idx` to `k_idx` and from `k_idx` to `j_idx`.
        * If such sub-paths exist, they are combined:

            `new_delay = d_ik + d_kj`
            `new_iter_diff = id_ik + id_kj`

        * These newly formed paths are added to the existing list of paths for `(i_idx, j_idx)`.
        * The combined list is then pruned using `prune::prune_envelope`.
        * The result is stored in a `d_next_rows` structure.

    * After all rows `i_idx` are processed for the current `k_idx`, `d_current_arc` is updated to point to the newly computed matrix (from `d_next_rows`).
3.  **Result:** After all `k_idx` iterations, the final `d_current_arc` contains the APLP results.

**FFI Function (`perform_aplp_pyo3`):**

This function is exposed to Python using the `#[pyfunction]` macro.

1.  **Input:** Takes `node_count: usize` and `edges_pylist: &PyList` (a Python list of edge tuples) as input.
2.  **Conversion:** Converts the Python list of edge tuples into a `Vec<PyRawEdge>` using `extract()` which leverages the `FromPyObject` derive on `PyRawEdge`.
3.  **Computation:** Calls `compute_aplp_internal` to perform the APLP. The `py.allow_threads(|| ...)` block releases the Python Global Interpreter Lock (GIL) during the potentially long computation, allowing Rust's Rayon parallelism to be effective.
4.  **Output Conversion:** Converts the resulting Rust matrix `Vec<Vec<Vec<(f32,f32)>>>` into a Python dictionary.
    * The dictionary keys are Python tuples `(u_idx, v_idx)`.
    * The dictionary values are Python lists of Python tuples `[(delay, iter_diff), ...]`.
5.  **Return:** Returns the Python dictionary to the Python caller.

**Python Module Definition (`aplp_rs_lib`):**
The `#[pymodule]` macro defines the Python module. The `perform_aplp_pyo3` function is added to this module, making it callable from Python as `aplp_rs_lib.perform_aplp_pyo3(...)`.

**Mermaid Diagram for APLP Computation Flow:**

.. mermaid::

    graph TD
        subgraph PythonSide [Python Caller]
        PyInput["Input: node_count, list_of_edge_tuples"]
        end

        subgraph RustFFI [Rust: perform_aplp_pyo3]
        direction LR
        ConvertPyInput["Convert Python list of edge tuples to Vec<PyRawEdge>"]
        CallInternal["Call compute_aplp_internal(node_count, rust_edges)"]
        ConvertRustOutput["Convert Rust D_matrix to Python Dict"]
        end

        subgraph RustInternalCompute [Rust: compute_aplp_internal]
        direction TB
        InitD["Initialize D matrix: D[i][i] = [(0,0)], direct edges + prune"]
        LoopK["Loop k from 0 to V-1 (Intermediate Node)"]
        subgraph ParallelForRowI [For each k: Parallelize 'i' Loop]
            direction TB
            MapI["map_with(d_current_arc, i_idx)"]
            subgraph WorkerForRowI ["Worker for row 'i'"]
                direction TB
                LoopJ["Loop j from 0 to V-1 (Destination Node)"]
                CombinePaths["Combine D[i][k] + D[k][j]"]
                AddToExisting["Add to existing D[i][j] paths"]
                Prune["Call prune_envelope()"]
                StoreResult["Store pruned D_next[i][j]"]
            end
            MapI --> WorkerForRowI
        end
        InitD --> LoopK
        LoopK --> ParallelForRowI
        ParallelForRowI --> CollectResults["Collect results into D_next matrix"]
        CollectResults --> UpdateD["Update D_current = D_next"]
        UpdateD --> LoopK
        LoopK -- After all k --> FinalDMatrix["Final D_matrix"]
        end

        subgraph FinalDMatrix
	    PyOutput["Output: Python Dict {(u,v): [(d,id), ...]}"]
        end

        PyInput --> RustFFI;
        ConvertPyInput --> CallInternal;
        CallInternal --> RustInternalCompute;
