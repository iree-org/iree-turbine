// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple}; // PyAny removed as it's less relevant now
use rayon::prelude::*;
use std::sync::Arc;

mod prune; // Contains the prune_envelope function

// Define PyRawEdge as a tuple struct to directly accept Python tuples.
// The order of fields in the Python tuple must match the order here:
// (from_node, to_node, delay, iter_diff)
#[derive(FromPyObject, Debug, Clone)]
struct PyRawEdge(u32, u32, f32, f32);

// Internal APLP computation function
fn compute_aplp_internal(
    node_count: usize,
    raw_edges: Vec<PyRawEdge>, // Takes Vec of PyRawEdge (tuple structs)
) -> Vec<Vec<Vec<(f32, f32)>>> {
    let mut d_current_vec_vec: Vec<Vec<Vec<(f32, f32)>>> =
        vec![vec![Vec::new(); node_count]; node_count];

    for i in 0..node_count {
        d_current_vec_vec[i][i] = prune::prune_envelope(vec![(0.0f32, 0.0f32)]);
    }

    for edge in raw_edges {
        let u_idx = edge.0 as usize;
        let v_idx = edge.1 as usize;
        let delay = edge.2;
        let iter_diff = edge.3;

        if u_idx < node_count && v_idx < node_count {
            let new_path_tuple = (delay, iter_diff);
            let mut current_paths_at_uv = d_current_vec_vec[u_idx][v_idx].clone();
            current_paths_at_uv.push(new_path_tuple);
            d_current_vec_vec[u_idx][v_idx] = prune::prune_envelope(current_paths_at_uv);
        }
    }

    let mut d_current_arc = Arc::new(d_current_vec_vec);

    for k_idx in 0..node_count {
        let d_next_rows: Vec<Vec<Vec<(f32, f32)>>> = (0..node_count)
            .into_par_iter()
            .map_with(Arc::clone(&d_current_arc), |d_curr_local, i_idx| {
                let mut row_j_updates: Vec<Vec<(f32, f32)>> = vec![Vec::new(); node_count];
                let paths_from_i_to_k = &d_curr_local[i_idx][k_idx];

                for j_idx in 0..node_count {
                    let paths_from_k_to_j = &d_curr_local[k_idx][j_idx];
                    let mut candidate_paths_for_ij = d_curr_local[i_idx][j_idx].clone();

                    if !paths_from_i_to_k.is_empty() && !paths_from_k_to_j.is_empty() {
                        for &(d_ik, id_ik) in paths_from_i_to_k.iter() {
                            if d_ik == -std::f32::INFINITY {
                                continue;
                            }
                            for &(d_kj, id_kj) in paths_from_k_to_j.iter() {
                                if d_kj == -std::f32::INFINITY {
                                    continue;
                                }
                                candidate_paths_for_ij.push((d_ik + d_kj, id_ik + id_kj));
                            }
                        }
                    }
                    row_j_updates[j_idx] = prune::prune_envelope(candidate_paths_for_ij);
                }
                row_j_updates
            })
            .collect();
        d_current_arc = Arc::new(d_next_rows);
    }
    Arc::try_unwrap(d_current_arc).unwrap_or_else(|_| panic!("Arc unwrap failed"))
}

// This is the function that will be callable from Python
#[pyfunction]
fn perform_aplp_pyo3(
    py: Python, // PyO3 GIL token
    node_count: usize,
    edges_pylist: &Bound<'_, PyList>, // Expect a Python list of edge data (tuples)
) -> PyResult<PyObject> {
    // Return a Python object (e.g., a PyDict)
    if node_count == 0 {
        return Ok(PyDict::new_bound(py).into()); // Return empty dict
    }

    // Convert PyList of Python tuples to Vec<PyRawEdge>
    // This will use the derived FromPyObject for the PyRawEdge tuple struct.
    let mut raw_edges_vec: Vec<PyRawEdge> = Vec::with_capacity(edges_pylist.len());
    for py_edge_item_any in edges_pylist.iter() {
        let edge: PyRawEdge = py_edge_item_any.extract()?; // This will now work for tuples
        raw_edges_vec.push(edge);
    }

    // Release the GIL for the duration of the computation.
    let final_d_matrix = py.allow_threads(|| compute_aplp_internal(node_count, raw_edges_vec));

    // Convert the Rust result (Vec<Vec<Vec<(f32,f32)>>>) to a Python dictionary
    // The dictionary will map (u_idx, v_idx) to a list of (delay, iter_diff) tuples
    let result_dict = PyDict::new_bound(py);
    for r_idx in 0..node_count {
        for c_idx in 0..node_count {
            if !final_d_matrix[r_idx][c_idx].is_empty() {
                let paths_for_pair_py_list = PyList::empty_bound(py);
                for &(delay, iter_diff) in &final_d_matrix[r_idx][c_idx] {
                    let path_tuple =
                        PyTuple::new_bound(py, &[delay.into_py(py), iter_diff.into_py(py)]);
                    paths_for_pair_py_list.append(path_tuple)?;
                }
                let key = PyTuple::new_bound(py, &[r_idx.into_py(py), c_idx.into_py(py)]);
                result_dict.set_item(key, paths_for_pair_py_list)?;
            }
        }
    }
    Ok(result_dict.into())
}

#[pymodule]
fn aplp_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perform_aplp_pyo3, m)?)?;
    Ok(())
}
