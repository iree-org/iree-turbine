// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use std::cmp::Ordering;

pub fn prune_envelope(paths: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    if paths.is_empty() {
        return Vec::new();
    }

    // Filter out duplicates and sort for canonical representation
    // This step helps in making the process deterministic.
    // Convert to (delay, iter_diff) where iter_diff is treated as f32 for sorting.
    let mut unique_raw_paths: Vec<(f32, f32)> = paths;
    unique_raw_paths.sort_by(|a, b| {
        (-a.0)
            .partial_cmp(&(-b.0))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
    });
    unique_raw_paths.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let finite_paths: Vec<(f32, f32)> = unique_raw_paths
        .iter()
        .filter(|&&(d, _)| d != -std::f32::INFINITY)
        .cloned()
        .collect();

    let infinite_paths: Vec<(f32, f32)> = unique_raw_paths
        .iter()
        .filter(|&&(d, _)| d == -std::f32::INFINITY)
        .cloned()
        .collect();

    if finite_paths.is_empty() {
        if !infinite_paths.is_empty() {
            let min_iter_diff_for_inf = infinite_paths
                .iter()
                .map(|&(_, i)| i)
                .fold(std::f32::INFINITY, f32::min);
            return infinite_paths
                .into_iter()
                .filter(|&(_, i)| (i - min_iter_diff_for_inf).abs() < 1e-9) // Compare floats with tolerance
                .collect();
        }
        return Vec::new();
    }

    // Transform finite paths to lines (m, c) where m = -iter_diff, c = delay
    let mut lines_mc: Vec<(f32, f32)> = finite_paths
        .iter()
        .map(|&(delay, iter_diff)| (-iter_diff, delay))
        .collect();

    // Handle lines with the same slope: keep the one with the highest intercept.
    // Sort by slope (m), then by intercept (c) in descending order to pick the max c.
    lines_mc.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| (-a.1).partial_cmp(&(-b.1)).unwrap_or(Ordering::Equal))
    });

    let mut unique_slope_lines: Vec<(f32, f32)> = Vec::new();
    if !lines_mc.is_empty() {
        unique_slope_lines.push(lines_mc[0]);
        for i in 1..lines_mc.len() {
            if (lines_mc[i].0 - lines_mc[i - 1].0).abs() > 1e-9 {
                // If slopes are different enough
                unique_slope_lines.push(lines_mc[i]);
            }
            // If slopes are same, lines_mc[i-1] was already the one with max intercept due to sort order
        }
    }
    lines_mc = unique_slope_lines;

    if lines_mc.len() <= 1 {
        return lines_mc.into_iter().map(|(m, c)| (c, -m)).collect();
    }

    // Build the upper envelope
    let mut envelope_lines_mc: Vec<(f32, f32)> = Vec::new();
    for &(current_m, current_c) in &lines_mc {
        while envelope_lines_mc.len() >= 2 {
            let (m2, c2) = envelope_lines_mc[envelope_lines_mc.len() - 1];
            let (m1, c1) = envelope_lines_mc[envelope_lines_mc.len() - 2];

            // Slopes m1, m2, current_m should be strictly increasing due to previous sort & unique.
            // (c1 - c2)*(current_m - m2) >= (c2 - current_c)*(m2 - m1)
            // Add a small epsilon for floating point comparisons if needed,
            // but direct comparison often works for this geometric check.
            if (c1 - c2) * (current_m - m2) >= (c2 - current_c) * (m2 - m1) {
                envelope_lines_mc.pop();
            } else {
                break;
            }
        }
        envelope_lines_mc.push((current_m, current_c));
    }

    // Convert lines from envelope back to (delay, iter_diff) format
    envelope_lines_mc
        .into_iter()
        .map(|(m, c)| (c, -m))
        .collect()
}
