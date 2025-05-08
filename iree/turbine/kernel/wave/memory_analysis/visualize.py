# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def visualize_memory_allocations(
    allocations_data, allocation_offsets, peak_memory_usage, filename
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        import seaborn as sns

        sns.set_theme()
    except ImportError:
        raise ImportError(
            "matplotlib is not installed. Please install it using 'pip install matplotlib'"
        )

    # --- Prepare Data for Plotting ---
    # Extract relevant data points
    sizes = [x[0] for x in allocations_data]
    start_times = [x[1] for x in allocations_data]
    end_times = [x[2] for x in allocations_data]
    offsets = [x for x in allocation_offsets]

    # Calculate rectangle properties for matplotlib patches
    # x-coordinate: start time
    # y-coordinate: offset
    # width: end_time - start_time + 1 (inclusive)
    # height: size
    rect_x = [start_times[i] for i in range(len(allocations_data))]
    rect_y = [offsets[i] for i in range(len(allocations_data))]
    rect_width = [
        end_times[i] - start_times[i] + 1 for i in range(len(allocations_data))
    ]
    rect_height = [sizes[i] for i in range(len(allocations_data))]

    # Define a list of hatch patterns
    hatch_patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    # --- Create the Plot using Matplotlib with Seaborn Style ---
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    # Add rectangles to the plot
    for i in range(len(allocations_data)):
        # Create a rectangle patch
        rect = patches.Rectangle(
            (rect_x[i], rect_y[i]),  # (x, y) bottom-left corner
            rect_width[i],  # width
            rect_height[i],  # height
            linewidth=2,  # Thicker border
            edgecolor="black",  # Border color
            facecolor=sns.color_palette("viridis")[
                i % len(allocations_data)
            ],  # Use a seaborn color palette
            alpha=0.8,  # Transparency
            hatch=hatch_patterns[i % len(hatch_patterns)],  # Add hatch pattern
        )
        # Add the patch to the axes
        ax.add_patch(rect)

        # Add text label (Allocation ID) in the center of the rectangle
        ax.text(
            rect_x[i] + rect_width[i] / 2,  # x-coordinate (center)
            rect_y[i] + rect_height[i] / 2,  # y-coordinate (center)
            f"Alloc {i}",  # Text to display
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            color=(
                "white"
                if sns.color_palette("viridis")[i % len(allocations_data)][0] < 0.5
                else "black"
            ),  # Choose text color based on background luminance
            fontsize=9,
            weight="bold",
        )

    # --- Set Plot Limits and Labels ---
    # Set x-axis limits to cover the time range
    min_time = min(start_times)
    max_time = max(end_times)
    ax.set_xlim(min_time - 0.5, max_time + 0.5)  # Add some padding

    # Set y-axis limits to cover the memory range up to peak usage
    min_offset = 0
    max_memory = peak_memory_usage
    ax.set_ylim(min_offset, max_memory + 5)  # Add some padding above peak

    # Set axis labels and title without LaTeX
    ax.set_xlabel("Time Step", fontsize=12, weight="bold")
    ax.set_ylabel("Memory Offset", fontsize=12, weight="bold")
    ax.set_title(
        "Heuristic Shared Memory Allocation Schedule", fontsize=14, weight="bold"
    )

    # Set x-axis ticks to be integers corresponding to time steps
    ax.set_xticks(np.arange(min_time, max_time + 1, 1))

    # Add a horizontal line indicating the peak memory usage without LaTeX in the label
    ax.axhline(
        y=peak_memory_usage,
        color="red",
        linestyle="--",
        label=f"Peak Usage: {peak_memory_usage}",
        linewidth=2,
    )
    ax.legend(loc="upper right")

    # Ensure layout is tight
    plt.tight_layout()
    plt.savefig(filename)
