#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

shapes = (
    "2048x10240x1280",
    "2048x1280x1280",
    "2048x1280x5120",
    "128x1280x2048",
    "8192x5120x640",
)
providers = {
    "wave": (380.7, 244.9, 287.4, 48.9, 370.2),
    "triton": (287.2, 170.4, 224.6, 26.2, 316.78),
    "pytorch(rocblas)": (478.97, 169.38, 204.61, 31, 454.6),
}

x = np.arange(len(shapes))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in providers.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=6)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.rc("font", weight="bold")
ax.set_ylabel("TFLOPS/s", fontsize=10, weight="bold")
ax.set_title("")
ax.set_xticks(x + width)
ax.set_xticklabels(shapes, fontsize=8, rotation=0)
plt.yticks(fontsize=8, weight="bold")
plt.xticks(weight="bold")
ax.legend(loc="upper right", ncols=3, prop={"size": 8}, frameon=False)
ax.set_ylim(0, 550)

plt.savefig("gemm_performance.svg", dpi=300, format="svg", bbox_inches="tight")
