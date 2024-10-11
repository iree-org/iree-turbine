#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

shapes = (
    "4.0\n16.0\n1024.0\nd128",
    "4.0\n16.0\n1024.0\nd64",
    "4.0\n16.0\n4096.0\nd128",
    "4.0\n16.0\n4096.0\nd64",
    "1.0\n16.0\n16384.0\nd128",
    "1.0\n16.0\n16384.0\nd64",
    "4.0\n48.0\n2048.0\nd128",
    "4.0\n48.0\n2048.0\nd64",
    "4.0\n48.0\n16384.0\nd128",
    "4.0\n48.0\n16384.0\nd64",
)
providers = {
    "wave": (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    "triton": (321.033856, 143.625658, 384.978024, 330.811170, 369.061602, 345.366595, 357.773456, 314.488573, 420.738852, 359.682056),
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
ax.set_xticklabels(shapes, fontsize=6, rotation=0)
plt.yticks(fontsize=8, weight="bold")
plt.xticks(weight="bold")
ax.legend(loc="upper right", ncols=3, prop={"size": 8}, frameon=False)
ax.set_ylim(0, 550)

plt.savefig("attention_performance.svg", dpi=300, format="svg", bbox_inches="tight")
