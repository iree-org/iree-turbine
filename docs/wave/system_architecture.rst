Wave System Architecture
========================

This document provides a detailed explanation of Wave's compilation pipeline and optimization passes.

Architecture Overview
----------------------

Wave provides a layered architecture that bridges high-level Python code with low-level GPU execution:

.. image:: wave_pipeline.excalidraw.png
   :alt: Wave Compilation Pipeline
   :align: center

The diagram above illustrates Wave's compilation pipeline, showing how Python code is transformed through various stages to generate optimized GPU code.
