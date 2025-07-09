Getting Started with Wave
=========================

This guide will help you get up and running with Wave, a high-performance machine learning programming language designed for accelerating ML kernel development.

Prerequisites
--------------

Before installing Wave, ensure you have the following prerequisites:

1. Python 3.10 or later
2. PyTorch
3. ROCm (for AMD GPU support)
4. A compatible AMD GPU with ROCm support (MI250, MI300, etc.)
5. Rust 1.70 or later

Installation
-------------

1. Install PyTorch with ROCm support:

   .. code-block:: bash

      pip install -r pytorch-rocm-requirements.txt

2. Install Rust:

   .. code-block:: bash

      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

3. Install Wave and its dependencies:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install -r requirements-wave-runtime.txt


Next Steps
-----------

- Read the :doc:`system_architecture` guide to understand Wave's compilation pipeline
- Check out the :doc:`gemm_tutorial` for a more complex example
- Explore :doc:`shared_memory` for optimization techniques
- Learn about the :doc:`runtime` for advanced usage

For more detailed information about Wave's architecture and optimization passes, see the :doc:`system_architecture` documentation.
