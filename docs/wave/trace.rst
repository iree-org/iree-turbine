Generating Thread Trace on ROCM-7
=================================

This document details how to set up ROCprof Compute Viewer on your device. ROCprof visualizes generated thread traces, including dispatching, runtime, and scheduling of lower-level IR.

1. Install ROCprof Compute Viewer
---------------------------------

**Get the ROCprof Compute Viewer**

- **Windows**

  1. Visit the ROCprof Compute Viewer release page and download version **0.1.1**:

     https://github.com/ROCm/rocprof-compute-viewer/releases/tag/0.1.1

  2. Extract the archive and launch the viewer from the `bin` directory:

     .. code-block:: powershell

        cd /path/to/ROCprof-Compute-Viewer/bin
        .\rocprof-compute-viewer.exe

- **Linux/macOS**

  1. Clone and build from source:

     .. code-block:: sh

        git clone https://github.com/ROCm/rocprof-compute-viewer.git
        cd rocprof-compute-viewer
        mkdir build && cd build
        cmake ..
        make -j$(nproc)
        sudo make install

     Follow any additional upstream instructions for your platform.

  2. Launch the viewer:

     .. code-block:: sh

        rocprof-compute-viewer

2. ROCM 7 Docker Setup
-----------------------

.. code-block:: sh

    docker pull rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha
    docker run --name <yourname>_rocm7 -it --network=host --ipc=host \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --cap-add=SYS_PTRACE -p8000:8000 -p18000:18000 \
      --security-opt seccomp=unconfined -v "$HOME":"$HOME" \
      --workdir $HOME \
      rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha
    docker start <yourname>_rocm7
    docker attach <yourname>_rocm7

3. Build rocprofiler-sdk-build
------------------------------

.. code-block:: sh

    git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source

    apt install libdw-dev

    cmake -B rocprofiler-sdk-build \
      -DCMAKE_INSTALL_PREFIX=rocprofiler-sdk-build/install \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      rocprofiler-sdk-source

    cmake --build rocprofiler-sdk-build \
      --target all --parallel $(nproc)

4. Download rocprof-trace-decode
--------------------------------

.. code-block:: sh

    wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.sh
    chmod +x rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.sh

    ./rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.sh
    # Follow direction and let install to your choice of path.

5. Run and Get Trace
--------------------

.. code-block:: sh

    /path/to/rocprofiler-sdk-build/bin/rocprofv3 \
      --att \
      --att-library-path /path/to/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux/opt/rocm/lib/ \
      -d your_trace_output_dir -- \
      <your program/command to trace>

6. Move Trace to Device with ROCprof Viewer
--------------------------------------------------------------------------------

.. code-block:: sh

    # On serverside
    tar -cvf your_trace_output_dir.tar your_trace_output_dir/

    # On local client/laptop with GUI
    scp your_server:/path/to/your_trace_output_dir.tar .
    tar -xf your_trace_output_dir.tar

    # Then use the viewer on this UI directory

7. Inspect that the Output Directory has
--------------------------------------------------

.. code-block:: sh

    ls your_trace_output_dir

    # Output expected to show directories such as:
    # <your server name>  stats_ui_output_agent_43452_dispatch_1.csv  ui_output_agent_43452_dispatch_1
    # <your server name>  stats_ui_output_agent_43452_dispatch_76.csv  ui_output_agent_43452_dispatch_76
    # <your server name>  stats_ui_output_agent_43452_dispatch_79.csv  ui_output_agent_43452_dispatch_79
    #   ...

The main dispatch to watch out for in the above example is ``ui_output_agent_43452_dispatch_1``.

8. Filter Kernels with att.json
---------------------------------------

If you want to filter for only certain dispatches or kernels, follow these instructions:

1. Save file below as ``att.json`` in your server

   .. code-block:: json

       {
           "jobs": [
               {
                   "kernel_include_regex": "<name of kernel, shapes/sizes, etc.>",
                   "kernel_exclude_regex": "",
                   "kernel_iteration_range": "[1]",
                   "advanced_thread_trace": true,
                   "att_parse" : "trace",
                   "att_target_cu" : 0,
                   "att_shader_engine_mask" : "0xF",
                   "att_simd_select": "0xF",
                   "att_buffer_size": "0x60000000"
               }
           ]
       }

2. Adjust ``kernel_include_regex`` or ``kernel_exclude_regex`` to filter out based on kernel names.

3. Re-run trace instructions with ``-i /path/to/att.json`` arg

   .. code-block:: sh

       /path/to/rocprofiler-sdk-build/bin/rocprofv3 \
         --att \
         --att-library-path /path/to/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux/opt/rocm/lib/ \
         -i /path/to/att.json \
         -d <your_trace_output_dir> -- \
         <your program/command to trace>
