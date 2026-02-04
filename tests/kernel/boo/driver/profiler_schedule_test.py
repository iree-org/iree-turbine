"""Unit tests for the profiler schedule logic in the BOO driver."""

import pytest
import torch
from torch.profiler import ProfilerAction

from iree.turbine.kernel.boo.driver.driver import (
    make_profiler_schedule,
    make_profiler_context,
)


def test_simple_case_no_cleanup():
    """
    Simple case: 10 iterations, 1 device, no cleanup needed.

    Expected schedule:
    - Step 0: NONE (warmup)
    - Steps 1-9: RECORD (9 iterations)
    - Step 10: RECORD_AND_SAVE (10th and final profiled iteration)

    Total profiled: 10 iterations
    """
    schedule_fn, total_num_iters, _ = make_profiler_schedule(
        timing_iter=10,
        num_devices=1,
        iter_thresh=100,  # High enough to avoid cleanup
    )

    assert total_num_iters == 11  # 10 + 1 warmup

    expected = [
        ProfilerAction.NONE,  # 0: warmup
        ProfilerAction.RECORD,  # 1
        ProfilerAction.RECORD,  # 2
        ProfilerAction.RECORD,  # 3
        ProfilerAction.RECORD,  # 4
        ProfilerAction.RECORD,  # 5
        ProfilerAction.RECORD,  # 6
        ProfilerAction.RECORD,  # 7
        ProfilerAction.RECORD,  # 8
        ProfilerAction.RECORD,  # 9
        ProfilerAction.RECORD_AND_SAVE,  # 10: last iteration
    ]

    actual = [schedule_fn(step) for step in range(total_num_iters)]
    assert actual == expected


def test_with_cleanup():
    """
    Case with cleanup: 12 iterations, 1 device, iter_thresh=4.

    Cleanup occurs every iter_thresh steps at steps 3, 7, 11, 15.

    Total profiled: 12 iterations
    """
    schedule_fn, total_num_iters, _ = make_profiler_schedule(
        timing_iter=12,
        num_devices=1,
        iter_thresh=4,
    )

    expected = [
        ProfilerAction.NONE,  # 0: warmup
        ProfilerAction.RECORD,  # 1
        ProfilerAction.RECORD_AND_SAVE,  # 2: save before cleanup
        ProfilerAction.NONE,  # 3: cleanup
        ProfilerAction.RECORD,  # 4
        ProfilerAction.RECORD,  # 5
        ProfilerAction.RECORD_AND_SAVE,  # 6: save before cleanup
        ProfilerAction.NONE,  # 7: cleanup
        ProfilerAction.RECORD,  # 8
        ProfilerAction.RECORD,  # 9
        ProfilerAction.RECORD_AND_SAVE,  # 10: save before cleanup
        ProfilerAction.NONE,  # 11: cleanup
        ProfilerAction.RECORD,  # 12
        ProfilerAction.RECORD,  # 13
        ProfilerAction.RECORD_AND_SAVE,  # 14: save before cleanup
        ProfilerAction.NONE,  # 15: cleanup
        ProfilerAction.RECORD_AND_SAVE,  # 16: last iteration
    ]

    actual = [schedule_fn(step) for step in range(total_num_iters)]
    assert actual == expected


def test_four_devices_no_cleanup():
    """
    4 iterations, 4 devices, no cleanup.

    Expected schedule:
    - Steps 0-3: NONE (warmup each device)
    - Steps 4-6: RECORD
    - Step 7: RECORD_AND_SAVE (last)

    Total profiled: 4 iterations
    """
    schedule_fn, total_num_iters, _ = make_profiler_schedule(
        timing_iter=4,
        num_devices=4,
        iter_thresh=100,
    )

    expected = [
        ProfilerAction.NONE,  # 0: warmup
        ProfilerAction.NONE,  # 1: warmup
        ProfilerAction.NONE,  # 2: warmup
        ProfilerAction.NONE,  # 3: warmup
        ProfilerAction.RECORD,  # 4
        ProfilerAction.RECORD,  # 5
        ProfilerAction.RECORD,  # 6
        ProfilerAction.RECORD_AND_SAVE,  # 7: last iteration
    ]

    actual = [schedule_fn(step) for step in range(total_num_iters)]
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profiler_integration():
    """
    Integration test: run torch profiler and verify it captures
    all requested iterations.
    """
    timing_iter = 100
    schedule_fn, total_num_iters, _ = make_profiler_schedule(
        timing_iter=timing_iter,
        num_devices=1,
        iter_thresh=16,
    )

    x = torch.zeros(1, device="cuda")

    def dummy_kernel():
        return x + 1

    with make_profiler_context(schedule_fn) as prof:
        for step in range(total_num_iters):
            dummy_kernel()
            prof.step()

    cuda_events = [
        e for e in prof.events() if e.device_type == torch.profiler.DeviceType.CUDA
    ]

    assert (
        len(cuda_events) == timing_iter
    ), f"Expected {timing_iter} CUDA events, got {len(cuda_events)}"
