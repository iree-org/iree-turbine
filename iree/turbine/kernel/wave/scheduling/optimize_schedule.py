from .verifier import ScheduleValidator as ScheduleModifier
import random
from typing import Callable, Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto
from iree.turbine.kernel.wave.scheduling.resources import (
    get_custom_operation_type,
    Operation,
)
from iree.turbine.kernel.ops.wave_ops import get_custom
from iree.turbine.kernel.wave.tuner.utils import latency_to_us, format_latency_us
import logging
import numpy as np


class OptimizationAlgorithm(Enum):
    HILL_CLIMBING = auto()
    # Add more algorithms here as needed


@dataclass
class OptimizationResult:
    schedule: Dict
    latency: float
    iterations: int
    algorithm: OptimizationAlgorithm
    improvement_history: List[float]


class ScheduleOptimizer:
    def __init__(
        self,
        validator: ScheduleModifier,
        measure_fn: Callable[[Dict], float],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.HILL_CLIMBING,
        logger: Optional[logging.Logger] = None,
        progress_file: Optional[str] = None,
        tuning_logger=None,
        random_seed: Optional[int] = None,
    ):
        """Initialize the schedule optimizer.

        Args:
            validator: A ScheduleModifier instance that validates and modifies schedules
            measure_fn: A function that takes a schedule and returns its latency
            algorithm: The optimization algorithm to use
            logger: Optional logger for tracking optimization progress
            progress_file: Optional path to progress file
            tuning_logger: Optional tuning logger for saving schedules
            random_seed: Optional seed for reproducible random number generation
        """
        self.validator = validator
        self.measure_fn = measure_fn
        self.algorithm = algorithm
        self.logger = logger
        self.progress_file = progress_file
        self.tuning_logger = tuning_logger
        self.current_best_schedule = None
        self.current_best_latency = float("inf")
        self.improvement_history = []
        self.current_iteration = 0

        # Initialize random number generator with seed for reproducibility
        self.rng = random.Random(random_seed) if random_seed is not None else random

        # Initialize progress file if provided
        if self.progress_file is not None:
            with open(self.progress_file, "w") as f:
                f.write("iteration,latency_us,is_improvement,is_best\n")

    def _write_progress(
        self, iteration: int, latency: float, is_improvement: bool, is_best: bool
    ) -> None:
        """Write progress to file if progress_file is set.

        Args:
            iteration: Current iteration number
            latency: Achieved latency
            is_improvement: Whether this is an improvement
            is_best: Whether this is the best latency so far
        """
        if self.progress_file is not None:
            latency_us = latency_to_us(latency)
            with open(self.progress_file, "a") as f:
                f.write(f"{iteration},{latency_us},{is_improvement},{is_best}\n")

    def _log_iteration(
        self, schedule: Dict, latency: float, is_improvement: bool
    ) -> None:
        """Log an optimization iteration.

        Args:
            schedule: Current schedule
            latency: Achieved latency
            is_improvement: Whether this is an improvement
        """
        if self.logger is None:
            return

        latency_str = format_latency_us(latency)

        if is_improvement:
            self.logger.info(
                f"Iteration {self.current_iteration}: Found improvement! "
                f"Latency: {latency_str}"
            )
        else:
            self.logger.debug(
                f"Iteration {self.current_iteration}: No improvement. "
                f"Latency: {latency_str}"
            )

    def _log_summary(self) -> None:
        """Log a summary of the optimization process."""
        if self.logger is None:
            return

        self.logger.info("\nOptimization Summary:")
        best_latency_str = format_latency_us(self.current_best_latency)
        improvement_history = [latency_to_us(h) for h in self.improvement_history]

        self.logger.info(f"Best latency: {best_latency_str}")
        self.logger.info(f"Total iterations: {self.current_iteration}")
        self.logger.info(f"Improvement history: {improvement_history}")

    def _measure_with_logging(self, schedule: Dict) -> float:
        """Measure schedule latency with logging.

        Args:
            schedule: Schedule to measure

        Returns:
            Measured latency
        """
        latency = self.measure_fn(schedule)
        self.current_iteration += 1
        return latency

    def _initialize_optimization(self, verbose: bool) -> Tuple[Dict, float]:
        """Initialize the optimization process with the current best schedule.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (current_best_schedule, current_best_latency)
        """
        if verbose and self.logger:
            self.logger.info("Starting Hill Climbing Optimization...")

        current_best_schedule, _ = self.validator.get_current_schedule_state()
        current_best_latency = self._measure_with_logging(current_best_schedule)
        self.improvement_history = [current_best_latency]

        # Write initial progress
        self._write_progress(0, current_best_latency, True, True)

        if verbose and self.logger:
            self.logger.info(
                f"Initial Best Latency: {format_latency_us(current_best_latency)} for schedule: {{ { {n.name: s for n,s in current_best_schedule.items()} } }}"
            )

        return current_best_schedule, current_best_latency

    def _get_schedulable_nodes(self) -> List:
        """Get list of nodes that can be scheduled (non-NOOP operations).

        Returns:
            List of schedulable nodes
        """
        return [
            n
            for n in self.validator.nodes
            if get_custom_operation_type_val(get_custom(n)) != Operation.NOOP
        ]

    def _select_random_move(
        self, schedulable_nodes: List, current_best_schedule: Dict
    ) -> Tuple:
        """Select a random node and generate a new target cycle for it.

        Args:
            schedulable_nodes: List of nodes that can be moved
            current_best_schedule: Current best schedule

        Returns:
            Tuple of (node_to_move, new_target_cycle)
        """
        node_to_move = self.rng.choice(schedulable_nodes)
        original_cycle = current_best_schedule[node_to_move]
        delta_cycle = self.rng.randrange(-self.validator.T, self.validator.T + 1)
        if delta_cycle == 0:
            delta_cycle = self.rng.choice([-1, 1])
        new_target_cycle = max(0, original_cycle + delta_cycle)

        return node_to_move, new_target_cycle

    def _evaluate_move(
        self,
        node_to_move,
        new_target_cycle: int,
        current_best_latency: float,
        verbose: bool,
    ) -> Tuple[bool, float, Optional[Dict], Optional[np.ndarray]]:
        """Evaluate a potential move and determine if it's an improvement.

        Args:
            node_to_move: Node to move
            new_target_cycle: New target cycle for the node
            current_best_latency: Current best latency
            verbose: Whether to print progress information

        Returns:
            Tuple of (is_improvement, new_latency, new_schedule, resource_table) or (False, current_latency, None, None) if invalid
        """
        if verbose and self.logger:
            self.logger.info(
                f"  Attempting to move node {node_to_move.name} to cycle {new_target_cycle}"
            )

        (
            is_valid_move,
            candidate_schedule,
            error_message,
        ) = self.validator.attempt_move(node_to_move, new_target_cycle)

        if not is_valid_move or candidate_schedule is None:
            return False, current_best_latency, None, None

        candidate_latency = self._measure_with_logging(candidate_schedule)

        # Skip invalid results (infinity indicates compilation failure)
        if candidate_latency == float("inf"):
            return False, current_best_latency, None, None

        is_improvement = candidate_latency < current_best_latency

        # Use tuning logger to save the schedule if available
        if self.tuning_logger is not None:
            self.tuning_logger.log_iteration(
                self.current_iteration,
                candidate_schedule,
                candidate_latency,
                is_improvement,
            )
        else:
            self._log_iteration(candidate_schedule, candidate_latency, is_improvement)

        self._write_progress(
            self.current_iteration, candidate_latency, is_improvement, is_improvement
        )

        if is_improvement and verbose and self.logger:
            self.logger.info(
                f"  *** Improvement found! New latency: {format_latency_us(candidate_latency)} (old: {format_latency_us(current_best_latency)}) ***"
            )

        # Get the resource table from the validator's current state
        _, resource_table = self.validator.get_current_schedule_state()
        return is_improvement, candidate_latency, candidate_schedule, resource_table

    def _update_best_solution(
        self,
        is_improvement: bool,
        candidate_latency: float,
        candidate_schedule: Dict,
        candidate_rt: Optional[np.ndarray],
        current_best_schedule: Dict,
        current_best_latency: float,
    ) -> Tuple[Dict, float, int]:
        """Update the best solution if an improvement is found.

        Args:
            is_improvement: Whether the candidate is an improvement
            candidate_latency: Latency of the candidate schedule
            candidate_schedule: Candidate schedule
            candidate_rt: Candidate resource table
            current_best_schedule: Current best schedule
            current_best_latency: Current best latency

        Returns:
            Tuple of (new_best_schedule, new_best_latency, no_improvement_streak)
        """
        if is_improvement:
            if candidate_rt is not None:
                self.validator.commit_move(candidate_schedule, candidate_rt)
            self.improvement_history.append(candidate_latency)
            return candidate_schedule, candidate_latency, 0
        else:
            return current_best_schedule, current_best_latency, 1

    def _log_final_results(
        self, current_best_latency: float, current_best_schedule: Dict, verbose: bool
    ) -> None:
        """Log the final optimization results.

        Args:
            current_best_latency: Final best latency
            current_best_schedule: Final best schedule
            verbose: Whether to print progress information
        """
        if verbose and self.logger:
            self.logger.info("\nOptimization Finished.")
            self.logger.info(
                f"Final Best Latency: {format_latency_us(current_best_latency)}"
            )
            self.logger.info(
                f"Final Best Schedule: {{ { {n.name: s for n,s in current_best_schedule.items()} } }}"
            )

    def _run_hill_climbing(
        self,
        max_iterations: int = 100,
        max_no_improvement: int = 20,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run hill climbing optimization algorithm.

        Args:
            max_iterations: Maximum number of iterations to run
            max_no_improvement: Maximum number of iterations without improvement before stopping
            verbose: Whether to print progress information

        Returns:
            OptimizationResult containing the best schedule and optimization metrics
        """
        # Initialize optimization
        current_best_schedule, current_best_latency = self._initialize_optimization(
            verbose
        )

        no_improvement_streak = 0
        iteration = 0

        while iteration < max_iterations:
            if verbose and self.logger:
                self.logger.info(f"\nIteration {iteration + 1}/{max_iterations}")

            # Get schedulable nodes
            schedulable_nodes = self._get_schedulable_nodes()
            if not schedulable_nodes:
                if verbose and self.logger:
                    self.logger.info(
                        "  No schedulable (non-NOOP) nodes to move. Stopping."
                    )
                break

            # Select and evaluate a random move
            node_to_move, new_target_cycle = self._select_random_move(
                schedulable_nodes, current_best_schedule
            )
            is_improvement, candidate_latency, candidate_schedule, candidate_rt = (
                self._evaluate_move(
                    node_to_move, new_target_cycle, current_best_latency, verbose
                )
            )

            # Update best solution if improvement found
            current_best_schedule, current_best_latency, streak_increment = (
                self._update_best_solution(
                    is_improvement,
                    candidate_latency,
                    candidate_schedule,
                    candidate_rt,
                    current_best_schedule,
                    current_best_latency,
                )
            )
            no_improvement_streak += streak_increment

            # Check early stopping condition
            if no_improvement_streak >= max_no_improvement:
                if verbose and self.logger:
                    self.logger.info(
                        f"\nStopping early: No improvement in {max_no_improvement} iterations."
                    )
                break

            iteration += 1

        # Log final results
        self._log_final_results(current_best_latency, current_best_schedule, verbose)

        # Update the instance variables for logging
        self.current_best_latency = current_best_latency
        self.current_iteration = iteration

        self._log_summary()

        return OptimizationResult(
            schedule=current_best_schedule,
            latency=current_best_latency,
            iterations=iteration,
            algorithm=OptimizationAlgorithm.HILL_CLIMBING,
            improvement_history=self.improvement_history,
        )

    def optimize(
        self,
        max_iterations: int = 100,
        max_no_improvement: int = 20,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run the selected optimization algorithm.

        Args:
            max_iterations: Maximum number of iterations to run
            max_no_improvement: Maximum number of iterations without improvement before stopping
            verbose: Whether to print progress information

        Returns:
            OptimizationResult containing the best schedule and optimization metrics
        """
        if self.algorithm == OptimizationAlgorithm.HILL_CLIMBING:
            return self._run_hill_climbing(
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported optimization algorithm: {self.algorithm}")


def get_custom_operation_type_val(custom: "CustomOp") -> str:
    """Get the string value of the operation type for a custom operation.

    Args:
        custom: The custom operation to get the type value for

    Returns:
        The string value of the operation type (e.g. "read_shared", "write_global", etc.)
    """
    op_type = get_custom_operation_type(custom)
    return op_type.value if op_type is not None else Operation.NOOP.value
