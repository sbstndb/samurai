"""
Core progress bar functionality for Samurai simulations.

This module provides context managers and utilities for tracking progress in
time-dependent simulations with adaptive mesh refinement.
"""

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

try:
    from tqdm import tqdm as tqdm_class
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm_class = None  # type: ignore

if TYPE_CHECKING:
    import samurai_python

from .stats import MeshStatistics


class ProgressManager:
    """Base class for progress management using tqdm.

    This class provides common functionality for all progress tracking
    in Samurai simulations, including rate estimation and time formatting.

    Args:
        desc: Description for the progress bar
        disable: If True, disable progress bar display
        **kwargs: Additional arguments passed to tqdm
    """

    def __init__(
        self,
        desc: str = "Progress",
        disable: bool = False,
        **kwargs
    ):
        """Initialize progress manager.

        Args:
            desc: Description for the progress bar
            disable: If True, disable progress bar display
            **kwargs: Additional arguments passed to tqdm
        """
        if not TQDM_AVAILABLE and not disable:
            print("Warning: tqdm not available, progress bar disabled")
            disable = True

        self.desc = desc
        self.disable = disable
        self._tqdm_kwargs = kwargs
        self._pbar: Optional[tqdm_class] = None
        self._start_time: Optional[float] = None

    def _create_pbar(self, total: Optional[int] = None) -> None:
        """Create tqdm progress bar instance.

        Args:
            total: Total number of iterations (None for unknown)
        """
        if self.disable:
            return

        if not TQDM_AVAILABLE:
            return

        self._pbar = tqdm_class(
            total=total,
            desc=self.desc,
            **self._tqdm_kwargs
        )
        self._start_time = time.time()

    def _update_pbar(self, n: int = 1) -> None:
        """Update progress bar by n steps.

        Args:
            n: Number of steps to increment
        """
        if self._pbar is not None:
            self._pbar.update(n)

    def _close_pbar(self) -> None:
        """Close progress bar and display final statistics."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _set_postfix(self, **kwargs) -> None:
        """Update postfix statistics on progress bar.

        Args:
            **kwargs: Key-value pairs to display in progress bar postfix
        """
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs)

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration in human-readable format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "1:23:45" or "45.2s")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class TimeLoop(ProgressManager):
    """Context manager for time-stepping loops with progress tracking.

    This class manages progress bars for time-dependent simulations,
    handling variable time steps and providing ETA estimates.

    Example:
        >>> with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
        ...     while pbar.continue_loop():
        ...         # Simulation step
        ...         pbar.update_stats(mesh=mesh)

    Args:
        Tf: Final simulation time
        dt: Initial time step (for estimation, can be variable)
        desc: Description for progress bar
        disable: If True, disable progress bar
        **kwargs: Additional arguments passed to tqdm
    """

    def __init__(
        self,
        Tf: float,
        dt: float,
        desc: str = "Time loop",
        disable: bool = False,
        **kwargs
    ):
        """Initialize time loop progress tracker.

        Args:
            Tf: Final simulation time
            dt: Initial time step (for estimation)
            desc: Description for progress bar
            disable: If True, disable progress bar
            **kwargs: Additional arguments passed to tqdm
        """
        super().__init__(desc=desc, disable=disable, **kwargs)

        self.Tf = Tf
        self.dt_initial = dt
        self.t = 0.0
        self.iteration = 0
        self._estimated_steps = int(Tf / dt) if dt > 0 else None

        # Mesh statistics tracking
        self._mesh_stats: Optional[MeshStatistics] = None
        self._track_mesh = False

        # Create progress bar
        self._create_pbar(total=self._estimated_steps)

    def __enter__(self) -> "TimeLoop":
        """Enter context manager.

        Returns:
            Self for method chaining
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close progress bar.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self._close_pbar()

    def continue_loop(self) -> bool:
        """Check if simulation should continue.

        Returns:
            True if simulation time is less than final time
        """
        return self.t < self.Tf

    def advance_time(self, dt: Optional[float] = None) -> None:
        """Advance simulation time and update progress bar.

        Args:
            dt: Time step to advance (uses initial dt if None)
        """
        if dt is None:
            dt = self.dt_initial

        self.t += dt
        self.iteration += 1
        self._update_pbar(1)

        # Update progress bar with current time
        if self._pbar is not None:
            pct = (self.t / self.Tf) * 100
            self._pbar.set_description(f"{self.desc} ({self.t:.3f}/{self.Tf:.3f}s)")

    def update_stats(
        self,
        mesh: Optional["samurai_python.Mesh"] = None,
        **stats
    ) -> None:
        """Update progress bar with mesh and custom statistics.

        Args:
            mesh: Mesh to extract statistics from
            **stats: Additional custom statistics to display

        Example:
            >>> pbar.update_stats(mesh=mesh, residual=1e-6)
        """
        if mesh is not None:
            # Enable mesh tracking on first call
            if not self._track_mesh:
                self._track_mesh = True
                self._mesh_stats = MeshStatistics(enable_level_breakdown=False)

            # Update mesh statistics
            self._mesh_stats.update(mesh)
            stats.update({
                "cells": self._mesh_stats.n_cells,
                "levels": f"{self._mesh_stats.min_level}-{self._mesh_stats.max_level}",
            })

        # Update progress bar postfix
        if stats:
            self._set_postfix(**stats)

    def get_eta(self) -> float:
        """Get estimated time to completion.

        Returns:
            Estimated seconds remaining
        """
        if self._start_time is None or self.iteration == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        rate = self.iteration / elapsed
        remaining = self._estimated_steps - self.iteration if self._estimated_steps else 0

        if rate > 0:
            return remaining / rate
        return 0.0

    @property
    def progress(self) -> float:
        """Get simulation progress as fraction [0, 1]."""
        return min(self.t / self.Tf, 1.0) if self.Tf > 0 else 0.0


class IterationLoop(ProgressManager):
    """Context manager for fixed-count iteration loops.

    This is simpler than TimeLoop for cases where you just need to track
    a fixed number of iterations without time tracking.

    Example:
        >>> with progress.iteration(total=100, desc="Optimization") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update()

    Args:
        total: Total number of iterations
        desc: Description for progress bar
        disable: If True, disable progress bar
        **kwargs: Additional arguments passed to tqdm
    """

    def __init__(
        self,
        total: int,
        desc: str = "Iterations",
        disable: bool = False,
        **kwargs
    ):
        """Initialize iteration loop progress tracker.

        Args:
            total: Total number of iterations
            desc: Description for progress bar
            disable: If True, disable progress bar
            **kwargs: Additional arguments passed to tqdm
        """
        super().__init__(desc=desc, disable=disable, **kwargs)

        self.total = total
        self.current = 0

        # Create progress bar
        self._create_pbar(total=total)

    def __enter__(self) -> "IterationLoop":
        """Enter context manager.

        Returns:
            Self for method chaining
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close progress bar."""
        self._close_pbar()

    def update(self, n: int = 1) -> None:
        """Update progress by n iterations.

        Args:
            n: Number of iterations to increment
        """
        self.current += n
        self._update_pbar(n)

    def set_postfix(self, **kwargs) -> None:
        """Update postfix statistics.

        Args:
            **kwargs: Key-value pairs to display
        """
        self._set_postfix(**kwargs)


@contextmanager
def mesh_adaptation(
    mesh: "samurai_python.Mesh",
    desc: str = "Mesh adaptation",
    disable: bool = False
):
    """Context manager for mesh adaptation operations.

    This provides progress tracking and statistics for mesh adaptation
    operations, showing cell count changes before and after adaptation.

    Example:
        >>> with progress.mesh_adaptation(mesh) as stats:
        ...     MRadaptation(config)
        ...     # Stats automatically updated on exit

    Args:
        mesh: Mesh being adapted
        desc: Description for progress
        disable: If True, disable progress output

    Yields:
        MeshStatistics object that is updated with adapted mesh
    """
    stats_before = MeshStatistics()
    stats_before.update(mesh)

    # Create simple progress indicator
    if not disable:
        print(f"{desc}... ", end="", flush=True)

    start_time = time.time()

    yield stats_before

    elapsed = time.time() - start_time

    # Compute statistics after adaptation
    stats_after = MeshStatistics()
    stats_after.update(mesh)

    if not disable:
        cell_change = stats_after.n_cells - stats_before.n_cells
        change_str = f"+{cell_change}" if cell_change > 0 else str(cell_change)
        print(
            f"done ({elapsed:.3f}s) | "
            f"{stats_before.n_cells} -> {stats_after.n_cells} cells ({change_str})"
        )
