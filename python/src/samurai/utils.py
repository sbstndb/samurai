"""
Utility functions for Samurai Python demos.

This module provides helper functions for common tasks in Samurai demos,
including progress bars, mesh adaptation context managers, and visualization.
"""

import time
from contextlib import contextmanager
from typing import Optional


class ProgressBar:
    """Simple progress bar for time stepping loops.

    Displays progress information including iteration, time, and mesh statistics.

    Example:
        >>> with progress.time_loop(Tf=1.0, dt=0.01, desc="Simulation") as pbar:
        ...     while pbar.advance(dt=0.01, mesh=mesh):
        ...         # ... simulation code ...
    """

    def __init__(self, total_time: float, dt: float, desc: str = "Simulation"):
        """Initialize progress bar.

        Args:
            total_time: Final simulation time
            dt: Time step size
            desc: Description for progress bar
        """
        self.total_time = total_time
        self.dt = dt
        self.desc = desc
        self.current_time = 0.0
        self.iteration = 0
        self.start_time = None
        self.last_update_time = 0
        self.update_interval = 1.0  # Update display every 1 second

    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and display final summary."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n{self.desc} complete in {elapsed:.2f}s")
        return False

    @contextmanager
    def mesh_adaptation(self, mesh):
        """Context manager for mesh adaptation operations.

        This provides a clean way to handle mesh adaptation with proper
        tracking and potential optimization hooks.

        Args:
            mesh: The mesh object to adapt

        Example:
            >>> with pbar.mesh_adaptation(mesh):
            ...     MRadaptation(mra_config)
        """
        # Store mesh state before adaptation
        cells_before = getattr(mesh, 'nb_cells', 0)

        yield mesh

        # Could track adaptation statistics here in the future
        cells_after = getattr(mesh, 'nb_cells', 0)
        # For now, we just provide the context structure

    def advance(self, dt: Optional[float] = None, mesh=None) -> bool:
        """Advance progress bar by one time step.

        Args:
            dt: Time step size (uses self.dt if None)
            mesh: Optional mesh object for statistics

        Returns:
            True if simulation should continue, False if total_time reached

        Example:
            >>> while pbar.advance(dt=0.01, mesh=mesh):
            ...     # ... perform time step ...
        """
        if dt is None:
            dt = self.dt

        # Check if we would exceed total time
        if self.current_time + dt > self.total_time + 1e-14:  # Small tolerance for floating point
            self._display_progress(mesh, force=True)
            return False

        self.current_time += dt
        self.iteration += 1

        # Check if simulation is complete (after advancing)
        if self.current_time >= self.total_time - 1e-14:  # Small tolerance
            self._display_progress(mesh, force=True)
            return False

        # Update display periodically
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._display_progress(mesh)
            self.last_update_time = current_time

        return True

    def _display_progress(self, mesh=None, force: bool = False):
        """Display progress information.

        Args:
            mesh: Optional mesh object for statistics
            force: Force display even if interval hasn't elapsed
        """
        # Get mesh statistics if available
        mesh_info = ""
        if mesh:
            try:
                n_cells = getattr(mesh, 'nb_cells', 'N/A')
                mesh_info = f" | cells: {n_cells}"
            except Exception:
                pass

        # Calculate progress percentage (handle zero division)
        if self.total_time > 0:
            percent = min(100.0, 100.0 * self.current_time / self.total_time)
        else:
            percent = 100.0

        # Display progress
        print(f"\r{self.desc}: {self.iteration} it | t={self.current_time:.6f}/{self.total_time:.6f} ({percent:5.1f}%){mesh_info}",
              end="", flush=True)


class TimeLoopProgress:
    """Alternative progress bar interface using context manager for the loop itself.

    Example:
        >>> with progress.time_loop(Tf=1.0, dt=0.01, desc="Simulation") as pbar:
        ...     while True:
        ...         if not pbar.advance(dt=0.01, mesh=mesh):
        ...             break
        ...         # ... simulation code ...
    """

    def __init__(self, Tf: float, dt: float, desc: str = "Simulation"):
        """Initialize time loop progress.

        Args:
            Tf: Final simulation time
            dt: Time step size
            desc: Description for progress bar
        """
        self.pbar = ProgressBar(total_time=Tf, dt=dt, desc=desc)

    def __enter__(self):
        """Enter context manager."""
        self.pbar.__enter__()
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        return self.pbar.__exit__(exc_type, exc_val, exc_tb)


# Create a module-level convenience object
class _ProgressModule:
    """Progress module with convenience functions."""

    def time_loop(self, Tf: float, dt: float, desc: str = "Simulation") -> TimeLoopProgress:
        """Create a time loop progress bar.

        Args:
            Tf: Final simulation time
            dt: Time step size
            desc: Description for progress bar

        Returns:
            TimeLoopProgress context manager

        Example:
            >>> with progress.time_loop(Tf=1.0, dt=0.01, desc="Simulation") as pbar:
            ...     while True:
            ...         if not pbar.advance(dt=0.01, mesh=mesh):
            ...             break
            ...         # ... simulation code ...
        """
        return TimeLoopProgress(Tf=Tf, dt=dt, desc=desc)


# Module-level instance
progress = _ProgressModule()
