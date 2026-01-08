"""
Progress bar utilities for Samurai simulations.

This module provides a simple, intuitive API for tracking progress in
time-dependent adaptive mesh refinement simulations.

Example usage:
    >>> import samurai_python as sam
    >>> from samurai_python.utils import progress
    >>>
    >>> # Time stepping loop with mesh statistics
    >>> with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    ...     while pbar.continue_loop():
    ...         # Simulation step
    ...         MRadaptation(config)
    ...         update_ghost_mr(u)
    ...         # Update time
    ...         pbar.advance_time(dt)
    ...         # Update progress bar with mesh stats
    ...         pbar.update_stats(mesh=u.mesh)
    >>>
    >>> # Mesh adaptation context manager
    >>> with progress.mesh_adaptation(mesh) as stats:
    ...     MRadaptation(config)
    >>>
    >>> # Simple iteration loop
    >>> with progress.iteration(total=100) as pbar:
    ...     for i in range(100):
    ...         # Do work
    ...         pbar.update()
"""

from .core import IterationLoop, TimeLoop, mesh_adaptation
from .stats import MeshStatistics, compute_mesh_stats

# Public API
__all__ = [
    "IterationLoop",
    "MeshStatistics",
    "TimeLoop",
    "compute_mesh_stats",
    "iteration",
    "mesh_adaptation",
    "time_loop",
]


def time_loop(
    Tf: float,
    dt: float,
    desc: str = "Time loop",
    disable: bool = False,
    **kwargs
) -> TimeLoop:
    """Create a time-stepping progress bar context manager.

    This is the recommended way to track progress in time-dependent simulations.
    It handles variable time steps, ETA estimation, and mesh statistics.

    Args:
        Tf: Final simulation time
        dt: Initial time step (for ETA estimation)
        desc: Description for progress bar (default: "Time loop")
        disable: If True, disable progress bar display
        **kwargs: Additional arguments passed to tqdm

    Returns:
        TimeLoop context manager

    Example:
        >>> with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
        ...     while pbar.continue_loop():
        ...         # Simulation step
        ...         MRadaptation(config)
        ...         update_ghost_mr(u)
        ...         pbar.advance_time(dt)
        ...         pbar.update_stats(mesh=u.mesh, residual=1e-6)

    For simulations with variable time steps:
        >>> with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
        ...     while pbar.continue_loop():
        ...         dt = compute_adaptive_dt()
        ...         pbar.advance_time(dt)

    Advanced usage with custom statistics:
        >>> with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
        ...     while pbar.continue_loop():
        ...         # Simulation step
        ...         pbar.advance_time(dt)
        ...         pbar.update_stats(
        ...             mesh=u.mesh,
        ...             max_val=float(max(u)),
        ...             min_val=float(min(u))
        ...         )
    """
    return TimeLoop(Tf=Tf, dt=dt, desc=desc, disable=disable, **kwargs)


def iteration(
    total: int,
    desc: str = "Iterations",
    disable: bool = False,
    **kwargs
) -> IterationLoop:
    """Create a fixed-count iteration progress bar.

    Use this for simple iteration loops without time tracking.

    Args:
        total: Total number of iterations
        desc: Description for progress bar (default: "Iterations")
        disable: If True, disable progress bar display
        **kwargs: Additional arguments passed to tqdm

    Returns:
        IterationLoop context manager

    Example:
        >>> with progress.iteration(total=100, desc="Optimization") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update()
        ...         pbar.set_postfix(loss=compute_loss())

    Using with Python's enumerate:
        >>> with progress.iteration(total=len(data)) as pbar:
        ...     for i, item in enumerate(data):
        ...         process(item)
        ...         pbar.update()
    """
    return IterationLoop(total=total, desc=desc, disable=disable, **kwargs)
