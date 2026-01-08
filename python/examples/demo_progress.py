#!/usr/bin/env python3
"""
Demo: Progress Bar API for Samurai Python
==========================================

This script demonstrates the progress bar API features without running
a full simulation. It shows different types of progress tracking:

1. **time_loop**: Track time stepping progress
2. **mesh_adaptation**: Track mesh adaptation progress
3. **iteration**: Track custom iteration progress
4. **custom metrics**: Display live statistics

Usage:
    python demo_progress.py

This demo requires:
    - samurai_python module (built from source)
    - tqdm (pip install tqdm)
    - No actual simulation is run, just demonstrations
"""

import os
import sys
import time

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)



# ============================================================================
# PROGRESS BAR IMPLEMENTATION
# ============================================================================

class ProgressBar:
    """
    Simple progress bar implementation for Samurai simulations.

    This class provides a lightweight progress tracking mechanism that
    works with standard output without requiring external dependencies.

    Features:
        - Progress percentage display
        - Elapsed time tracking
        - Custom metric display (cells, min/max level, etc.)
        - Minimal overhead

    Example:
        >>> progress = ProgressBar(total=100, desc="Time stepping")
        >>> for i in range(100):
        ...     # Do work
        ...     progress.update(1, metrics={"cells": 1000, "dt": 0.01})
        >>> progress.close()
    """

    def __init__(self, total, desc="Progress", metrics=None):
        """Initialize the progress bar.

        Args:
            total: Total number of iterations/steps
            desc: Description text to display
            metrics: Dictionary of initial metric values
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.metrics = metrics or {}
        self.start_time = time.time()
        self.last_update = time.time()
        self._closed = False

    def update(self, n=1, metrics=None):
        """Update progress by n steps.

        Args:
            n: Number of steps to advance (default: 1)
            metrics: Dictionary of updated metric values
        """
        if self._closed:
            return

        self.current += n
        if metrics:
            self.metrics.update(metrics)

        # Update display every 0.1 seconds to avoid flickering
        now = time.time()
        if now - self.last_update > 0.1 or self.current >= self.total:
            self._display()
            self.last_update = now

    def set_description(self, desc):
        """Update the description text.

        Args:
            desc: New description text
        """
        self.desc = desc
        self._display()

    def _display(self):
        """Display the progress bar."""
        percent = min(100, self.current * 100 / self.total)
        elapsed = time.time() - self.start_time

        # Build metrics string
        metrics_str = ""
        if self.metrics:
            metrics_items = [f"{k}={v}" for k, v in self.metrics.items()]
            metrics_str = " | " + ", ".join(metrics_items)

        # Build progress bar
        bar_width = 40
        filled = int(bar_width * self.current / self.total)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print with carriage return to overwrite
        sys.stdout.write(
            f"\r{self.desc}: [{bar}] {percent:5.1f}% "
            f"({self.current}/{self.total}) "
            f"elapsed: {elapsed:.1f}s{metrics_str}"
        )
        sys.stdout.flush()

    def close(self):
        """Close the progress bar and print final newline."""
        if not self._closed:
            self._closed = True
            elapsed = time.time() - self.start_time
            print()  # Final newline
            print(f"Complete! Total time: {elapsed:.2f}s")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TQDMProgressBar:
    """
    Wrapper around tqdm for richer progress bar functionality.

    This class provides a more feature-rich progress bar using tqdm if available.
    Falls back to basic ProgressBar if tqdm is not installed.

    Features:
        - Progress bar with ETA
        - Custom metrics via postfix
        - Rate (iterations per second)
        - Rich formatting options

    Example:
        >>> with TQDMProgressBar(total=100, desc="Time stepping") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update(1, cells=1000, dt=0.01)
    """

    def __init__(self, total, desc="Progress"):
        """Initialize the tqdm progress bar.

        Args:
            total: Total number of iterations/steps
            desc: Description text to display
        """
        self.total = total
        self.desc = desc
        self._use_tqdm = self._check_tqdm()
        self._pbar = None

        if self._use_tqdm:
            try:
                from tqdm import tqdm
                self._pbar = tqdm(total=total, desc=desc, unit="iter")
            except ImportError:
                self._use_tqdm = False
                self._pbar = ProgressBar(total=total, desc=desc)

        if not self._use_tqdm:
            self._pbar = ProgressBar(total=total, desc=desc)

    def _check_tqdm(self):
        """Check if tqdm is available."""
        try:
            import tqdm
            return True
        except ImportError:
            return False

    def update(self, n=1, **metrics):
        """Update progress by n steps.

        Args:
            n: Number of steps to advance (default: 1)
            **metrics: Keyword arguments of metric values
        """
        if self._use_tqdm:
            self._pbar.update(n)
            if metrics:
                self._pbar.set_postfix(metrics)
        else:
            self._pbar.update(n, metrics=metrics)

    def set_description(self, desc):
        """Update the description text.

        Args:
            desc: New description text
        """
        if self._use_tqdm:
            self._pbar.set_description(desc)
        else:
            self._pbar.set_description(desc)

    def close(self):
        """Close the progress bar."""
        if self._pbar:
            self._pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_basic_progress():
    """Demonstration 1: Basic progress bar usage."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Progress Bar")
    print("=" * 70)

    total_steps = 50

    with ProgressBar(total=total_steps, desc="Processing") as pbar:
        for i in range(total_steps):
            # Simulate work
            time.sleep(0.05)

            # Update progress with custom metrics
            pbar.update(1, metrics={"step": i + 1, "value": i * 2})

    print()


def demo_time_loop():
    """Demonstration 2: Time stepping progress."""
    print("\n" + "=" * 70)
    print("DEMO 2: Time Loop Progress (Simulation Style)")
    print("=" * 70)

    # Simulation parameters
    Tf = 1.0  # Final time
    dt = 0.02  # Time step
    nt = int(Tf / dt)  # Number of time steps

    print(f"Simulating from t=0 to t={Tf} with dt={dt}")
    print(f"Total time steps: {nt}\n")

    with ProgressBar(total=nt, desc="Time stepping") as pbar:
        t = 0.0
        for it in range(nt):
            # Simulate physics computation
            time.sleep(0.03)

            t += dt

            # Update progress every 10 steps
            if (it + 1) % 10 == 0:
                pbar.update(10, metrics={
                    "time": f"{t:.3f}",
                    "iter": it + 1,
                    "cells": 1250
                })
            elif it + 1 == nt:
                # Final update
                pbar.update(1, metrics={
                    "time": f"{t:.3f}",
                    "iter": it + 1,
                    "cells": 1250
                })

    print()


def demo_mesh_adaptation():
    """Demonstration 3: Mesh adaptation progress."""
    print("\n" + "=" * 70)
    print("DEMO 3: Mesh Adaptation Progress")
    print("=" * 70)

    # Simulate mesh adaptation cycle
    n_adaptations = 10

    print("Performing mesh adaptation cycles\n")

    with ProgressBar(total=n_adaptations, desc="Mesh adaptation") as pbar:
        for i in range(n_adaptations):
            # Simulate mesh adaptation work
            time.sleep(0.1)

            # Simulate changing cell counts and levels
            cells = 1000 + i * 100
            min_level = 4 + (i // 3)
            max_level = 8 + (i // 5)

            pbar.update(1, metrics={
                "cells": cells,
                "min_level": min_level,
                "max_level": max_level
            })

    print()


def demo_custom_iteration():
    """Demonstration 4: Custom iteration progress."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Iteration (Nonlinear Solver)")
    print("=" * 70)

    # Simulate nonlinear solver iterations
    max_iter = 30
    tolerance = 1e-6

    print(f"Solving nonlinear system (max {max_iter} iterations)\n")

    residual = 1.0
    with ProgressBar(total=max_iter, desc="Nonlinear solve") as pbar:
        for it in range(max_iter):
            # Simulate solver iteration
            time.sleep(0.05)

            # Simulate residual reduction
            residual *= 0.7

            pbar.update(1, metrics={
                "iter": it + 1,
                "residual": f"{residual:.2e}"
            })

            # Check convergence (optional early exit)
            if residual < tolerance:
                print(f"\n  Converged after {it + 1} iterations!")
                break

    print()


def demo_nested_progress():
    """Demonstration 5: Nested progress tracking."""
    print("\n" + "=" * 70)
    print("DEMO 5: Nested Progress (Time steps + Inner iterations)")
    print("=" * 70)

    n_timesteps = 5
    n_inner_iter = 20

    print(f"Running {n_timesteps} time steps with {n_inner_iter} inner iterations\n")

    outer_desc = "Time steps"
    for ts in range(n_timesteps):
        print(f"\nTime step {ts + 1}/{n_timesteps}:")

        # Inner loop
        with ProgressBar(total=n_inner_iter, desc="  Inner iterations") as pbar:
            for it in range(n_inner_iter):
                time.sleep(0.02)
                pbar.update(1, metrics={"inner": it + 1})

        print(f"  Time step {ts + 1} complete")

    print()


def demo_tqdm_fallback():
    """Demonstration 6: TQDM with fallback."""
    print("\n" + "=" * 70)
    print("DEMO 6: TQDM Progress Bar (with automatic fallback)")
    print("=" * 70)

    total_steps = 30

    print("Using TQDM-style progress bar (falls back to basic if tqdm unavailable)\n")

    with TQDMProgressBar(total=total_steps, desc="TQDM test") as pbar:
        for i in range(total_steps):
            time.sleep(0.03)
            pbar.update(1, cells=1000 + i * 10, dt=0.01)

    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all progress bar demonstrations."""
    print("\n" + "=" * 70)
    print(" SAMURAI PROGRESS BAR DEMO")
    print("=" * 70)
    print("\nThis demo shows various ways to use progress bars in Samurai.")
    print("No actual simulation is run - these are just examples.\n")

    # Run all demonstrations
    demo_basic_progress()
    demo_time_loop()
    demo_mesh_adaptation()
    demo_custom_iteration()
    demo_nested_progress()
    demo_tqdm_fallback()

    print("\n" + "=" * 70)
    print(" ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nTo use progress bars in your own code:")
    print("  1. Import the ProgressBar class from this demo")
    print("  2. Create a progress bar with ProgressBar(total=..., desc=...)")
    print("  3. Update progress in your loops with pbar.update(n, metrics=...)")
    print("  4. Use context manager: 'with ProgressBar(...) as pbar:'")
    print("\nExample:")
    print("""
    from demo_progress import ProgressBar

    with ProgressBar(total=nt, desc="Time stepping") as pbar:
        for it in range(nt):
            # Your physics here
            u = compute_next_step(u)

            # Update progress
            pbar.update(1, metrics={"time": t, "cells": mesh.nb_cells})
    """)


if __name__ == "__main__":
    main()
