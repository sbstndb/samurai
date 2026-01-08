"""
Mesh statistics tracking for progress reporting.

This module provides efficient tracking of mesh statistics for adaptive mesh
refinement simulations, including cell counts and refinement levels.
"""

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import samurai_python


class MeshStatistics:
    """Track and compute mesh statistics efficiently.

    This class caches mesh statistics to avoid recomputing them on every
    iteration, which is important for performance in adaptive mesh refinement
    simulations.

    Example:
        >>> stats = MeshStatistics()
        >>> stats.update(mesh)
        >>> print(f"Cells: {stats.n_cells}, Levels: [{stats.min_level}, {stats.max_level}]")
        Cells: 15234, Levels: [4, 10]

    Args        enable_level_breakdown: If True, track cell counts per level
    """

    def __init__(self, enable_level_breakdown: bool = False):
        """Initialize mesh statistics tracker.

        Args:
            enable_level_breakdown: If True, track cell counts per level
        """
        self._enable_level_breakdown = enable_level_breakdown
        self._n_cells: int = 0
        self._min_level: int = 0
        self._max_level: int = 0
        self._level_counts: Dict[int, int] = {}
        self._dirty: bool = True  # Cache is invalid

    @property
    def n_cells(self) -> int:
        """Total number of cells in the mesh."""
        return self._n_cells

    @property
    def min_level(self) -> int:
        """Minimum refinement level in the mesh."""
        return self._min_level

    @property
    def max_level(self) -> int:
        """Maximum refinement level in the mesh."""
        return self._max_level

    @property
    def level_counts(self) -> Dict[int, int]:
        """Dictionary mapping level to cell count at that level.

        Only available if enable_level_breakdown=True.
        """
        return self._level_counts.copy()

    def update(self, mesh: "samurai_python.Mesh") -> None:
        """Update statistics from current mesh state.

        This method efficiently computes mesh statistics by iterating over
        all cells and tracking their levels.

        Args:
            mesh: The mesh to analyze
        """
        # Reset counters
        self._n_cells = 0
        self._level_counts.clear() if self._enable_level_breakdown else None

        # Initialize level tracking
        min_level = float("inf")
        max_level = -float("inf")

        # Count cells by iterating over mesh
        def count_cell(cell):
            nonlocal min_level, max_level
            level = cell.level
            self._n_cells += 1
            min_level = min(min_level, level)
            max_level = max(max_level, level)

            if self._enable_level_breakdown:
                self._level_counts[level] = self._level_counts.get(level, 0) + 1

        import samurai_python
        samurai_python.algorithms.for_each_cell(mesh, count_cell)

        # Store results
        self._min_level = int(min_level) if self._n_cells > 0 else 0
        self._max_level = int(max_level) if self._n_cells > 0 else 0
        self._dirty = False

    def get_summary(self) -> str:
        """Get a formatted summary string of mesh statistics.

        Returns:
            Formatted string with key statistics

        Example:
            >>> stats.get_summary()
            '15234 cells [4-10]'
        """
        if self._dirty:
            return "Mesh stats not computed"
        return f"{self._n_cells} cells [{self._min_level}-{self._max_level}]"

    def get_level_breakdown(self) -> str:
        """Get a formatted breakdown of cells by refinement level.

        Returns:
            Formatted string showing cell count per level

        Example:
            >>> stats.get_level_breakdown()
            'L4: 1024, L5: 2048, L6: 4096, L7: 5120, L8: 2048, L9: 768, L10: 130'
        """
        if not self._enable_level_breakdown:
            return "Level breakdown not enabled"

        if self._dirty:
            return "Mesh stats not computed"

        parts = []
        for level in range(self._min_level, self._max_level + 1):
            count = self._level_counts.get(level, 0)
            if count > 0:
                parts.append(f"L{level}: {count}")

        return ", ".join(parts)

    def __repr__(self) -> str:
        """String representation of mesh statistics."""
        if self._dirty:
            return "MeshStatistics(not computed)"
        return f"MeshStatistics({self.get_summary()})"


def compute_mesh_stats(mesh: "samurai_python.Mesh") -> Dict[str, int]:
    """Convenience function to compute mesh statistics in one call.

    This is useful for one-off computations where you don't need the
    caching behavior of MeshStatistics.

    Args:
        mesh: The mesh to analyze

    Returns:
        Dictionary with keys: 'n_cells', 'min_level', 'max_level'

    Example:
        >>> stats = compute_mesh_stats(mesh)
        >>> print(f"Simulation has {stats['n_cells']} cells")
        Simulation has 15234 cells
    """
    stats = MeshStatistics()
    stats.update(mesh)
    return {
        "n_cells": stats.n_cells,
        "min_level": stats.min_level,
        "max_level": stats.max_level,
    }
