"""
Matplotlib-based visualization for Samurai AMR fields.

This module provides tools for visualizing adaptive mesh refinement (AMR)
data from Samurai using matplotlib. It handles the multiresolution nature
of Samurai meshes, properly rendering cells at different refinement levels.

Classes:
    FieldPlotter: Real-time scalar field visualization with efficient updates
    VectorPlotter: Real-time vector field visualization

Functions:
    plot_field: One-shot plotting of scalar fields
    plot_vector: One-shot plotting of vector fields (quiver)
    plot_mesh: Plot mesh structure (cell outlines by level)
    set_axes_equal: Helper to set equal aspect ratio

Example:
    >>> import matplotlib.pyplot as plt
    >>> from samurai.viz import matplotlib as svmpl
    >>>
    >>> # Static plot
    >>> fig, ax = plt.subplots()
    >>> svmpl.plot_field(u, ax=ax, cmap='viridis')
    >>> plt.show()
    >>>
    >>> # Real-time monitoring
    >>> plotter = svmpl.FieldPlotter(u, ax=ax)
    >>> for i in range(100):
    >>>     # ... update simulation ...
    >>>     plotter.update(u)  # Efficient update
    >>>     plt.pause(0.01)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, QuadMesh
from matplotlib.colors import Normalize
from typing import Optional, Tuple, List, Union


def _extract_cell_data(field) -> Tuple[List, List, List, List]:
    """Extract cell geometry and data from a Samurai field.

    Args:
        field: Samurai ScalarField or VectorField

    Returns:
        Tuple of (x_corners, y_corners, values, levels)
        - x_corners: List of x-coordinates of cell corners
        - y_corners: List of y-coordinates of cell corners
        - values: List of field values at cell centers
        - levels: List of refinement levels for each cell
    """
    import samurai_python as sam

    x_corners = []
    y_corners = []
    values = []
    levels = []

    def collect_cell(cell):
        cx, cy = cell.corner()
        L = cell.length
        val = field[cell.index]

        x_corners.append(cx)
        y_corners.append(cy)
        values.append(val)
        levels.append(cell.level)

    sam.for_each_cell(field.mesh, collect_cell)

    return x_corners, y_corners, values, levels


def _extract_vector_data(vector_field) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """Extract vector field data for quiver plotting.

    Args:
        vector_field: Samurai VectorField with 2 or 3 components

    Returns:
        Tuple of (x_centers, y_centers, u_comp, v_comp, levels)
    """
    import samurai_python as sam

    x_centers = []
    y_centers = []
    u_comp = []
    v_comp = []
    levels = []

    def collect_cell(cell):
        cx, cy = cell.center()
        val = vector_field[cell.index]

        x_centers.append(cx)
        y_centers.append(cy)
        u_comp.append(val[0])
        v_comp.append(val[1])
        levels.append(cell.level)

    sam.for_each_cell(vector_field.mesh, collect_cell)

    return (np.array(x_centers), np.array(y_centers),
            np.array(u_comp), np.array(v_comp), levels)


def _create_patches_from_cells(x_corners, y_corners, levels, cell_length_func) -> List[Rectangle]:
    """Create matplotlib Rectangle patches from cell corner data.

    Args:
        x_corners: List of x-coordinates of cell corners
        y_corners: List of y-coordinates of cell corners
        levels: List of refinement levels
        cell_length_func: Function to get cell size from level

    Returns:
        List of Rectangle patches
    """
    patches = []
    for x, y, level in zip(x_corners, y_corners, levels):
        L = cell_length_func(level)
        patches.append(Rectangle((x, y), L, L))
    return patches


def plot_field(
    field,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_mesh: bool = False,
    show_level: bool = False,
    colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> PatchCollection:
    """Plot a scalar field on an AMR mesh.

    Creates a pcolormesh-like visualization of a scalar field on an
    adaptive mesh refinement grid. Cells at different refinement levels
    are rendered correctly with proper sizing.

    Args:
        field: Samurai ScalarField to plot
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap name (default: 'viridis')
        vmin: Minimum value for color scaling (auto if None)
        vmax: Maximum value for color scaling (auto if None)
        show_mesh: If True, overlay cell outlines
        show_level: If True, color by refinement level instead of field value
        colorbar: If True, add colorbar
        title: Optional plot title
        **kwargs: Additional arguments passed to PatchCollection

    Returns:
        PatchCollection: The collection of cell patches

    Example:
        >>> fig, ax = plt.subplots()
        >>> svmpl.plot_field(u, ax=ax, cmap='RdBu_r', show_mesh=True)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Extract cell data
    x_corners, y_corners, values, levels = _extract_cell_data(field)

    # Determine what to color by
    if show_level:
        color_values = np.array(levels)
        cmap = kwargs.pop('level_cmap', 'tab10')
        vmin = vmin if vmin is not None else min(levels)
        vmax = vmax if vmax is not None else max(levels)
    else:
        color_values = np.array(values)
        if vmin is None:
            vmin = np.min(color_values)
        if vmax is None:
            vmax = np.max(color_values)

    # Create cell size function
    def cell_size(level):
        return field.mesh.cell_length(level) if hasattr(field.mesh, 'cell_length') else 2.0 ** (-level)

    # Create patches
    patches = _create_patches_from_cells(x_corners, y_corners, levels, cell_size)

    # Create collection
    collection = PatchCollection(patches, cmap=cmap, **kwargs)
    collection.set_array(color_values)
    collection.set_clim(vmin, vmax)

    # Add to axes
    ax.add_collection(collection)

    # Auto-scale axes
    mesh = field.mesh
    ax.set_xlim(left=0)  # Will be updated by autoscale
    ax.autoscale_view()

    # Overlay mesh if requested
    if show_mesh:
        plot_mesh(field, ax=ax, edgecolors='black', linewidths=0.5, alpha=0.3)

    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(collection, ax=ax)
        cbar.set_label('Level' if show_level else 'Value')

    # Set title
    if title:
        ax.set_title(title)

    return collection


def plot_vector(
    vector_field,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    scale: Optional[float] = None,
    scale_units: str = "xy",
    angles: str = "xy",
    show_mesh: bool = False,
    colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> plt.quiver:
    """Plot a vector field on an AMR mesh using quiver.

    Visualizes a vector field (e.g., velocity) using arrows on the
    adaptive mesh refinement grid.

    Args:
        vector_field: Samurai VectorField with 2 components
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap for arrow colors (based on magnitude)
        scale: Arrow scaling factor (auto if None)
        scale_units: Units for scaling (default: 'xy')
        angles: Arrow angle calculation (default: 'xy')
        show_mesh: If True, overlay cell outlines
        colorbar: If True, add colorbar for magnitude
        title: Optional plot title
        **kwargs: Additional arguments passed to quiver

    Returns:
        matplotlib.quiver: The quiver object

    Example:
        >>> fig, ax = plt.subplots()
        >>> svmpl.plot_vector(velocity, ax=ax, scale=20)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Extract vector data
    x, y, u, v, levels = _extract_vector_data(vector_field)

    # Compute magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)

    # Create quiver plot
    Q = ax.quiver(x, y, u, v, magnitude,
                  cmap=cmap, scale=scale, scale_units=scale_units,
                  angles=angles, **kwargs)

    # Auto-scale axes
    ax.autoscale_view()

    # Overlay mesh if requested
    if show_mesh:
        plot_mesh(vector_field, ax=ax, edgecolors='black', linewidths=0.5, alpha=0.3)

    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(Q, ax=ax)
        cbar.set_label('Magnitude')

    # Set title
    if title:
        ax.set_title(title)

    return Q


def plot_mesh(
    field,
    ax: Optional[plt.Axes] = None,
    edgecolors: str = "black",
    facecolors: str = "none",
    linewidths: float = 0.5,
    alpha: float = 0.5,
    by_level: bool = False,
    **kwargs
) -> PatchCollection:
    """Plot the mesh structure (cell outlines).

    Visualizes the adaptive mesh refinement structure by drawing
    cell boundaries. Can color cells by refinement level.

    Args:
        field: Samurai field (mesh is extracted from field.mesh)
        ax: Matplotlib axes (creates new figure if None)
        edgecolors: Color of cell edges
        facecolors: Color of cell faces ('none' for transparent)
        linewidths: Width of cell edges
        alpha: Transparency (0=transparent, 1=opaque)
        by_level: If True, color cells by refinement level
        **kwargs: Additional arguments passed to PatchCollection

    Returns:
        PatchCollection: The mesh collection

    Example:
        >>> fig, ax = plt.subplots()
        >>> svmpl.plot_mesh(u, ax=ax, by_level=True)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Extract cell data
    x_corners, y_corners, _, levels = _extract_cell_data(field)

    # Create cell size function
    def cell_size(level):
        return field.mesh.cell_length(level) if hasattr(field.mesh, 'cell_length') else 2.0 ** (-level)

    # Create patches
    patches = _create_patches_from_cells(x_corners, y_corners, levels, cell_size)

    # Determine colors
    if by_level:
        edgecolors = 'black'
        cmap = kwargs.pop('cmap', 'tab10')
        # Don't set facecolors initially, will be set by set_array()
        collection = PatchCollection(patches,
                                      edgecolors=edgecolors,
                                      linewidths=linewidths,
                                      cmap=cmap,
                                      **kwargs)
        collection.set_array(np.array(levels))
        collection.set_clim(min(levels), max(levels))
    else:
        collection = PatchCollection(patches,
                                      edgecolors=edgecolors,
                                      facecolors=facecolors,
                                      linewidths=linewidths,
                                      **kwargs)

    ax.add_collection(collection)
    ax.autoscale_view()

    return collection


def set_axes_equal(ax: plt.Axes):
    """Set equal aspect ratio for 2D plot.

    Ensures that one unit in x is the same length as one unit in y.

    Args:
        ax: Matplotlib axes to modify

    Example:
        >>> fig, ax = plt.subplots()
        >>> svmpl.plot_field(u, ax=ax)
        >>> svmpl.set_axes_equal(ax)
        >>> plt.show()
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    if x_range > y_range:
        pad = (x_range - y_range) / 2
        ax.set_ylim(ylim[0] - pad, ylim[1] + pad)
    else:
        pad = (y_range - x_range) / 2
        ax.set_xlim(xlim[0] - pad, xlim[1] + pad)

    ax.set_aspect('equal')


class FieldPlotter:
    """Real-time scalar field plotter with efficient updates.

    Designed for monitoring simulations in-progress. The plot is updated
    efficiently by modifying the existing collection rather than recreating
    the entire figure.

    Args:
        field: Initial Samurai ScalarField to plot
        ax: Matplotlib axes (creates new if None)
        cmap: Colormap name
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        show_mesh: If True, show cell outlines
        title: Optional plot title

    Example:
        >>> plotter = svmpl.FieldPlotter(u, cmap='RdBu_r')
        >>> for i in range(100):
        >>>     # ... update simulation ...
        >>>     plotter.update(u)
        >>>     plt.pause(0.01)
    """

    def __init__(
        self,
        field,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_mesh: bool = False,
        title: Optional[str] = None
    ):
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = ax.figure
            self.ax = ax

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.show_mesh = show_mesh
        self.title = title

        # Initial plot
        self.collection = plot_field(field, ax=self.ax, cmap=cmap,
                                     vmin=vmin, vmax=vmax,
                                     show_mesh=show_mesh, title=title)
        if title:
            self.ax.set_title(title)

        plt.tight_layout()

    def update(self, field, title: Optional[str] = None):
        """Update the plot with new field data.

        Efficiently updates the visualization without recreating the
        entire figure. The cell structure may change between updates
        (due to mesh adaptation).

        Args:
            field: New Samurai ScalarField to plot
            title: Optional new title
        """
        # Extract new data
        _, _, values, levels = _extract_cell_data(field)

        # Update color values
        self.collection.set_array(np.array(values))

        # Update limits if needed
        if self.vmin is None:
            self.collection.set_clim(vmin=np.min(values))
        if self.vmax is None:
            self.collection.set_clim(vmax=np.max(values))

        # Update title
        if title:
            self.ax.set_title(title)
        elif self.title:
            self.ax.set_title(self.title)

        # Redraw
        self.fig.canvas.draw_idle()

    def pause(self, interval: float = 0.01):
        """Pause for interactive plotting.

        Args:
            interval: Pause time in seconds
        """
        plt.pause(interval)


class VectorPlotter:
    """Real-time vector field plotter with efficient updates.

    Designed for monitoring vector fields (e.g., velocity) during
    simulations.

    Args:
        vector_field: Initial Samurai VectorField to plot
        ax: Matplotlib axes (creates new if None)
        cmap: Colormap for arrow colors
        scale: Arrow scaling factor
        show_mesh: If True, show cell outlines
        title: Optional plot title

    Example:
        >>> plotter = svmpl.VectorPlotter(velocity)
        >>> for i in range(100):
        >>>     # ... update simulation ...
        >>>     plotter.update(velocity)
        >>>     plt.pause(0.01)
    """

    def __init__(
        self,
        vector_field,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        scale: Optional[float] = None,
        show_mesh: bool = False,
        title: Optional[str] = None
    ):
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = ax.figure
            self.ax = ax

        self.cmap = cmap
        self.scale = scale
        self.show_mesh = show_mesh
        self.title = title

        # Initial plot
        self.quiver = plot_vector(vector_field, ax=self.ax, cmap=cmap,
                                  scale=scale, show_mesh=show_mesh, title=title)

        plt.tight_layout()

    def update(self, vector_field, title: Optional[str] = None):
        """Update the plot with new vector field data.

        Args:
            vector_field: New Samurai VectorField to plot
            title: Optional new title
        """
        # Extract new data
        x, y, u, v, _ = _extract_vector_data(vector_field)

        # Update quiver data
        self.quiver.set_UVC(u, v)

        # Update title
        if title:
            self.ax.set_title(title)
        elif self.title:
            self.ax.set_title(self.title)

        # Redraw
        self.fig.canvas.draw_idle()

    def pause(self, interval: float = 0.01):
        """Pause for interactive plotting.

        Args:
            interval: Pause time in seconds
        """
        plt.pause(interval)
