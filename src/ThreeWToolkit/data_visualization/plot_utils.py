from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def create_subplot_grid(
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] | None = None,
    default_width_per_col: int = 5,
    default_height_per_row: int = 4,
) -> tuple[Figure, np.ndarray]:
    """
    Create a grid of matplotlib subplots with consistent sizing and layout.

    Args:
        nrows: Number of rows in the subplot grid.
        ncols: Number of columns in the subplot grid.
        figsize: Optional figure size as (width, height). If None, a default
            size is computed based on the grid dimensions.
        default_width_per_col: Default width (in inches) for each column when
            figsize is not provided.
        default_height_per_row: Default height (in inches) for each row when
            figsize is not provided.

    Returns:
        A tuple containing:
            - fig: The created matplotlib Figure.
            - axes: A 2D NumPy array of Axes objects for consistent indexing.

    Raises:
        ValueError: If nrows or ncols are not positive integers.
    """
    if figsize is None:
        figsize = (default_width_per_col * ncols, default_height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    fig.tight_layout(pad=3.0)

    return fig, axes
