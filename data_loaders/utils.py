"""
Shared utilities for data loaders.
"""

from typing import List, Union

import numpy as np


def format_values_as_data_str(
    values: Union[List[float], List[List[float]], np.ndarray], decimal_places: int = 2
) -> str:
    """
    Format time series values as a well-formatted data_str string.

    Format:
    - Univariate: comma-separated values with specified decimal places
      Example: "1.234,2.345,3.456"
    - Multivariate: newline-separated dimensions, each dimension comma-separated
      Example: "dim_0: 1.234,2.345\ndim_1: 3.456,4.567"

    Args:
        values: Time series values
            - Univariate: List[float] or 1D array
            - Multivariate: List[List[float]] or 2D array
        decimal_places: Number of decimal places for formatting (default: 2)

    Returns:
        Formatted string representation of the time series data
    """
    # Handle empty values
    if values is None:
        return ""

    # Handle numpy arrays first (before checking length)
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return ""
        if values.ndim > 1:
            # Multivariate: format each dimension separately
            formatted_dims = []
            for i, dim in enumerate(values):
                dim_str = ",".join(f"{x:.{decimal_places}f}" for x in dim)
                formatted_dims.append(f"dim_{i}: {dim_str}")
            return "\n".join(formatted_dims)
        # Univariate: flatten to 1D
        values_flat = values.flatten()
        return ",".join(f"{x:.{decimal_places}f}" for x in values_flat)

    # Handle lists
    if isinstance(values, list):
        if len(values) == 0:
            return ""
        # Check if multivariate (list of lists)
        if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
            # Multivariate: format each dimension separately
            formatted_dims = []
            for i, dim in enumerate(values):
                # Convert dimension to list if needed
                if isinstance(dim, np.ndarray):
                    dim = dim.tolist()
                dim_str = ",".join(f"{x:.{decimal_places}f}" for x in dim)
                formatted_dims.append(f"dim_{i}: {dim_str}")
            return "\n".join(formatted_dims)
        # Univariate: single list
        return ",".join(f"{x:.{decimal_places}f}" for x in values)

    # Fallback: convert to string
    return str(values)
