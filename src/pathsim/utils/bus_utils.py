import numpy as np


def reconstruct_bus_dict(flat_array, flat_layout):
    """Reconstruct a nested bus dict from a flat numpy array.

    Use this after simulation to convert a recorded flat bus array back into
    a structured dictionary matching the original bus definition.

    Parameters
    ----------
    flat_array : array-like
        1-D array of bus values as produced by a prepared BusCreator.
    flat_layout : dict[str, int]
        Dotted-key → flat-index mapping from ``BusCreator.flat_layout``.

    Returns
    -------
    dict
        Nested dict matching the bus hierarchy, e.g.
        ``{'Speed': 55.0, 'Sensors': {'Temperature': 22.0, 'Pressure': 101.0}}``.

    Example
    -------
    .. code-block:: python

        t, y = scope.read()          # y shape: (n_ports, n_steps)
        # reconstruct bus at each timestep
        buses = [reconstruct_bus_dict(y[:, i], creator.flat_layout)
                 for i in range(y.shape[1])]
    """
    result = {}
    for dotted_key, idx in flat_layout.items():
        parts = dotted_key.split('.')
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = float(flat_array[idx])
    return result


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------

def flatten_bus(bus_dict, bus_keys):
    """Flatten a bus dict to a numpy array using a fixed key order."""
    return np.array([bus_dict[k] for k in bus_keys], dtype=object)


def reconstruct_bus(bus_array, bus_keys):
    """Reconstruct a bus dict from a numpy array and key order."""
    return {k: v for k, v in zip(bus_keys, bus_array)}
