# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
"""
Wrappers to Matplotlib plotting functions
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import typing_extensions as t

from .gamedata import (
    SECTOR_SIZE,
)

if t.TYPE_CHECKING:
    import matplotlib.collections as mpl_collections
    import numpy.typing as npt

log = logging.getLogger(__name__)


def draw_towers(
    towers: npt.ArrayLike,
    obj: mpl_collections.PathCollection | None = None,
    ax: plt.Axes | None = None,
    size=20,
    marker="*",
    color="red",
    label="Tower",
    val: float | None = None,  # stats.avg
    val_fmt="6.2f",
    rescale=False,
    **kwargs,
):
    if rescale and len(towers):
        towers = np.divide(towers, SECTOR_SIZE)
        log.debug(f"Tower data for scatter plot:\n%s", towers)
    if obj is not None:
        obj.set_offsets(towers)
        return obj
    if ax is None:
        ax = plt.gca()
    return ax.scatter(
        *zip(*towers),
        marker=marker,
        s=size**2,
        c=color,
        label=label if val is None else f"{val:{val_fmt}}: {label}",
        **kwargs,
    )
