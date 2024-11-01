# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
"""
Wrappers to Matplotlib plotting functions
"""
from __future__ import annotations

import logging
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typing_extensions as t

from .gamedata import (
    SECTOR_SIZE,
    SECTOR_GRID,
    SECTOR_MAX_BOOST,
    MAP_SIZE,
)
from . import util as u

if t.TYPE_CHECKING:
    import matplotlib.collections as mpl_collections
    import numpy.typing as npt

    AxesGrid: t.TypeAlias = t.Union[
        plt.Axes,
        t.List[t.List[plt.Axes]],
        t.Dict[str, plt.Axes],
    ]

HEATMAP_PALETTE = "viridis"


log = logging.getLogger(__name__)


def init_window(
    *args,
    backend: str = "",
    theme: bool = True,
    mosaic: bool = False,
    **kwargs,
) -> t.Tuple[plt.Figure, AxesGrid]:
    if backend:
        plt.switch_backend(backend)
    if theme:
        sns.set_theme()
    kwargs.setdefault("layout", "constrained")
    fig, ax = (plt.subplot_mosaic if mosaic else plt.subplots)(*args, **kwargs)
    return fig, ax


def init_plot_axes(ax: t.Optional[plt.Axes] = None, sector_scaling: bool = True) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    end = 0 if sector_scaling else 1  # int(not sector_scaling)
    xticks = np.arange(0, SECTOR_GRID[0] + end)
    yticks = np.arange(0, SECTOR_GRID[1] + end)
    labels = string.ascii_uppercase[: len(xticks)]
    if sector_scaling:
        ax.set_xlim(0, SECTOR_GRID[0])
        ax.set_ylim(0, SECTOR_GRID[1])
        ax.set_xticks(xticks, labels)
        ax.set_yticks(yticks)
    else:
        ax.set_xlim(0, MAP_SIZE[0])
        ax.set_ylim(0, MAP_SIZE[1])
        ax.set_xticks(xticks * SECTOR_SIZE[0])
        ax.set_yticks(yticks * SECTOR_SIZE[1])
    return ax


def show_window(title="Surviving Mars' Sensor Towers scan boost", block=True, aspect=1.15):
    # Enlarge window keeping a nice aspect. Works on GTK/Qt/Tk Agg backends (sorry, macOS!)
    # See https://stackoverflow.com/q/12439588/624066
    mng = plt.get_current_fig_manager()
    backend = plt.get_backend()
    log.debug("Matplotlib backend: %s", backend)
    if backend.startswith("GTK"):  # GTK3Agg, GTK4Agg
        rect = mng.window.get_screen().get_display().get_primary_monitor().get_workarea()
        size = rect.width, rect.height
    elif backend.startswith("Qt"):  # QtAgg, Qt5Agg, Qt6Agg
        qsize = mng.window.screen().availableSize()
        size = qsize.width(), qsize.height()
    elif backend.startswith("Tk"):  # TkAgg
        size = mng.window.maxsize()
    else:
        size = (1200, 680)  # Assume desktop of at least 720p (1280x720 minus panels)
    resize = (min(*size) - 40,) * 2
    resize = (u.clamp(int(resize[0] * aspect), size[0]), resize[1])  # Nice semi-square aspect
    log.debug("Detected usable desktop area: %s. Resizing to %s", size, resize)
    mng.resize(*resize)

    if title:
        mng.set_window_title(title)
        plt.suptitle(title, size="x-large", weight="bold")
    plt.show(block=block)


def draw_towers(
    towers: npt.ArrayLike,
    obj: t.Optional[mpl_collections.PathCollection] = None,
    ax: t.Optional[plt.Axes] = None,
    size=20,
    marker="*",
    color="red",
    label="Tower",
    val: t.Optional[float] = None,  # stats.avg
    val_fmt="6.2f",
    rescale=False,
    **kwargs,
):
    num = len(towers)
    if rescale and num:
        towers = np.divide(towers, SECTOR_SIZE)
        log.debug(f"Tower data for scatter plot:\n%s", towers)
    if obj is not None:
        obj.set_offsets(towers)
        return obj
    if ax is None:
        ax = plt.gca()
    return ax.scatter(
        *(zip(*towers) if num else [(), ()]),
        marker=marker,
        s=size**2,
        c=color,
        label=label if val is None else f"{val:{val_fmt}}: {label}",
        **kwargs,
    )


def draw_seaborn_heatmap(
    data,
    title: str = "",
    palette: str = HEATMAP_PALETTE,
    colorbar: bool = True,
    ax: plt.Axes = None,
) -> plt.Axes:
    # Good colormaps ("_r" means reversed):
    # turbo, rainbow, plasma, viridis, gnuplot2, YlOrBr_r, Spectral_r
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html

    # Sectors Heatmap
    ax_heatmap = sns.heatmap(
        data,
        cmap=palette,
        annot=True,
        fmt=".0f",
        vmin=0,
        vmax=SECTOR_MAX_BOOST,
        linewidths=1,
        square=True,
        xticklabels=string.ascii_uppercase[0 : SECTOR_GRID[0]],
        mask=(data == 0),
        ax=ax,
        cbar=colorbar,
    )
    if ax is None:
        ax = ax_heatmap

    def format_coord(sx, sy):
        x, y = np.multiply((sx, sy), SECTOR_SIZE)
        x_label = ax.get_xticklabels()[int(sx)].get_text()
        y_label = ax.get_yticklabels()[int(sy)].get_text()  # str(int(sy))
        z = data[int(sx), int(sy)]
        return f"Sector {x_label}{y_label} (X={x:3.0f}, Y={y:3.0f}) [Scan boost = {z}]"

    ax.format_coord = format_coord
    ax.invert_yaxis()
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_right()
    if title:
        ax.set_title(title)
    return ax
