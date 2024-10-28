# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>

"""
Surviving Mars Sensor Towers scan boost heatmap using Matplotlib
"""
import dataclasses
import logging
import math
import string
import sys
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import util as u

__version__ = "2024.10"

# Constants from game data
# https://github.com/surviving-mars/SurvivingMars/blob/master/Lua/_GameConst.lua#L102
SECTOR_GRID = (10, 10)
SECTOR_SIZE = (40, 40)
BOOST_MAX = 390  # max boost provided by the nearest tower to the sector
MIN_RANGE = 20  # range of max boost (boost = max)
MAX_RANGE = 120  # range of no boost (boost = 0%)
NUM_BOOST = 10  # every working tower provides this much cumulative boost to all sectors
NUM_BOOST_MAX = 100  # max cumulative scan boost

# Derived constants
MAP_SIZE = np.multiply(SECTOR_GRID, SECTOR_SIZE)  # (400, 400)
SECTOR_MAX_BOOST = BOOST_MAX + NUM_BOOST_MAX  # 490
COST_REFERENCE = 44.12  # Average Sector boost for 1 Tower on map center

# Parameters default values
TOWERS_GRID_LAYOUT = "margin"
TOWERS_GRID_SIDE = 3
TOWERS_GRID_MARGIN = (2, 2)
HEATMAP_COLORS = "viridis"

log = logging.getLogger(__name__)
tower_generator = u.FunctionCollectionDecorator(remove_suffix="_grid")


@dataclasses.dataclass
class Statistics:
    num: int
    avg: float
    cost: float
    title: str

    def __init__(self, data, towers, global_boost: bool = True):
        self.num, self.avg, self.cost = self.statistics(data, towers, global_boost)
        self.title = (
            f"{self.num} Towers, "
            f"{self.avg:6.2f} mean boost, "
            f"normalized cost {self.cost:.1%}"
        )

    @staticmethod
    def statistics(data, towers, global_boost: bool = True):
        num, avg = len(towers), np.average(data)
        if num:
            avg_tower = avg / num
            cost = (COST_REFERENCE + (NUM_BOOST if global_boost else 0)) / avg_tower
        else:
            avg_tower = cost = 0
        log.info(
            "Statistics:\n"
            f"Towers: {num}\n"
            f"Average Boost per Sector: {avg:6.2f}\n"
            f"Sector average per Tower: {avg_tower:6.2f}\n"
            f"Normalized cost: {cost:.1%}"
        )
        return num, avg, cost

    def __str__(self):
        return self.title


@dataclasses.dataclass
class MapSector:
    BOOST_RANGE: t.ClassVar = MAX_RANGE - MIN_RANGE
    sx: int
    sy: int
    center: t.Tuple[float, float] = dataclasses.field(init=False)

    def __post_init__(self):
        self.center = (
            self.sx * SECTOR_SIZE[0] + SECTOR_SIZE[0] / 2,
            self.sy * SECTOR_SIZE[1] + SECTOR_SIZE[1] / 2,
        )

    def distance(self, point):
        return math.dist(self.center, point)

    def scan_boost(self, towers, global_boost=True) -> int:
        """Scan boost for this Sector given the towers placement"""
        # https://github.com/surviving-mars/SurvivingMars/blob/master/Lua/Exploration.lua#L284
        # function MapSector:GetTowerBoost(city)
        boost = u.clamp(len(towers) * NUM_BOOST, NUM_BOOST_MAX) if global_boost else 0
        best = min(self.distance(tower) for tower in towers) if len(towers) else MAX_RANGE
        if best < MAX_RANGE:
            boost += u.scale(
                BOOST_MAX, u.clamp(MAX_RANGE - best, self.BOOST_RANGE), self.BOOST_RANGE
            )
        log.debug("Sector boost (%s, %s) = %s", self.sx, self.sy, boost)
        return boost

    @classmethod
    def map_scan_boost(cls, towers, global_boost=True) -> np.ndarray:
        """Scan boost for all sectors in given the towers placement"""

        def sector_scan_boost(sx, sy):
            return cls(sx, sy).scan_boost(towers, global_boost)

        return np.fromfunction(np.vectorize(sector_scan_boost, otypes=[int]), SECTOR_GRID).T


@tower_generator(side="grid_side")
def inner_grid(side: int, area_size=MAP_SIZE):
    """
    Coordinates of a side x side grid of points evenly spaced on an outer area

    The grid is also equally spaced away from the area borders, hence "inner" grid
    Coordinates of a side 3 grid (9 points) over a (400, 400) area are:
    [(100, 100), (100, 200), (100, 300),
     (200, 100), (200, 200), (200, 300),
     (300, 100), (300, 200), (300, 300)]
    """
    axes = (np.linspace(0, s, num=int(side) + 2)[1:-1] for s in area_size)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.vstack(list(map(np.ravel, mesh))).T


@tower_generator(side="grid_side", margin="grid_margin")
def margin_grid(side: int, margin=SECTOR_SIZE, area_size=MAP_SIZE):
    """
    Coordinates of a side x side grid of evenly-spaced points inside an area

    The grid is spaced away from the area borders by a margin, hence "margin" grid
    Coordinates of a side 3 grid over a (400, 400) area and a (40, 40) margin:
    [( 40,  40), ( 40, 200), ( 40, 360),
     (200,  40), (200, 200), (200, 360),
     (360,  40), (360, 200), (360, 360)]
    """
    axes = np.linspace(margin, np.subtract(area_size, margin), num=side)
    if not len(axes):
        axes = [[np.array([])] * 2]
    mesh = np.meshgrid(*zip(*axes), indexing="ij")
    return np.vstack(list(map(np.ravel, mesh))).T


@tower_generator("path")
def custom_grid(path):
    """
    Read custom points from numpy-generated text file
    """
    return np.loadtxt(path, delimiter=",")


def parse_args(argv=None):
    parser = u.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-l",
        "--towers-layout",
        dest="function",
        default=TOWERS_GRID_LAYOUT,
        choices=(tower_generator.functions.keys()),
        help="Method to generate towers [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--grid-side",
        default=TOWERS_GRID_SIDE,
        type=int,
        help="Towers grid side [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--grid-margin",
        default=TOWERS_GRID_MARGIN,
        type=float,
        nargs="+",
        metavar=("SX", "SY"),
        help=(
            "Towers grid margins, in Sectors. Consider SY=SX if SY is not specified. "
            "[Default: %(default)s]"
        ),
    )
    parser.add_argument(
        "-f",
        "--towers-file",
        dest="path",
        help=(
            "File with custom towers placement in Numpy text format. "
            "Implies '--layout custom', and is required by it."
        )
    )
    parser.add_argument(
        "-N",
        "--no-show",
        dest="show",
        default=True,
        action="store_false",
        help="Do not open a window to display the heatmap",
    )
    parser.add_argument(
        "-G",
        "--no-global-boost",
        dest="global_boost",
        default=True,
        action="store_false",
        help="Do not consider the map-wide boost from the global number of towers",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="",
        help=f"Force a matplotlib backend. [Current: {plt.get_backend()}]",
    )
    parser.add_argument(
        "-p",
        "--palette",
        default=HEATMAP_COLORS,
        help=f"Color palette ('cmap') to use in the heatmap. [Default: %(default)s]",
    )
    args = parser.parse_args(argv, log_args=False)

    # Post-process grid margin
    if len(args.grid_margin) == 1:  # make SY=SX if only SX
        args.grid_margin = args.grid_margin * 2
    elif len(args.grid_margin) > 2:  # emulate nargs="{1,2}"
        parser.error("expected at most two arguments", "-m")
    # Scale from Sector coordinates (sx, sy) to Map coordinates (x, y)
    args.grid_margin = [m * s for m, s in zip(args.grid_margin, SECTOR_SIZE)]

    # Post-process file and custom layout
    if args.function == "custom" and args.path is None:
        parser.error("file is required when using --layout custom", "-f")
    if args.path is not None:
        args.function = "custom"

    # Post-process tower generator function and args
    function = tower_generator.functions[args.function]
    args.function = function.func
    args.function_args = {k: getattr(args, k) for k in function.params}
    args.function_args.update({k: getattr(args, v) for k, v in function.params_map.items()})

    log.debug(args)
    return args


def draw_heatmap(
    data,
    title: str = "",
    palette: str = HEATMAP_COLORS,
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


def draw_towers(towers, tower_size=20, tower_color="red", ax: plt.Axes = None):
    # Scale towers from (x, y) to (sx, sy) to fit on heatmap and reshape as meshgrid
    tower_data = [*zip(*np.divide(towers, SECTOR_SIZE))] if len(towers) else ([], [])
    log.debug(f"Tower data for scatter plot:\n%s", tower_data)
    if ax is None:
        ax = plt.gca()  # == plt for scatter purposes
    return ax.scatter(*tower_data, marker="*", s=tower_size**2, c=tower_color, picker=True)


def main(argv: t.Optional[t.List[str]] = None):
    args = parse_args(argv)

    towers = args.function(**args.function_args)
    log.info(f"Towers coordinates:\n{towers}")
    data = MapSector.map_scan_boost(towers, args.global_boost)
    log.info(f"Scan Boost per Sector:\n{data}")
    stats = Statistics(data, towers, args.global_boost)

    if not args.show:
        return

    u.init_window(backend=args.backend)
    heatmap = draw_heatmap(data, title=stats.title, palette=args.palette)
    _towers = draw_towers(towers, ax=heatmap)

    u.show_window()


def run(argv: t.Optional[t.List[str]] = None) -> None:
    """CLI entry point, handling exceptions from main() and setting exit code"""
    try:
        main(argv)
    except KeyboardInterrupt:
        log.info("Aborting")
        sys.exit(2)  # signal.SIGINT.value, but not actually killed by SIGINT
