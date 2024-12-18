# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
"""
Surviving Mars Sensor Towers scan boost heatmap using Matplotlib
"""
import dataclasses
import logging
import sys
import typing as t

import matplotlib.pyplot as plt
import numpy as np

from . import util as u

__version__ = "2024.10"

from .gamedata import (
    SECTOR_SIZE,
    NUM_BOOST,
    MAP_SIZE,
    MapSector,
)
from . import drawing as d

# Parameters default values
TOWERS_GRID_LAYOUT = "margin"
TOWERS_GRID_SIDE = 3
TOWERS_GRID_MARGIN = (2, 2)

log = logging.getLogger(__name__)
tower_generator = u.FunctionCollectionDecorator(remove_suffix="_grid")


@dataclasses.dataclass
class Statistics:
    num: int
    avg: float
    cost: float
    title: str

    # Average Sector boost for 1 Tower on map center and no global boost
    COST_REFERENCE: t.ClassVar[float] = 44.12

    def __init__(self, data, towers, global_boost: bool = True):
        self.num, self.avg, self.cost = self.statistics(data, towers, global_boost)
        self.title = (
            f"{self.num} Towers, "
            f"{self.avg:6.2f} mean boost, "
            f"normalized cost {self.cost:.1%}"
        )

    @classmethod
    def statistics(cls, data, towers, global_boost: bool = True):
        num, avg = len(towers), np.average(data)
        if num:
            avg_tower = avg / num
            cost = (cls.COST_REFERENCE + (NUM_BOOST if global_boost else 0)) / avg_tower
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
        ),
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
        default=d.HEATMAP_PALETTE,
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


def main(argv: t.Optional[t.List[str]] = None):
    args = parse_args(argv)

    towers = args.function(**args.function_args)
    log.info(f"Towers coordinates:\n{towers}")
    data = MapSector.map_scan_boost(towers, args.global_boost)
    log.info(f"Scan Boost per Sector:\n{data}")
    stats = Statistics(data, towers, args.global_boost)

    if not args.show:
        return

    d.init_window(backend=args.backend)
    heatmap = d.draw_seaborn_heatmap(data, title=stats.title, palette=args.palette)
    _towers = d.draw_towers(towers, ax=heatmap)

    d.show_window()


def run(argv: t.Optional[t.List[str]] = None) -> None:
    """CLI entry point, handling exceptions from main() and setting exit code"""
    try:
        main(argv)
    except KeyboardInterrupt:
        log.info("Aborting")
        sys.exit(2)  # signal.SIGINT.value, but not actually killed by SIGINT
