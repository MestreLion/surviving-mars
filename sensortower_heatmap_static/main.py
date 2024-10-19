#!/usr/bin/env python3
#
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>

"""
Surviving Mars Tower Sensor scan boost heatmap using Matlib
"""

import argparse
import dataclasses
import logging
import math
import os
import sys
import typing as t

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

# Constants from game data
# https://github.com/surviving-mars/SurvivingMars/blob/master/Lua/_GameConst.lua#L102
SECTOR_GRID = (10, 10)
SECTOR_SIZE = (40, 40)
BOOST_MAX = 390      # max boost provided by the nearest tower to the sector
MIN_RANGE =  20      # range of max boost (boost = max)
MAX_RANGE = 120      # range of no boost (boost = 0%)
NUM_BOOST =  10      # every working tower provides this much cumulative boost to all sectors
NUM_BOOST_MAX = 100  # max cumulative scan boost

# Derived constants
MAP_SIZE = np.multiply(SECTOR_GRID, SECTOR_SIZE)  # (400, 400)

# Parameters default values
TOWERS_GRID_RANK = 3

log = logging.getLogger(os.path.basename(os.path.splitext(__file__)[0]))


@dataclasses.dataclass
class MapSector:
    BOOST_RANGE: t.ClassVar = MAX_RANGE - MIN_RANGE
    sx: int
    sy: int

    @property
    def center(self):
        return (self.sx * SECTOR_SIZE[0] + SECTOR_SIZE[0] / 2,
                self.sy * SECTOR_SIZE[1] + SECTOR_SIZE[1] / 2)

    def distance(self, point):
        return math.dist(self.center, point)

    def scan_boost(self, towers) -> int:
        """Scan boost for this Sector given the towers placement"""
        # https://github.com/surviving-mars/SurvivingMars/blob/master/Lua/Exploration.lua#L284
        # function MapSector:GetTowerBoost(city)
        boost = clamp(len(towers) * NUM_BOOST, NUM_BOOST_MAX)
        best = min(self.distance(tower) for tower in towers) if len(towers) else MAX_RANGE
        if best < MAX_RANGE:
            boost += scale(BOOST_MAX,  clamp(MAX_RANGE - best, self.BOOST_RANGE), self.BOOST_RANGE)
        return boost

    @classmethod
    def map_scan_boost(cls, towers) -> np.ndarray:
        """Scan boost for all sectors in given the towers placement"""
        def sector_scan_boost(sx, sy):
            return cls(sx, sy).scan_boost(towers)
        return np.fromfunction(np.vectorize(sector_scan_boost, otypes=[int]), SECTOR_GRID)


def scale(value, numerator, denominator) -> int:
    """Helper function mimicking Surviving Mars' MulDivRound()"""
    return round(value * numerator / denominator)


def clamp(value, upper=None, lower=None):
    """Helper function to replace min()/max() usage as upper/lower bounds of a value"""
    v = value
    if upper is not None: v = min(value, upper)
    if lower is not None: v = max(value, lower)
    return v


def inner_grid(area_size, rank: int):
    """
    Coordinates of a rank x rank grid of points evenly spaced on an outer area

    The grid is also equally spaced away from the area borders, hence "inner" grid
    Coordinates of a rank 3 grid (9 points) over a (400, 400) area are:
    [(100, 100), (100, 200), (100, 300),
     (200, 100), (200, 200), (200, 300),
     (300, 100), (300, 200), (300, 300)]

    Return a 2-Tuple of the list of 2D coordinates and its corresponding Numpy meshgrid
    """
    axes = (np.linspace(0, s, num=rank+2)[1:-1] for s in area_size)
    grid = np.meshgrid(*axes, indexing='ij')
    return np.vstack(list(map(np.ravel, grid))).T, grid

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q",
        "--quiet",
        dest="loglevel",
        const=logging.WARNING,
        default=logging.INFO,
        action="store_const",
        help="Suppress informative messages.",
    )
    group.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        const=logging.DEBUG,
        action="store_const",
        help="Verbose mode, output extra info.",
    )

    parser.add_argument(
        "-r",
        "--rank",
        default=TOWERS_GRID_RANK,
        type=int,
        help='Towers Grid "Rank" [Default: %(default)s]'
    )

    args = parser.parse_args(argv)
    args.debug = args.loglevel == logging.DEBUG
    logging.basicConfig(
        level=args.loglevel,
        format="[%(asctime)s %(funcName)-5s %(levelname)-6.6s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.basicConfig(level=args.loglevel, format="%(levelname)-5.5s: %(message)s")
    log.debug(args)
    return args


def main(argv: t.Optional[t.List[str]] = None):
    args = parse_args(argv)
    towers, mesh = inner_grid(MAP_SIZE, args.rank)
    df = MapSector.map_scan_boost(towers)
    _, ax = plt.subplots()
    sns.set_theme()
    # Good color maps (append "_r" to reverse):
    # viridis, YlOrBr, Spectral, plasma, rainbow, jet
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    sns.heatmap(df, cmap="rainbow", annot=True, fmt='.0f', vmin=0, vmax=BOOST_MAX+NUM_BOOST_MAX, ax=ax, linewidths=1)
    plot = (m/s for m, s in zip(mesh, SECTOR_SIZE))  # Scale towers (x, y) to (sx, sy)
    plt.plot(*plot, marker='*', color="black", markersize=20, linestyle="")

    # Maximize window on display. Works on Ubuntu with TkAgg backend
    # See https://stackoverflow.com/q/12439588/624066
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(1)
