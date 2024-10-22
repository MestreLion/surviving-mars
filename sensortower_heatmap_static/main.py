#!/usr/bin/env python3
#
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

import util as u

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
SECTOR_MAX_BOOST = BOOST_MAX + NUM_BOOST_MAX  # 490
COST_REFERENCE = 44.12  # Average Sector boost for 1 Tower on map center

# Parameters default values
TOWERS_GRID_RANK = 3

log = logging.getLogger(__name__.replace("__", ""))

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
        return boost

    @classmethod
    def map_scan_boost(cls, towers, global_boost=True) -> np.ndarray:
        """Scan boost for all sectors in given the towers placement"""
        def sector_scan_boost(sx, sy):
            return cls(sx, sy).scan_boost(towers, global_boost)
        return np.fromfunction(np.vectorize(sector_scan_boost, otypes=[int]), SECTOR_GRID)


def inner_grid(area_size, rank: int):
    """
    Coordinates of a rank x rank grid of points evenly spaced on an outer area

    The grid is also equally spaced away from the area borders, hence "inner" grid
    Coordinates of a rank 3 grid (9 points) over a (400, 400) area are:
    [(100, 100), (100, 200), (100, 300),
     (200, 100), (200, 200), (200, 300),
     (300, 100), (300, 200), (300, 300)]
    """
    axes = (np.linspace(0, s, num=rank+2)[1:-1] for s in area_size)
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.vstack(list(map(np.ravel, mesh))).T



def parse_args(argv=None):
    parser = u.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--rank",
        default=TOWERS_GRID_RANK,
        type=int,
        help='Towers Grid "Rank" [Default: %(default)s]'
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
    return parser.parse_args(argv)



def heatmap(data, towers, title="", palette="viridis", tower_size=20, tower_color="red"):
    # Good colormaps ("_r" means reversed):
    # turbo, rainbow, plasma, viridis, gnuplot2, YlOrBr_r, Spectral_r
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html

    # Sectors Heatmap
    sns.set_theme()
    ax = sns.heatmap(
        data,
        cmap=palette,
        annot=True,
        fmt='.0f',
        vmin=0,
        vmax=SECTOR_MAX_BOOST,
        linewidths=1,
        square=True,
        xticklabels=string.ascii_uppercase[0:SECTOR_GRID[0]],
        mask=(data == 0),
    )

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

    # Tower markers
    # Scale towers from (x, y) to (sx, sy) to fit on heatmap and reshape as meshgrid
    if len(towers):
        tower_data = tuple(np.divide(m, s) for m, s in zip(zip(*towers), SECTOR_SIZE))
        plt.scatter(*tower_data, marker="*", s=tower_size**2, c=tower_color)

    if title:
        plt.title(title)

def main(argv: t.Optional[t.List[str]] = None):
    args = parse_args(argv)

    towers = inner_grid(MAP_SIZE, args.rank)
    data = MapSector.map_scan_boost(towers, args.global_boost)

    # Statistics
    num, avg = len(towers), np.average(data)
    if num:
        avg_tower = avg / num
        cost = (COST_REFERENCE + (NUM_BOOST if args.global_boost else 0)) / avg_tower
    else:
        avg_tower = cost = 0
    log.info(f"Towers coordinates:\n{towers}")
    log.info(f"Scan Boost per Sector:\n{data}")
    log.info(
        "Statistics:\n"
        f"Towers: {num}\n"
        f"Average Boost per Sector: {avg:6.2f}\n"
        f"Sector average per Tower: {avg_tower:6.2f}\n"
        f"Normalized cost: {cost:.1%}"
    )

    if args.show:
        heatmap(
            data,
            towers,
            title=f"{num} Towers, {avg:6.2f} mean boost, normalized cost {cost:.1%}",
        )
        u.show_window("Surviving Mars' Sensor Towers scan boost")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(1)
