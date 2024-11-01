# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
"""
Constants, functions and classes from game data
"""
import dataclasses
import logging
import math
import typing as t

import numpy as np

from . import util as u

# Constants
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

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class MapSector:
    BOOST_RANGE: t.ClassVar[int] = MAX_RANGE - MIN_RANGE
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
        _log.debug("Sector boost (%s, %s) = %s", self.sx, self.sy, boost)
        return boost

    @classmethod
    def map_scan_boost(cls, towers, global_boost=True) -> np.ndarray:
        """Scan boost for all sectors in given the towers placement"""

        def sector_scan_boost(sx, sy):
            return cls(sx, sy).scan_boost(towers, global_boost)

        return np.fromfunction(np.vectorize(sector_scan_boost, otypes=[int]), SECTOR_GRID).T
