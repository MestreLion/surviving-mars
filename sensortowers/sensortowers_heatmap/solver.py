# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>

"""
Functions to find best tower placement to maximize scan boost
"""
import datetime
import functools
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco

from . import main as m
from . import util as u

# if len(sys.argv) <= 2:
#     print("Usage: solver TOWERS [ITERATIONS=10]", file=sys.stderr)
#     sys.exit(1)
NUM_TOWERS = int(sys.argv[1]) if len(sys.argv) >= 2 else 9
SIDE = 3
MARGINS = {
    1: np.divide(m.MAP_SIZE, 2),
    2: (100, 100),
    3: (80, 80),
}
if not NUM_TOWERS:
    NUM_TOWERS = SIDE ** 2
SHAPE_TOWER = (NUM_TOWERS, 2)
BOUNDS = [(0, dim) for dim in m.MAP_SIZE]
u.init_window()
plt.xlim(BOUNDS[0])
plt.ylim(BOUNDS[1])
plt.gca().set_aspect("equal")
plt.xticks(np.linspace(*BOUNDS[0], num=m.SECTOR_GRID[0]+1))
plt.yticks(np.linspace(*BOUNDS[1], num=m.SECTOR_GRID[1]+1))

def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


@count_calls
def mean_boost(flat: np.ndarray):
    result = m.MapSector.map_scan_boost(flat.reshape(SHAPE_TOWER))
    return -result.mean()


def dual_annealing(func, bounds, maxiter=1000):
    print(f"optimizing {func} with bounds={bounds[:2]}..., maxiter={maxiter}")
    # return sco.minimize(func, x0, bounds=bounds)
    return sco.dual_annealing(func, bounds=bounds, maxiter=maxiter)


def minimize(func, x0, bounds):
    print(f"optimizing {func}({x0}) with bounds={bounds}")
    return sco.minimize(func, x0, bounds=bounds)


def sorted_points(arr:np.ndarray):
    return arr[np.lexsort(np.transpose(arr)[::-1])].copy()


def save_solution(arr:np.ndarray):
    path = f"solution_{time.time_ns()}.txt"
    np.savetxt(path, arr, fmt='%6.2f', delimiter=", ")
    return path


def draw_towers(towers, size=20, color="red", marker="*", label="Tower", mean=0.0):
    return plt.scatter(*zip(*towers), marker=marker, s=size**2, c=color, label=f"{mean:6.2f}: {label}")


def measure(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    delta = time.time() - start
    print(f"Results:\n{result!r}")
    if hasattr(result, "x"):
        solution = sorted_points(result.x.reshape(SHAPE_TOWER))
        path = save_solution(solution)
    else:
        try:
            solution = "\n".join(f"\t{k} = {v!r}" for k, v in sorted(vars(result).items()))
        except TypeError:
            pass
            solution = None
        path = ""
    val = ""
    if hasattr(result, "fun") and isinstance(result.fun, float):
        val = f", Boost={result.fun:.2f}"
    print(f"Solution:\n{solution}{val}")
    print(f"mean_boost() calls: {mean_boost.calls}")
    if path:
        print(f"Saved as: {path}")
    print(f"Runtime: {datetime.timedelta(seconds=int(delta))}")
    return solution, -result.fun if path else result


def main():
    maxiter = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

    for towers, label, color in (
        (m.inner_grid(SIDE), "Inner", "blue"),
        (m.margin_grid(SIDE, MARGINS[SIDE]), "Margin", "red"),
    ):
        mean = m.MapSector.map_scan_boost(towers).mean()
        # draw_towers(towers, label=label, color=color, mean=mean)
        print(f"{towers}, {mean:6.2f} {label}, {len(towers)} towers")
    # func = count_calls(lambda x: len(set(x)) * abs(100 - x.mean()) )
    func = mean_boost
    bounds = BOUNDS * NUM_TOWERS
    # x0 = np.zeros((2 * NUM_TOWERS,))
    towers, value = measure(dual_annealing, func, bounds=bounds, maxiter=maxiter)
    if isinstance(towers, np.ndarray):
        draw_towers(towers, color="green", marker="o", size=10, label="Best", mean=value)
        plt.legend()
        u.show_window()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(2)
