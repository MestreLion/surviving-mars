#!/usr/bin/env python3
#
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>

"""
Surviving Mars Martian Patents and Martian Copyrights optimal research order
"""

import argparse
import logging
import os
import sys
import typing as t

START = 1
LENGTH = 30

log = logging.getLogger(os.path.basename(os.path.splitext(__file__)[0]))

class Research:
    # TODO: Parametrize this for Patent and Copyright in cmd-line args or config file
    INITIAL = 100
    DELTA = 100
    GAIN = 100

    def __init__(self, cost: int = 0):
        self.cost = cost or self.INITIAL

    def next(self):
        return self.__class__(self.cost + self.DELTA)

    def ratio(self) -> float:
        return self.cost / self.GAIN

    def __lt__(self, other):
        if not isinstance(other, Research):
            return NotImplemented
        return self.ratio() < other.ratio()

    def __repr__(self):
        return "{}\t{:5}\t{:5.2f}".format(self.__class__.__name__, self.cost, self.ratio())

class Patent(Research):
    INITIAL = 1500
    DELTA = 300
    GAIN = 500

class Copyright(Research):
    INITIAL = 15000
    DELTA = 1500
    GAIN = 2000

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
        "-s", "--start", default=START, type=int, help="List start [Default: %(default)s]"
    )
    parser.add_argument(
        "-l", "--length", default=LENGTH, type=int, help="List length [Default: %(default)s]"
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

    length: int = args.length + args.start - 1
    items = []

    for research in Research.__subclasses__():
        item = research()
        for _ in range(length):
            items.append(item)
            item = item.next()

    for i, item in enumerate(sorted(items)[args.start - 1: length], args.start):
        print("{:3} {}".format(i, item))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as err:
        log.exception(err)
        sys.exit(1)
