# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
"""
Miscellaneous utility functions
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
import typing_extensions as t

AnyFunction: t.TypeAlias = t.Callable[..., t.Any]
log: logging.Logger = logging.getLogger(__name__)

# Disable debug logging for some chatty libs
for _ in ("matplotlib", "PIL"):
    logging.getLogger(_).setLevel(logging.INFO)


def clamp(value, upper=None, lower=None):
    """Helper function to replace min()/max() usage as upper/lower bounds of a value"""
    v = value

    if upper is not None:
        v = min(value, upper)
    if lower is not None:
        v = max(value, lower)
    return v


def timestamp(secs: t.Optional[int] = None) -> str:
    """Compact date and time string in local time from seconds since UNIX epoch

    Output format is suitable as a compact timestamp for logs and filenames
    Example: timestamp(1234567890) -> '20090213213130'
    """
    return time.strftime("%Y%m%d%H%M%S", time.localtime(secs))


if sys.version_info >= (3, 9):
    removesuffix = str.removesuffix  # noqa
else:

    def removesuffix(self: str, suffix: str, /) -> str:
        """str.removesuffix() for Python < 3.9: https://peps.python.org/pep-0616"""
        if suffix and self.endswith(suffix):
            return self[: -len(suffix)]
        else:
            return self[:]


@dataclasses.dataclass
class FunctionParams:
    func: AnyFunction
    params: t.Tuple[str, ...] = ()
    params_map: t.Dict[str, str] = dataclasses.field(default_factory=dict)

    def __repr__(self):
        return "<{} {}({})>".format(
            self.__class__.__name__,
            self.func.__qualname__,
            ", ".join(
                [*self.params, *["=".join((k, repr(v))) for k, v in self.params_map.items()]]
            ),
        )


class FunctionCollectionDecorator(dict):
    """
    Factory for decorators that keep track of decorated functions

    Can also record argument names and mappings, useful for argparse purposes.
    Usage:
        my_decorator = FunctionCollectionDecorator(remove_suffix="_suf")
        @my_decorator
        def f1_suf(...): ...
        @my_decorator
        def f2(...): ...
        @my_decorator("a", "b", foo="bar")
        def f3(...): ...
        print(my_decorator.functions)  # or simply `print(my_decorator)`
            {'func1': <FunctionParams f1_suffix()>,
             'func2': <FunctionParams f2()>,
             'func3': <FunctionParams f3(a, b, foo='bar')>}
    """

    def __init__(self, *args, **kwargs):
        self.suffix = kwargs.pop("remove_suffix", "")
        super().__init__(*args, **kwargs)

    def __call__(
        self, arg: t.Union[AnyFunction, str, None] = None, /, *args: str, **kwargs: str
    ):
        def decorator(func: AnyFunction):
            self[removesuffix(func.__name__, self.suffix)] = FunctionParams(func, args, kwargs)
            return func

        if callable(arg) and not (args or kwargs):
            # Assume decoration without arguments
            return decorator(arg)
        if arg is not None:
            args = (arg, *args)
        return decorator

    @property
    def functions(self) -> t.Dict[str, FunctionParams]:
        return self


class ArgumentParser(argparse.ArgumentParser):
    __doc__ = (
        (argparse.ArgumentParser.__doc__ or "") + "\nWith many changes and new features"
    ).strip()
    COPYRIGHT = """
    Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
    License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
    """.replace(
        "    ", ""
    )

    FileType = argparse.FileType

    def __init__(
        self,
        *args: t.Any,
        multiline: bool = False,
        loglevel_options: str = "loglevel",
        debug_option: str = "debug",
        version: t.Optional[str] = None,
        **kwargs: t.Any,
    ):
        super().__init__(*args, **kwargs)

        if self.description is not None and not multiline:
            self.description = self.description.strip().split("\n", maxsplit=1)[0]

        if not self.epilog:
            if self.epilog is None:
                self.epilog = self.COPYRIGHT
                self.formatter_class = argparse.RawDescriptionHelpFormatter
            else:
                self.epilog = None

        self.loglevel_options = loglevel_options
        self.debug_option = debug_option

        if self.loglevel_options:
            group = self.add_mutually_exclusive_group()
            group.add_argument(
                "-q",
                "--quiet",
                dest=self.loglevel_options,
                const=logging.WARNING,
                default=logging.INFO,
                action="store_const",
                help="Suppress informative messages.",
            )
            group.add_argument(
                "-v",
                "--verbose",
                dest=self.loglevel_options,
                const=logging.DEBUG,
                action="store_const",
                help="Verbose mode, output extra info.",
            )

        if version:
            self.add_argument(
                "-V",
                "--version",
                action="version",
                version=f"%(prog)s {version}",
            )

    def parse_args(  # type: ignore  # accurate typing requires overload
        self, *args: t.Any, log_args=True, **kwargs: t.Any
    ) -> argparse.Namespace:
        __doc__ = argparse.ArgumentParser.parse_args.__doc__
        arguments: argparse.Namespace = super().parse_args(*args, **kwargs)
        if self.debug_option and self.loglevel_options:
            setattr(
                arguments,
                self.debug_option,
                getattr(arguments, self.loglevel_options) == logging.DEBUG,
            )
            logging.basicConfig(
                level=getattr(arguments, self.loglevel_options),
                format="[%(asctime)s %(funcName)s %(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if log_args:
                log.debug(arguments)
        return arguments

    def error(self, message, argument: str = ""):
        if not argument:
            super().error(message)
        for action in self._actions:
            if argument in action.option_strings:
                super().error(str(argparse.ArgumentError(action, message)))
        raise AssertionError(f"No such command line option: {argument}")
