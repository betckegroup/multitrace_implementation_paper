"""Global initialization for Bempp-UQ."""

from __future__ import absolute_import

__all__ = [
    "assembly",
    "functions",
    "operators",
    "preconditioning",
    "shapes",
    "utils",
    "config",
]

from . import (
    assembly,
    functions,
    operators,
    preconditioning,
    shapes,
    utils,
    config
)