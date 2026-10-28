"""Topology transforms.

MATLAB origin: split_add_GSUs.m, main.m renumbering loop, BuildANet.m.
Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md.
"""

from .gsu_split import split_add_gsus
from .incidence import build_incidence
from .numbering import to_internal_numbering

__all__ = ["split_add_gsus", "build_incidence", "to_internal_numbering"]
