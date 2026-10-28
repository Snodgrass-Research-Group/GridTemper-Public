"""Cost curves, economic dispatch, and net-injection construction.

MATLAB origin: costcurve_to_cubic.m, UCED.m, Create_SNet.m.
Algorithm doc: conversion_manual/algorithms/03_dispatch.md.
"""

from .cost_curves import fit_quadratic_costs
from .snet import build_snet
from .uced import EconomicDispatch

__all__ = ["fit_quadratic_costs", "build_snet", "EconomicDispatch"]
