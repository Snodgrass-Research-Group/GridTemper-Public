"""Input readers and reference-data loaders.

MATLAB origin: convertPWtoTable.m + the load() calls scattered through the
pipeline. Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md.
"""

from .pw_csv import read_pw_table
from .reference_data import load_conductor_table, load_gmd_data, load_xf_data

__all__ = ["read_pw_table", "load_conductor_table", "load_gmd_data", "load_xf_data"]
