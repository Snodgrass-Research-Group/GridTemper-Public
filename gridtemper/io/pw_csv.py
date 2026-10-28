"""PowerWorld CSV reader — convertPWtoTable.m.

Algorithm doc: conversion_manual/algorithms/02_io_and_topology.md §convertPWtoTable.

MATLAB's ``readtable(filename,'ReadVariableNames',true)`` runs each raw header
through ``matlab.lang.makeValidName`` (R2024a warns and modifies), so the
column names the rest of the pipeline consumes are the *normalized* ones
(``"Number of Bus"`` -> ``NumberOfBus``, ``"Volt (kV)"`` -> ``Volt_kV_``,
``"# Cost Curve Points"`` -> ``x_CostCurvePoints``). :func:`read_pw_table`
reproduces that normalization exactly so downstream code can address columns by
the same identifiers it uses in MATLAB. The un-normalized headers are preserved
in ``df.attrs["pw_headers"]`` (MATLAB's second return, ``Info_header``).

MATLAB also parses ``a+bi`` cells (e.g. the demand ``S_pu`` column) as complex;
pandas leaves them as strings, so we convert any column whose values are all
MATLAB-style complex literals to a numpy complex dtype.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

def _to_complex(v: str) -> complex:
    """Parse a MATLAB complex literal (``a+bi``, ``bi``, or plain ``a``)."""
    return complex(v.replace("i", "j").replace(" ", ""))


def make_valid_name(name: str) -> str:
    """Reproduce ``matlab.lang.makeValidName`` (default options) for one header.

    Rules (matching MATLAB readtable's warned-modification behavior):

    1. Delete whitespace; uppercase the character that followed it.
    2. Replace every remaining character that is not ``[A-Za-z0-9_]`` with ``_``.
    3. If the result does not start with a letter, prepend ``x``.
    4. Empty input becomes ``x``.
    """
    chars: list[str] = []
    capitalize_next = False
    for ch in name:
        if ch.isspace():
            capitalize_next = True
            continue
        chars.append(ch.upper() if capitalize_next else ch)
        capitalize_next = False
    s = re.sub(r"[^0-9A-Za-z_]", "_", "".join(chars))
    if not s or not s[0].isalpha():
        s = "x" + s
    return s


def _maybe_complex(series: pd.Series) -> pd.Series:
    """Convert a string column of MATLAB complex literals to complex dtype.

    Only touches object columns where at least one value carries the imaginary
    unit ``i`` and *every* non-null value parses as complex — so name/text
    columns (``"#1"``, bus names) are never converted, while a column mixing
    ``a+bi`` and plain-real cells is.
    """
    # Numeric columns (int/float/complex) are already typed — pandas>=3.0 gives
    # string columns a dedicated ``str`` dtype (not ``object``), so gate on
    # "not numeric" rather than "== object".
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series
    non_null = series.dropna()
    if non_null.empty or not non_null.map(lambda v: isinstance(v, str)).all():
        return series
    if not non_null.map(lambda v: "i" in v).any():
        return series
    try:
        for v in non_null.to_numpy(dtype=object):
            _to_complex(v)
    except ValueError:
        return series  # something didn't parse — leave the column as-is
    # Build the complex values in numpy: under pandas>=3.0, ``map`` on an
    # Arrow-backed ``str`` column hands ``astype`` to pyarrow, which has no
    # complex type.
    values = np.array([_to_complex(v) if isinstance(v, str) else v
                       for v in series.to_numpy(dtype=object)], dtype=np.complex128)
    return pd.Series(values, index=series.index, name=series.name)


def read_pw_table(path: str | Path) -> pd.DataFrame:
    """Read a PowerWorld CSV export into a DataFrame.

    Mirrors ``convertPWtoTable(filename)``: the header row becomes column names,
    normalized the way MATLAB ``readtable`` does (see :func:`make_valid_name`).
    Original headers are preserved in ``df.attrs["pw_headers"]`` (MATLAB's
    second return, ``Info_header``). Columns of MATLAB complex literals are
    converted to complex dtype.

    Parameters
    ----------
    path : str or Path
        Path to a PowerWorld CSV export (may carry a UTF-8 BOM).

    Returns
    -------
    pandas.DataFrame
        Data with MATLAB-normalized column names; ``.attrs["pw_headers"]``
        holds the raw headers in column order.
    """
    # utf-8-sig transparently strips the BOM some PW exports carry (Demand).
    df = pd.read_csv(path, encoding="utf-8-sig")
    raw_headers = list(df.columns)
    df.columns = [make_valid_name(h) for h in raw_headers]
    for col in df.columns:
        df[col] = _maybe_complex(df[col])
    df.attrs["pw_headers"] = raw_headers
    return df
