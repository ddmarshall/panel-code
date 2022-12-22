#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to read Theory of wing sections data from files."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type


def read_camber_data(filename: str) -> Tuple[np_type.NDArray,
                                             np_type.NDArray,
                                             np_type.NDArray]:
    """
    Read Theory of Wing Sections camber data from file.

    Parameters
    ----------
    filename : str
        Name of file to be read.

    Returns
    -------
    x : numpy.ndarray
        X-coordinate.
    y : numpy.ndarray
        Y-coordinate.
    dydx : numpy.ndarray
        Slope, dy/dx.
    """
    file = open(filename, "r", encoding="utf8")
    lines = file.readlines()
    file.close()

    header_offset = 5
    n = len(lines) - header_offset
    x = np.zeros(n)
    y = np.zeros(n)
    dydx = np.zeros(n)

    for i in range(0,n):
        col = lines[i + header_offset].split(",")
        x[i] = float(col[0])/100.0
        y[i] = float(col[1])/100.0
        dydx[i] = float(col[2])

    return x, y, dydx


def read_thickness_data(filename:str) -> Tuple[np_type.NDArray,
                                               np_type.NDArray]:
    """
    Read Theory of Wing Sections camber data from file.

    Parameters
    ----------
    filename : str
        Name of file to be read.

    Returns
    -------
    x : numpy.ndarray
        X-coordinate.
    y : numpy.ndarray
        Y-coordinate.
    """
    file = open(filename, "r", encoding="utf8")
    lines = file.readlines()
    file.close()

    header_offset = 3
    n = len(lines) - header_offset
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(0,n):
        col = lines[i + header_offset].split(",")
        x[i] = float(col[0])/100.0
        y[i] = float(col[1])/100.0

    return x, y


def read_airfoil_data(filename:str) -> Tuple[np_type.NDArray, np_type.NDArray]:
    """
    Read Theory of Wing Sections airfoil data from file.

    Parameters
    ----------
    filename : str
        Name of file to be read.

    Returns
    -------
    xu : numpy.ndarray
        Upper surface x-coordinate.
    yu : numpy.ndarray
        Upper surface y-coordinate.
    xl : numpy.ndarray
        Lower surface x-coordinate.
    yl : numpy.ndarray
        Lower surface y-coordinate.
    """
    file = open(filename, "r", encoding="utf8")
    lines = file.readlines()
    file.close()

    header_offset = 5
    n = lines[header_offset:-1].index("\n")
    xu = np.zeros(n)
    yu = np.zeros(n)

    for i in range(0,n):
        col = lines[i + header_offset].split(",")
        xu[i] = float(col[0])/100.0
        yu[i] = float(col[1])/100.0

    header_offset = header_offset + n + 3
    n = len(lines) - header_offset
    xl = np.zeros(n)
    yl = np.zeros(n)

    for i in range(0,n):
        col = lines[i + header_offset].split(",")
        xl[i] = float(col[0])/100.0
        yl[i] = float(col[1])/100.0

    return xu, yu, xl, yl
