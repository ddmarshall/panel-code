#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes to get Theory of wing sections data from files."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type


class thickness_data:
    """
    Theory of Wing Sections thickness data class.

    Attributes
    ----------
    le_radius : float
        Leading edge radius
    x : numpy.ndarray
        X-coordinate
    y : numpy.ndarray
        Y-coordinate
    v : numpy.ndarray
        Relative verical velocity
    delta_va : numpy.ndarray
        Relative increment of vertical velocity due to angle of attack
    """

    def __init__(self, filename: str):
        self._le_radius = 0.0
        self._x = np.zeros(1)
        self._y = np.zeros(1)
        self._v = np.zeros(1)
        self._delta_va = np.zeros(1)
        if filename is not None:
            self.change_case_data(filename=filename)

    @property
    def le_radius(self) -> float:
        """Leading edge radius."""
        return self._le_radius

    @property
    def x(self) -> np_type.NDArray:
        """X-coordinate."""
        return self._x

    @property
    def y(self) -> np_type.NDArray:
        """Y-coordinate."""
        return self._y

    @property
    def v(self) -> np_type.NDArray:
        """Relative vertical velocity."""
        return self._v

    @property
    def delta_va(self) -> np_type.NDArray:
        """Relative increment of vertical velocity due to angle of attack."""
        return self._delta_va

    def change_case_data(self, filename: str) -> None:
        """
        Change case data to be stored.

        Parameters
        ----------
        filename : str
            Filename of the new case data.
        """
        # read in data
        file = open(filename, "r", encoding="utf8")
        lines = file.readlines()
        file.close()

        # get header info
        self._le_radius = float(lines[0][12:-1])/100.0

        # get rest of data
        header_offset = 3
        n = len(lines) - header_offset
        self._x = np.zeros(n)
        self._y = np.zeros(n)
        self._v = np.zeros(n)
        self._delta_va = np.zeros(n)

        for i in range(0,n):
            col = lines[i + header_offset].split(",")
            self._x[i] = float(col[0])/100.0
            self._y[i] = float(col[1])/100.0
            self._v[i] = float(col[3])
            self._delta_va[i] = float(col[4])


class camber_data:
    """
    Theory of Wing Sections thickness data class.

    Attributes
    ----------
    ideal_Cl : float
        Ideal lift coefficient
    ideal_alpha : float
        Ideal angle of attack (radians)
    Cm_c4 : float
        Quarter-chord moment coefficient
    x : numpy.ndarray
        X-coordinate
    y : numpy.ndarray
        Y-coordinate
    dydx : numpy.ndarray
        Slope of camber line
    delta_v : numpy.ndarray
        Relative increment of vertical velocity due to camber
    """

    def __init__(self, filename: str):
        self._ideal_Cl = 0.0
        self._ideal_alpha = 0.0
        self._Cm_c4 = 0.0
        self._x = np.zeros(1)
        self._y = np.zeros(1)
        self._dydx = np.zeros(1)
        self._delta_va = np.zeros(1)
        if filename is not None:
            self.change_case_data(filename=filename)

    @property
    def ideal_Cl(self) -> float:
        """Ideal lift coefficient."""
        return self._ideal_Cl

    @property
    def ideal_alpha(self) -> float:
        """Ideal angle of attack."""
        return self._ideal_alpha

    @property
    def Cm_c4(self) -> float:
        """Quarter-chord moment coefficient."""
        return self._Cm_c4

    @property
    def x(self) -> np_type.NDArray:
        """X-coordinate."""
        return self._x

    @property
    def y(self) -> np_type.NDArray:
        """Y-coordinate."""
        return self._y

    @property
    def dydx(self) -> np_type.NDArray:
        """Slope of camber line."""
        return self._dydx

    @property
    def delta_v(self) -> np_type.NDArray:
        """Relative increment of vertical velocity due to camber."""
        return self._delta_v

    def change_case_data(self, filename: str) -> None:
        """
        Change case data to be stored.

        Parameters
        ----------
        filename : str
            Filename of the new case data.
        """
        # read in data
        file = open(filename, "r", encoding="utf8")
        lines = file.readlines()
        file.close()

        # get header info
        self._ideal_Cl = float(lines[0][5:-1])
        self._ideal_alpha = float(lines[1][8:-1])*np.pi/180.0
        self._Cm_c4 = float(lines[2][7:-1])

        # get rest of data
        header_offset = 5
        n = len(lines) - header_offset
        self._x = np.zeros(n)
        self._y = np.zeros(n)
        self._dydx = np.zeros(n)
        self._delta_v = np.zeros(n)

        for i in range(0,n):
            col = lines[i + header_offset].split(",")
            self._x[i] = float(col[0])/100.0
            self._y[i] = float(col[1])/100.0
            self._dydx[i] = float(col[2])
            self._delta_v[i] = float(col[3])


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
