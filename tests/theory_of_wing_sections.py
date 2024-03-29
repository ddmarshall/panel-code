#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes to get Theory of wing sections data from files."""

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
        with open(filename, "r", encoding="utf8") as file:
            lines = file.readlines()

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

    # pylint: disable=too-many-instance-attributes

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
        with open(filename, "r", encoding="utf8") as file:
            lines = file.readlines()

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


class airfoil_data:
    """
    Theory of Wing Sections airfoil data class.

    Attributes
    ----------
    le_radius : float
        Leading edge radius
    le_radius_slope : float
        Slope at leading edge
    x_upper : numpy.ndarray
        Upper surface x-coordinate
    y_upper : numpy.ndarray
        Upper surface y-coordinate
    x_lower : numpy.ndarray
        Lower surface x-coordinate
    y_lower : numpy.ndarray
        Lower surface y-coordinate
    """

    def __init__(self, filename: str):
        self._le_radius = 0.0
        self._le_radius_slope = 0.0
        self._x_upper = np.zeros(1)
        self._y_upper = np.zeros(1)
        self._x_lower = np.zeros(1)
        self._y_lower = np.zeros(1)
        if filename is not None:
            self.change_case_data(filename=filename)

    @property
    def le_radius(self) -> float:
        """Leading edge radius."""
        return self._le_radius

    @property
    def le_radius_slope(self) -> float:
        """Leading edge radius slope."""
        return self._le_radius_slope

    @property
    def x_upper(self) -> np_type.NDArray:
        """Upper surface x-coordinate."""
        return self._x_upper

    @property
    def y_upper(self) -> np_type.NDArray:
        """Upper surface y-coordinate."""
        return self._y_upper

    @property
    def x_lower(self) -> np_type.NDArray:
        """Lower surface x-coordinate."""
        return self._x_lower

    @property
    def y_lower(self) -> np_type.NDArray:
        """Lower surface y-coordinate."""
        return self._y_lower

    def change_case_data(self, filename: str) -> None:
        """
        Change case data to be stored.

        Parameters
        ----------
        filename : str
            Filename of the new case data.
        """
        # read in data
        with open(filename, "r", encoding="utf8") as file:
            lines = file.readlines()

        # get header info
        self._le_radius = float(lines[0][12:-1])/100.0
        self._le_radius_slope = float(lines[1][29:-1])

        # get rest of data
        header_offset = 5
        n = lines[header_offset:-1].index("\n")
        self._x_upper = np.zeros(n)
        self._y_upper = np.zeros(n)

        for i in range(0,n):
            col = lines[i + header_offset].split(",")
            self._x_upper[i] = float(col[0])/100.0
            self._y_upper[i] = float(col[1])/100.0

        header_offset = header_offset + n + 3
        n = len(lines) - header_offset
        self._x_lower = np.zeros(n)
        self._y_lower = np.zeros(n)

        for i in range(0,n):
            col = lines[i + header_offset].split(",")
            self._x_lower[i] = float(col[0])/100.0
            self._y_lower[i] = float(col[1])/100.0
