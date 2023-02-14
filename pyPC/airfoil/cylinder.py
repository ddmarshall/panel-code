#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example implementation of a cylinder shape as an airfoil."""

from typing import Tuple, List

import numpy as np
import numpy.typing as np_type

from pyPC.airfoil.airfoil import Airfoil


class Cylinder(Airfoil):
    """
    Cylinderical shaped 2D airfoil.

    Attributes
    ----------
    radius : float
        Radius of the cylinder.
    """

    def __init__(self, radius) -> None:
        super().__init__()
        self._r = radius

    @property
    def radius(self) -> float:
        """Radius of the cylinder."""
        return self._r

    @radius.setter
    def radius(self, radius) -> float:
        self._r = radius

    def xy(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            X-coordinate of point.
        numpy.ndarray
            Y-coordinate of point.
        """
        theta = self._convert_theta(t)
        x = self.radius*(1+np.cos(theta))
        y = self.radius*np.sin(theta)
        return x, y

    def xy_t(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric rate of change of the x-coordinate of point.
        numpy.ndarray
            Parametric rate of change of the y-coordinate of point.
        """
        theta = self._convert_theta(t)
        x_t = np.pi*self.radius*np.sin(theta)
        y_t = -np.pi*self.radius*np.cos(theta)
        return x_t, y_t

    def xy_tt(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Return second derivative of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric second derivative of the x-coordinate of point.
        numpy.ndarray
            Parametric second derivative of the y-coordinate of point.
        """
        theta = self._convert_theta(t)
        x_tt = -np.pi**2*self.radius*np.cos(theta)
        y_tt = -np.pi**2*self.radius*np.sin(theta)
        return x_tt, y_tt

    def camber_location(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                           np_type.NDArray]:
        """
        Return the amount of camber at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

        Returns
        -------
        numpy.ndarray
            X-coordinate of camber at specified point.
        numpy.ndarray
            Y-coordinate of camber at specified point.
        """
        return t, np.zeros_like(t)

    def thickness_value(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        return self.xy(t)[1]

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the curve.

        Returns
        -------
        List[float]
            Parametric coordinates of any discontinuities.
        """
        return [-1.0, 1.0]

    @staticmethod
    def _convert_theta(t: np_type.NDArray) -> np_type.NDArray:
        return np.pi*(1-t)
