#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example implementation of a cylinder shape as an airfoil."""

from typing import Tuple

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
        self._r = radius

    @property
    def radius(self) -> float:
        """Radius of the cylinder."""
        return self._r

    @radius.setter
    def radius(self, radius) -> float:
        self._r = radius

    def camber(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        return np.zeros_like(xi)

    def thickness(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        _, th = self.xy_from_xi(xi)
        return th

    def xy_from_xi(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                       np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            X-coordinate of point.
        numpy.ndarray
            Y-coordinate of point.
        """
        theta = np.pi*(1-xi)
        x = self.radius*(1+np.cos(theta))
        y = self.radius*np.sin(theta)
        return x, y

    def xy_p(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric rate of change of the x-coordinate of point.
        numpy.ndarray
            Parametric rate of change of the y-coordinate of point.
        """
        theta = np.pi*(1-xi)
        x_p = np.pi*self.radius*np.sin(theta)
        y_p = -np.pi*self.radius*np.cos(theta)
        return x_p, y_p

    def xy_pp(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Return second derivative of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric second derivative of the x-coordinate of point.
        numpy.ndarray
            Parametric second derivative of the y-coordinate of point.
        """
        theta = np.pi*(1-xi)
        x_pp = -np.pi**2*self.radius*np.cos(theta)
        y_pp = -np.pi**2*self.radius*np.sin(theta)
        return x_pp, y_pp
