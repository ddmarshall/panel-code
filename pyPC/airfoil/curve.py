#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with general curves."""

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type
from scipy.integrate import quadrature


class Curve(ABC):
    """
    Base class for 1-d curves.

    Curves can be interrogated based on their natural parameterization, using
    the parameter, xi.
    """

    #
    # Parameteric Interface
    #
    @abstractmethod
    def xy(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

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

    @abstractmethod
    def xy_p(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

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

    @abstractmethod
    def xy_pp(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at parameter location.

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

    def normal(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                   np_type.NDArray]:
        """
        Calculate the unit normal at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Unit normal at point.
        """
        sx, sy = self.tangent(xi)
        nx = -sy
        ny = sx
        return nx, ny

    def tangent(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                    np_type.NDArray]:
        """
        Calculate the unit tangent at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Unit tangent at point.
        """
        sx, sy = self.xy_p(xi)
        temp = np.sqrt(sx**2 + sy**2)
        sx = sx/temp
        sy = sy/temp
        return sx, sy

    def k(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the curvature at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Curvature of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xp, yp = self.xy_p(xi)
        xpp, ypp = self.xy_pp(xi)
        return (xp*ypp-yp*xpp)/(xp**2+yp**2)**(3/2)

    def arc_length(self, xi_s: float,
                   xi_e: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the arc-length distance between two points on surface.

        Parameters
        ----------
        xi_s : float
            Start point of distance calculation.
        xi_e : numpy.ndarray
            End point of distance calculation.

        Returns
        -------
        numpy.ndarray
            Distance from start point to end point.
        """

        def fun(xi):
            xp, yp = self.xy_p(xi)
            return np.sqrt(xp**2+yp**2)

        xi_ea = np.asarray(xi_e)
        it = np.nditer([xi_ea, None])
        with it:
            for xi, alen in it:
                alen[...], _ = quadrature(fun, xi_s, xi)

            return it.operands[1]

    @abstractmethod
    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the curve.

        The resulting list needs to contain any parameteric locations where
        some non-standard discontinuity (slope, curvature, etc.) occurs as
        well as the end points for the curve (if they exist).

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
