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
    the parameter, t.
    """

    #
    # Parameteric Interface
    #
    @abstractmethod
    def xy(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

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

    @abstractmethod
    def xy_t(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

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

    @abstractmethod
    def xy_tt(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at parameter location.

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

    def normal(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Calculate the unit normal at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Unit normal at point.
        """
        sx, sy = self.tangent(t)
        nx = -sy
        ny = sx
        return nx, ny

    def tangent(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                   np_type.NDArray]:
        """
        Calculate the unit tangent at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Unit tangent at point.
        """
        sx, sy = self.xy_t(t)
        temp = np.sqrt(sx**2 + sy**2)
        sx = sx/temp
        sy = sy/temp
        return sx, sy

    def k(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the curvature at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Curvature of surface at point.
        """
        xt, yt = self.xy_t(t)
        xtt, ytt = self.xy_tt(t)
        return (xt*ytt-yt*xtt)/(xt**2+yt**2)**(3/2)

    def arc_length(self, t_s: float,
                   t_e: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the arc-length distance between two points on surface.

        Parameters
        ----------
        t_s : float
            Start point of distance calculation.
        t_e : numpy.ndarray
            End point of distance calculation.

        Returns
        -------
        numpy.ndarray
            Distance from start point to end point.
        """

        def fun(t):
            xt, yt = self.xy_t(t)
            return np.sqrt(xt**2+yt**2)

        t_e = np.asarray(t_e, dtype=np.float64)
        it = np.nditer([t_e, None])
        with it:
            for ti, alen in it:
                segment_ends = [x for x in self.joints()
                                if (x < ti) and (x > t_s)]
                segment_ends.append(ti)
                t_begin = t_s
                alen[...] = 0.0
                for t_end in segment_ends:
                    alen[...] += quadrature(fun, t_begin, t_end)[0]
                    t_begin = t_end

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
            Parametric coordinates of any discontinuities.
        """
