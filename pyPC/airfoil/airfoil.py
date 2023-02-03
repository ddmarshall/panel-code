#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils and their analysis."""

from abc import abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type
from scipy.optimize import root_scalar

from pyPC.airfoil.curve import Curve
from pyPC.airfoil.camber import Camber
from pyPC.airfoil.thickness import Thickness


class Airfoil(Curve):
    """
    Base class for airfoil specific geometries.

    There are a few different ways that the airfoil coordinates (and their
    derivatives) can be queried.
        - The standard parameterization uses a range from -1 to 0 for the lower
          surface (trailing edge to leading edge) and from 0 to 1 for the
          upper surface (leading edge to trailing edge). This allows a smooth
          parametric variation for the entire airfoil and is consistent with
          how most airfoils are described in equation form.
        - Arc-length parameterization use the arc-length along the surface as
          the parameter for specifying the location on the surface. It also
          starts at the lower surface trailing edge and ends at the upper
          surface trailing edge. This method is typically computationally
          expensive because the arc-length is not a typical parameter that is
          easy to calculate for airfoil shapes.
        - Providing the x-coordinate and whether the upper or lower surface
          is desired. This method allows for the querying of actual points
          on the airfoil, but it might result in a ValueError exception raised
          if there is no surface point at the provided x-coordinate. This
          method is typically more computationally intensive since few airfoils
          are defined explicitly as a function of x-coordinate.

    The first two methods are exposed by the base class, while the last method
    is specific to airfoils.
    """

    #
    # Functional Interface
    #
    def y(self, x: np_type.NDArray, upper: bool) -> np_type.NDArray:
        """
        Calculate the y-coordinates at x-coordinate for upper or lower side.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate for desired locations.
        upper : bool
            True if want upper surface point otherwise get lower surface point.

        Returns
        -------
        numpy.ndarray
            Y-coordinate of point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xi = self._xi_from_x(x, upper)
        _, y = self.xy(xi)
        return y

    def dydx(self, x: np_type.NDArray, upper: bool) -> np_type.NDArray:
        """
        Calculate the slope at x-coordinate for upper or lower side.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate for desired locations.
        upper : bool
            True if want upper surface point otherwise get lower surface point.

        Returns
        -------
        numpy.ndarray
            Slope of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xi = self._xi_from_x(x, upper)
        xp, yp = self.xy_p(xi)
        return yp/xp

    def d2ydx2(self, x: np_type.NDArray, upper: bool) -> np_type.NDArray:
        """
        Calculate the second derivative for upper or lower side.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate for desired locations.
        upper : bool
            True if want upper surface point otherwise get lower surface point.

        Returns
        -------
        numpy.ndarray
            Second derivative of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xi = self._xi_from_x(x, upper)
        xp, yp = self.xy_p(xi)
        xpp, ypp = self.xy_pp(xi)
        return (xp*ypp-yp*xpp)/xp**3

    def _xi_from_x(self, x:np_type.NDArray, upper: bool) -> np_type.NDArray:
        """
        Calculate the parametric value for x-location provided.

        Parameters
        ----------
        x : np_type.NDArray
            X-coordinate of interest.
        upper : bool
            True if want upper surface point otherwise get lower surface point.

        Returns
        -------
        numpy.ndarray
            Parameteric value for location provided.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        x_a = np.asarray(x)
        xmin, _ = self.leading_edge()
        xmax, _ = self.trailing_edge()
        if ((x_a < xmin).any() or (x_a > xmax).any()):
            raise ValueError("Invalid x-coordinate provided.")

        def fun(xi: float, x: float) -> float:
            xr, _ = self.xy(xi)
            return xr - x

        it = np.nditer([x_a, None])
        if upper:
            bracket = [0, 1]
        else:
            bracket = [-1, 0]
        with it:
            # pylint: disable=cell-var-from-loop
            for xx, xi in it:
                if np.abs(fun(bracket[0], xx)) < 1e-8:
                    xi[...] = bracket[0]
                elif np.abs(fun(bracket[1], xx)) < 1e-8:
                    xi[...] = bracket[1]
                else:
                    root = root_scalar(lambda xi: fun(xi, xx), bracket=bracket)
                    xi[...] = root.root

            return it.operands[1]

    #
    # Arc-length Interface
    #
    def xy_from_s(self, s: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the coordinates of geometry at arc-length location.

        Parameters
        ----------
        s : numpy.ndarray
            Arc-length location for point.

        Returns
        -------
        numpy.ndarray
            X-coordinate of point.
        numpy.ndarray
            Y-coordinate of point.
        """
        xi = self._xi_from_s(s)
        return self.xy(xi)

    def xy_s(self, s: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at arc-length location.

        Parameters
        ----------
        s : numpy.ndarray
            Arc-length location for point.

        Returns
        -------
        numpy.ndarray
            Arc-length rate of change of the x-coordinate of point.
        numpy.ndarray
            Arc-length rate of change of the y-coordinate of point.
        """
        xi = self._xi_from_s(s)
        return self.tangent(xi)

    def xy_ss(self, s: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at arc-length location.

        Parameters
        ----------
        s : numpy.ndarray
            Arc-length location for point.

        Returns
        -------
        numpy.ndarray
            Arc-length second derivative of the x-coordinate of point.
        numpy.ndarray
            Arc-length second derivative of the y-coordinate of point.
        """
        xi = self._xi_from_s(s)
        nx, ny = self.normal(xi)
        k = self.k(xi)
        return k*nx, k*ny

    def _xi_from_s(self, s: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the parametric value for arc-length provided.

        Parameters
        ----------
        s : numpy.ndarray
            Arc-length location of point.

        Raises
        ------
        ValueError
            When arc-length provided is larger than airfoil surface length.

        Returns
        -------
        numpy.ndarray
            Parametric value for location provided.
        """
        # TODO: This needs to be optimized so that s_max is stored and does not
        #       need to be calculated each time this method is called. This
        #       will require some abstract method that make the properties
        #       consistent one the shape has changed.
        s_max = self.arc_length(-1.0, 1.0)
        s_a = np.asarray(s)
        if (s_a > s_max).any() or (s_a < 0).any():
            raise ValueError("Invalid arc length provided.")

        def fun(xi: float, s: float) -> float:
            return self.arc_length(-1, xi) - s

        # pylint: disable=cell-var-from-loop
        it = np.nditer([s_a, None])
        with it:
            for ss, xi in it:
                root = root_scalar(lambda xi: fun(xi, ss), bracket=[-1, 1])
                xi[...] = root.root

            return it.operands[1]

    def chord(self) -> float:
        """
        Return the chord length of the airfoil.

        Returns
        -------
        float
            Chord length.
        """
        xle, yle = self.leading_edge()
        xte, yte = self.trailing_edge()
        return np.sqrt((xte-xle)**2+(yte-yle)**2)

    def leading_edge(self) -> Tuple[float, float]:
        """
        Return the location of the leading edge.

        Returns
        -------
        float
            X-coordinate of leading edge.
        float
            Y-coordinate of leading edge.
        """
        return self.xy(0)

    def trailing_edge(self) -> Tuple[float, float]:
        """
        Return the location of the trailing edge.

        Notes
        -----
        Since some airfoil descriptions have gap between the upper and lower
        surface at the trailing edge (such as NACA 4-digit and 5-digit
        airfoils), the point returned is the average of the two trailing
        edge points. If the specific location of the upper or the lower
        surface of the trailing edge is desired, use :py:meth:`xy` passing in
        either -1 (lower) or +1 (upper).

        Returns
        -------
        float
            X-coordinate of leading edge.
        float
            Y-coordinate of leading edge.
        """
        xl, yl = self.xy(-1)
        xu, yu = self.xy(1)
        return 0.5*(xl+xu), 0.5*(yl+yu)

    @abstractmethod
    def camber_value(self, xi: np_type.NDArray) -> np_type.NDArray:
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

    @abstractmethod
    def thickness_value(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Notes
        -----
        There are different ways of expressing the thickness, with the "formal"
        definition being the distance from the camber to the surface in the
        direction normal to the camber. This definition can be challenging to
        follow for some airfoils, so the value returned might be an approximate
        of the formal definition.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """


class OrthogonalAirfoil(Airfoil):
    """Airfoils that can be decomposed to camber and thickness."""

    def __init__(self, camber: Camber, thickness: Thickness) -> None:
        self._camber = camber
        self._thickness = thickness

    @property
    def camber(self) -> Camber:
        """Return the camber function for airfoil."""
        return self._camber

    @property
    def thickness(self) -> Thickness:
        """Return the thickness function for airfoil."""
        return self._thickness

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
        # TODO: Need to check for xi close to zero and return limit value
        raise NotImplementedError("Need to implement this method.")

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
        # TODO: Need to check for xi close to zero and return limit value
        raise NotImplementedError("Need to implement this method.")
