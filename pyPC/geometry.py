#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with geometries modeled in panel codes."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as np_type
from scipy.integrate import quadrature
from scipy.optimize import root_scalar


class Geometry(ABC):
    """
    Base class for aerodynamic geometries.

    There are a few different ways that the airfoil coordinates (and their
    derivatives) can be queried.
        - The standard parameterization uses a range from -1 to 0 for the lower
          surface (trailing edge to leading edge) and from 0 to 1 for the
          upper surface (leading edge to trailing edge). This allows a smooth
          parametric variation for the entire airfoil and is consistent with
          how most airfoils are described in equation form.
        - Providing the x-coordinate and whether the upper or lower surface
          is desired. This method allows for the querying of actual points
          on the airfoil, but it might result in a ValueError exception raised
          if there is no surface point at the provided x-coordinate. This
          method is typically more computationally intensive since few airfoils
          are defined explicitly as a function of x-coordinate.
        - Arc-length parameterization use the arc-length along the surface as
          the parameter for specifying the location on the surface. It also
          starts at the lower surface trailing edge and ends at the upper
          surface trailing edge. This method is typically computationally
          expensive because the arc-length is not a typical parameter that is
          easy to calculate for airfoil shapes.
    """

    # Airfoil specific interface
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
        return self.xy_from_xi(0)

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
        xl, yl = self.xy_from_xi(-1)
        xu, yu = self.xy_from_xi(1)
        return 0.5*(xl+xu), 0.5*(yl+yu)

    def tangent(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the unit tangent at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Unit tangent at point.
        """
        sx, sy = self.xy_p(xi)
        temp = np.sqrt(sx**2 + sy**2)
        sx = sx/temp
        sy = sy/temp
        return sx, sy

    def normal(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the unit normal at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Unit normal at point.
        """
        sx, sy = self.tangent(xi)
        nx = -sy
        ny = sx
        return nx, ny

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

    @abstractmethod
    def thickness(self, xi: np_type.NDArray) -> np_type.NDArray:
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

    #
    # Parameteric Interface
    #
    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def xy_pp(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at parameter location.

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
        _, y = self.xy_from_xi(xi)
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
            xr, _ = self.xy_from_xi(xi)
            return xr - x

        it = np.nditer([x_a, None])
        if upper:
            bracket = [0, 1]
        else:
            bracket = [-1, 0]
        with it:
            for x, xi in it:
                root = root_scalar(lambda xi: fun(xi, x), bracket=bracket)
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
        return self.xy_from_xi(xi)

    def xy_dot(self, s: np_type.NDArray) -> Tuple[np_type.NDArray,
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

    def xy_ddot(self, s: np_type.NDArray) -> Tuple[np_type.NDArray,
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
        s_max = self.arc_length(-1.0, 1.0)
        s_a = np.asarray(s)
        if (s_a > s_max).any() or (s_a < 0).any():
            raise ValueError("Invalid arc length provided.")

        def fun(xi: float, s: float) -> float:
            return self.arc_length(-1, xi) - s

        it = np.nditer([s_a, None])
        with it:
            for s, xi in it:
                root = root_scalar(lambda xi: fun(xi, s), bracket=[-1, 1])
                xi[...] = root.root

            return it.operands[1]


class Cylinder(Geometry):
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
