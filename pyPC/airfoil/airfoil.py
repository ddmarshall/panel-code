#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils and their analysis."""

from abc import abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type

from scipy.integrate import quadrature
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

    def __init__(self) -> None:
        super().__init__()
        self._s_max = None

    @property
    def surface_length(self) -> float:
        """Return length of entire airfoil surface."""
        if self._s_max is None:
            self._s_max = self.arc_length(-1.0, 1.0)

        return self._s_max

    def xi_from_x(self, x:np_type.NDArray, upper: bool) -> np_type.NDArray:
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
        # TODO: the actual minimum for cambered airfoil is NOT at leading edge
        # TODO: this should be part of a bounding box method perhaps
        xmin, _ = self.leading_edge()
        xmin += -0.01
        xmax = max(self.xy(-1)[0], self.xy(1)[0])
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
                if xx < 0:
                    root = root_scalar(lambda xi: fun(xi, xx),
                                       x0=0, x1=abs(xx))
                    xi[...] = root.root
                elif np.abs(fun(bracket[0], xx)) < 1e-8:
                    xi[...] = bracket[0]
                elif np.abs(fun(bracket[1], xx)) < 1e-8:
                    xi[...] = bracket[1]
                else:
                    root = root_scalar(lambda xi: fun(xi, xx), bracket=bracket)
                    xi[...] = root.root

            return it.operands[1]

    def dydx(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the slope at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Slope of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xp, yp = self.xy_p(xi)
        return yp/xp

    def d2ydx2(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the second derivative at parameter location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Second derivative of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        xp, yp = self.xy_p(xi)
        xpp, ypp = self.xy_pp(xi)
        return (xp*ypp-yp*xpp)/xp**3

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
        s_max = self.surface_length()

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

    def _airfoil_changed(self) -> None:
        """
        Notify airfoil that shape has changed.

        This needs to be called by child classes when the airfoil geometry
        has changed so any cached values can be invalidated.
        """
        self._s_max = None

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
        super().__init__()
        self._camber = camber
        self._thickness = thickness
        self._xi_xmin = None
        self._xi_xmax = None

    def _airfoil_changed(self) -> None:
        self._xi_xmin = None
        self._xi_xmax = None
        super()._airfoil_changed()

    @property
    def camber(self) -> Camber:
        """Return the camber function for airfoil."""
        return self._camber

    @property
    def thickness(self) -> Thickness:
        """Return the thickness function for airfoil."""
        return self._thickness

    @property
    def xi_xmin(self) -> float:
        """Parameter coordinate of smallest x-coordinate for airfoil."""
        if self._xi_xmin is None:
            max_camber = self._camber.max_camber()[1]
            if abs(max_camber) < 1e-7:
                self._xi_xmin = 0.0
            else:
                if max_camber > 0:
                    xi0 = 1e-7
                    xi1 = 0.1
                else:
                    xi0 = -1e-7
                    xi1 = -0.1
                root = root_scalar(lambda xi: self.xy_p(xi)[0],
                                   bracket=[xi0, xi1])
                self._xi_xmin = root.root

        return self._xi_xmin

    @property
    def xi_xmax(self) -> float:
        """Parameter coordinate of largest x-coordinate for airfoil."""
        if self._xi_xmax is None:
            if self.x(-1)[0] >= self.x(1)[0]:
                self._xi_xmax = -1.0
            else:
                self._xi_xmax = 1.0

        return self._xi_xmax

    def xy(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
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
        xic, sgn = self._convert_xi(xi)

        delta_t = self._thickness.y(xic)
        yc = self._camber.y(xic)
        yc_p = self._camber.y_p(xic)
        denom = np.sqrt(1+yc_p**2)
        x = xic - sgn*delta_t*yc_p/denom
        y = yc + sgn*delta_t/denom
        # return self.xo + self.scale*x, self.yo + self.scale*y
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
        xic, sgn = self._convert_xi(xi)

        delta_t = self._thickness.y(xic)
        delta_tp = self._thickness.y_p(xic)
        yc_p = self._camber.y_p(xic)
        yc_pp = self._camber.y_pp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_p = 1.0 - sgn/denom*(delta_tp*yc_p + delta_t*yc_pp/denom**2)
        y_p = yc_p + sgn/denom*(delta_tp - delta_t*yc_p*yc_pp/denom**2)
        # return sgn*self.scale*x_p, sgn*self.scale*y_p
        return sgn*x_p, sgn*y_p

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
        xic, sgn = self._convert_xi(xi)

        delta_t = self._thickness.y(xic)
        delta_tp = self._thickness.y_p(xic)
        delta_tpp = self._thickness.y_pp(xic)
        yc_p = self._camber.y_p(xic)
        yc_pp = self._camber.y_pp(xic)
        yc_ppp = self._camber.y_ppp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_pp = -sgn/denom*(delta_tpp*yc_p + (2*delta_tp*yc_pp
                                             + delta_t*yc_ppp)/denom**2
                           - 3*delta_t*yc_p*yc_pp**2/denom**4)
        y_pp = yc_pp + sgn/denom*(delta_tpp - (yc_p*(2*delta_tp*yc_pp
                                                     + delta_t*yc_ppp)
                                               - 2*delta_t*yc_pp**2)/denom**2
                                  - 3*delta_t*yc_pp**2/denom**4)
        # return self.scale*x_pp, self.scale*y_pp
        return x_pp, y_pp

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
        eps = 1e-7
        xi = np.asarray(xi, dtype=np.float64)

        # np.piecewise does not work when returning tuples so need to manually
        # do the iteration to find the leading edge cases
        it = np.nditer([xi, None, None])

        with it:
            for xir, sxr, syr in it:
                if np.abs(xir) < eps:
                    tmp0 = -self.camber.y_p(xir)
                    tmp1 = np.sqrt(tmp0**2+1)
                    sxr[...] = tmp0/tmp1
                    syr[...] = 1/tmp1
                else:
                    sxr[...], syr[...] = super().tangent(xir)

            return it.operands[1], it.operands[2]

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
        eps = 1e-7
        xi = np.asarray(xi, dtype=np.float64)

        return np.piecewise(xi, [np.abs(xi) < eps, np.abs(xi) >= eps],
                            [lambda xi: (self.thickness.k(xi)
                                         * np.sqrt(1.0
                                                   + self.camber.y_p(xi)**2)),
                             lambda xi: super(self.__class__, self).k(xi)])

    # def arc_length(self, xi_s: float,
    #                xi_e: np_type.NDArray) -> np_type.NDArray:
    #     """
    #     Calculate the arc-length distance between two points on surface.

    #     Parameters
    #     ----------
    #     xi_s : float
    #         Start point of distance calculation.
    #     xi_e : numpy.ndarray
    #         End point of distance calculation.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Distance from start point to end point.
    #     """
    #     xi_e = np.asarray(xi_e, dtype=np.float64)

    #     # NOTE: This prototype works for xi>=0 as a parameterization
    #     #       of xi=t**2. How can this be integrated into existing
    #     #       architecture? Current airfoil code breaks near leading edge
    #     #       because of the sqrt(x) term in thickness.
    #     def fun(t):
    #         xi = t**2
    #         yc_p = self._camber.y_p(xi)
    #         alpha = np.sqrt(1+yc_p**2)
    #         beta = yc_p*alpha
    #         delta = self._thickness.y(xi)
    #         k_c = self._camber.k(xi)
    #         a = (self._thickness._tmax/0.20)*self._thickness.a
    #         ddeltadt = a[0]+2*t*(a[1]+2*a[2]*xi+3*a[3]*xi**2+4*a[4]*xi**3)
    #         dxdt = 2*t-(beta*ddeltadt+2*t*delta*k_c)
    #         dydt = 2*t*yc_p+(alpha*ddeltadt-2*t*delta*yc_p*k_c)
    #         return np.sqrt(dxdt**2+dydt**2)
    #     s_test = quadrature(fun, np.sqrt(xi_min), np.sqrt(xi_max))
    #     def fun(xi):
    #         xp, yp = self.xy_p(xi)
    #         return np.sqrt(xp**2+yp**2)

    #     xi_ea = np.asarray(xi_e)
    #     it = np.nditer([xi_ea, None])
    #     with it:
    #         for xi, alen in it:
    #             alen[...], _ = quadrature(fun, xi_s, xi)

    #         return it.operands[1]

    def dydx(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the slope at parameteric location.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Slope of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        eps = 1e-7
        xi = np.asarray(xi, dtype=np.float64)

        def fun(xi: float) -> float:
            if self.camber.max_camber()[1] == 0:
                return -np.inf*np.ones_like(xi)
            else:
                return -1/self.camber.y_p(np.abs(xi))

        return np.piecewise(xi, [np.abs(xi) < eps, np.abs(xi) >= eps],
                            [lambda xi: fun(xi),
                             lambda xi: super(self.__class__, self).dydx(xi,)])

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
        return self._camber.y(xi)

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
        return self._thickness.y(xi)

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the curve.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        joints = list(set(self._camber.joints() + self._thickness.joints()))
        joints = list(set(joints + [-x for x in joints]))
        joints.sort()
        return joints

    @staticmethod
    def _convert_xi(xi: np_type.NDArray) -> np_type.NDArray:
        xic = np.asarray(xi).copy()
        sgn = np.ones_like(xic)
        sgn[xic < 0] = -1.0
        xic[xic < 0] = -xic[xic < 0]
        return xic, sgn
