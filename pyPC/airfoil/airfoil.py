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

    There are two different ways that the airfoil coordinates (and their
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

    def t_from_x(self, x:np_type.NDArray, upper: bool) -> np_type.NDArray:
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
        x = np.asarray(x)
        # TODO: the actual minimum for cambered airfoil is NOT at leading edge
        # TODO: this should be part of a bounding box method perhaps
        xmin, _ = self.leading_edge()
        xmin += -0.01
        xmax = max(self.xy(-1)[0], self.xy(1)[0])
        if ((x < xmin).any() or (x > xmax).any()):
            raise ValueError("Invalid x-coordinate provided.")

        def fun(ti: float, x: float) -> float:
            return self.xy(ti)[0] - x

        it = np.nditer([x, None])
        if upper:
            bracket = [0, 1]
        else:
            bracket = [-1, 0]
        with it:
            # pylint: disable=cell-var-from-loop
            for xx, ti in it:
                if xx < 0:
                    root = root_scalar(lambda t: fun(t, xx),
                                       x0=0, x1=abs(xx))
                    ti[...] = root.root
                elif np.abs(fun(bracket[0], xx)) < 1e-8:
                    ti[...] = bracket[0]
                elif np.abs(fun(bracket[1], xx)) < 1e-8:
                    ti[...] = bracket[1]
                else:
                    root = root_scalar(lambda t: fun(t, xx), bracket=bracket)
                    ti[...] = root.root

            return it.operands[1]

    def t_from_s(self, s: np_type.NDArray) -> np_type.NDArray:
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
        s_a = np.asarray(s)
        if (s_a > self.surface_length).any() or (s_a < 0).any():
            raise ValueError("Invalid arc length provided.")

        def fun(t: float, s: float) -> float:
            return self.arc_length(-1, t) - s

        # pylint: disable=cell-var-from-loop
        it = np.nditer([s_a, None])
        with it:
            for ss, ti in it:
                root = root_scalar(lambda t: fun(t, ss), bracket=[-1, 1])
                ti[...] = root.root

            return it.operands[1]

    def dydx(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the slope at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
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
        xt, yt = self.xy_t(t)
        return yt/xt

    def d2ydx2(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the second derivative at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
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
        xt, yt = self.xy_t(t)
        xtt, ytt = self.xy_tt(t)
        return (xt*ytt-yt*xtt)/xt**3

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
        t = self.t_from_s(s)
        return self.xy(t)

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
        t = self.t_from_s(s)
        return self.tangent(t)

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
        t = self.t_from_s(s)
        nx, ny = self.normal(t)
        k = self.k(t)
        return k*nx, k*ny

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
    def camber_location(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                           np_type.NDArray]:
        """
        Return the location of camber at specified parameter location.

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

    @abstractmethod
    def thickness_value(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

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
    """
    Airfoils that can be decomposed to camber and thickness.

    This class represents airfoils that are naturally described by a camber
    line (curve) and a thickness normal to the camber line, both above and
    below the camber line. The parameterization of the camber representation
    and the thickness representation must be based on the same transformation.

    Attributes
    ----------
    camber : Camber
        Curve representing the camber of the airfoil
    thickness : Thickness
        Thickness distribution of the airfoil
    xmin_parameter : float
    xmax_parameter : float
    """

    def __init__(self, camber: Camber, thickness: Thickness) -> None:
        super().__init__()
        self._camber = camber
        self._thickness = thickness
        self._t_xmin = None
        self._t_xmax = None

    @property
    def camber(self) -> Camber:
        """Return the camber function for airfoil."""
        return self._camber

    @property
    def thickness(self) -> Thickness:
        """Return the thickness function for airfoil."""
        return self._thickness

    @property
    def xmin_parameter(self) -> float:
        """Parameter of smallest x-coordinate for airfoil."""
        if self._t_xmin is None:
            max_camber_parameter = self._camber.max_camber_parameter()
            max_camber = self._camber.xy(max_camber_parameter)[1]
            if abs(max_camber) < 1e-7:
                self._t_xmin = 0.0
            else:
                if max_camber > 0:
                    t0 = 1e-7
                    t1 = 0.1
                else:
                    t0 = -1e-7
                    t1 = -0.1
                root = root_scalar(lambda t: self.xy_t(t)[0], bracket=[t0, t1])
                self._t_xmin = root.root

        return self._t_xmin

    @property
    def xmax_parameter(self) -> float:
        """Parameter of largest x-coordinate for airfoil."""
        if self._t_xmax is None:
            if self.x(-1)[0] >= self.x(1)[0]:
                self._t_xmax = -1.0
            else:
                self._t_xmax = 1.0

        return self._t_xmax

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
        tc, sgn = self._convert_t(t)

        u = self._convert_t_to_u(tc)[0]
        xi, eta = self._camber.xy(u)
        nx, ny = self._camber.normal(u)

        v = self._convert_t_to_v(tc)[0]
        delta = sgn*self._thickness.delta(v)

        x = xi + delta*nx
        y = eta + delta*ny
        # return self.xo + self.scale*x, self.yo + self.scale*y
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
        tc, sgn = self._convert_t(t)

        u, u_t = self._convert_t_to_u(tc)[0:2]
        xi, eta = self._camber.xy(u)
        xi_u, eta_u = self._camber.xy_t(u)
        k = self._camber.k(u)
        nx, ny = self._camber.normal(u)
        nx_u = -xi_u*k
        ny_u = -eta_u*k

        v, v_t = self._convert_t_to_v(tc)[0:2]
        delta = sgn*self._thickness.delta(v)
        delta_v = sgn*self._thickness.delta_t(v)

        x_t = sgn*(u_t*xi_u + v_t*delta_v*nx + delta*u_t*nx_u)
        y_t = sgn*(u_t*eta_u + v_t*delta_v*ny + delta*u_t*ny_u)

        # return self.scale*x_t, self.scale*y_t
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
        tc, sgn = self._convert_t(t)

        u, u_t, u_tt = self._convert_t_to_u(tc)
        xi, eta = self._camber.xy(u)
        xi_u, eta_u = self._camber.xy_t(u)
        xi_uu, eta_uu = self._camber.xy_tt(u)
        k = self._camber.k(u)
        k_u = self._camber.k_t(u)
        nx, ny = self._camber.normal(u)
        nx_u = -xi_u*k
        nx_uu = -(xi_uu*k + xi_u*k_u)
        ny_u = -eta_u*k
        ny_uu = -(eta_uu*k + eta_u*k_u)

        v, v_t, v_tt = self._convert_t_to_v(tc)
        delta = sgn*self._thickness.delta(v)
        delta_v = sgn*self._thickness.delta_t(v)
        delta_vv = sgn*self._thickness.delta_tt(v)

        x_tt = (u_tt*xi_u + u_t**2*xi_uu + v_t**2*nx*delta_vv
                + u_t**2*delta*nx_uu + 2*u_t*v_t*delta_v*nx_u
                + v_tt*nx*delta_v + u_tt*delta*nx_u)
        y_tt = (u_tt*eta_u + u_t**2*eta_uu + v_t**2*ny*delta_vv
                + u_t**2*delta*ny_uu + 2*u_t*v_t*delta_v*ny_u
                + v_tt*ny*delta_v + u_tt*delta*ny_u)

        # return self.scale*x_tt, self.scale*y_tt
        return x_tt, y_tt

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

    def camber_location(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                           np_type.NDArray]:
        """
        Return the location of camber at specified parameter location.

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
        return self._camber.xy(self._convert_t_to_u(t)[0])

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
        return self._thickness.delta(self._convert_t_to_v(t)[0])

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the curve.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        camber_joints = [self._convert_u_to_t(j)
                         for j in self._camber.joints()]
        thickness_joints = [self._convert_v_to_t(d)
                            for d in self._thickness.discontinuities()]
        joints = list(set(camber_joints + thickness_joints))
        joints = list(set(joints + [-x for x in joints]))
        joints.sort()
        return joints

    def _airfoil_changed(self) -> None:
        self._t_xmin = None
        self._t_xmax = None
        super()._airfoil_changed()

    def _convert_t_to_u(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                           np_type.NDArray,
                                                           np_type.NDArray]:
        """
        Convert the airfoil parameterization to the camber parameterization.

        Parameters
        ----------
        t : numpy.ndarray
            Airfoil parameterization.

        Returns
        -------
        numpy.ndarray
            Corresponding camber parameterization.
        numpy.ndarray
            Corresponding first derivative of camber parameterization.
        numpy.ndarray
            Corresponding second derivative of camber parameterization.
        """
        return t**2, 2*t, 2*np.ones_like(t)

    def _convert_u_to_t(self, u: np_type.NDArray) -> np_type.NDArray:
        """
        Convert the camber parameterization to the airfoil parameterization.

        Parameters
        ----------
        u : numpy.ndarray
            Camber parameterization.

        Returns
        -------
        Corresponding airfoil parameterization.
        """
        return np.sqrt(u)

    def _convert_t_to_v(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                           np_type.NDArray,
                                                           np_type.NDArray]:
        """
        Convert the airfoil parameterization to the thickness parameterization.

        Parameters
        ----------
        t : numpy.ndarray
            Airfoil parameterization.

        Returns
        -------
        numpy.ndarray
            Corresponding thickness parameterization.
        numpy.ndarray
            Corresponding first derivative of thickness parameterization.
        numpy.ndarray
            Corresponding second derivative of thickness parameterization.
        """
        return t, np.ones_like(t), np.zeros_like(t)

    def _convert_v_to_t(self, u: np_type.NDArray) -> np_type.NDArray:
        """
        Convert the thickness parameterization to the airfoil parameterization.

        Parameters
        ----------
        u : numpy.ndarray
            Camber parameterization.

        Returns
        -------
        Corresponding airfoil parameterization.
        """
        return np.sqrt(u)

    @staticmethod
    def _convert_t(t: np_type.NDArray) -> np_type.NDArray:
        tc = np.asarray(t).copy()
        sgn = np.ones_like(tc)
        sgn[tc < 0] = -1.0
        tc[tc < 0] = -tc[tc < 0]
        return tc, sgn
