#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils modeled in panel codes."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.geometry import Geometry
from pyPC.camber import Naca4DigitCamber


class Naca4DigitThicknessBase:
    """
    Base class for the NACA 4-digit airfoil thickness.

    Attributes
    ----------
    thickness : float
        Maximum thickness per chord length.
    a : numpy.ndarray
        Coefficients for equation.
    """

    def __init__(self, thickness: float, a: np_type.NDArray) -> None:
        self._t = thickness
        self._a = a

    @property
    def thickness(self) -> float:
        """Maximum thickness."""
        return self._t

    @thickness.setter
    def thickness(self, thickness: float) -> None:
        self._t = thickness

    @property
    def a(self) -> float:
        """Equation coefficients."""
        return self._a

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        return (self.thickness/0.20)*(self.a[0]*np.sqrt(xi)
                                      + xi*(self.a[1]
                                            + xi*(self.a[2]
                                                  + xi*(self.a[3]
                                                        + xi*self.a[4]))))

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified point.
        """
        return (self.thickness/0.20)*(0.5*self.a[0]/np.sqrt(xi)
                                      + (self.a[1]
                                         + xi*(2*self.a[2]
                                               + xi*(3*self.a[3]
                                                     + 4*xi*self.a[4]))))

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified point.
        """
        return (self.thickness/0.20)*(-0.25*self.a[0]/(xi*np.sqrt(xi))
                                      + 2*(self.a[2]
                                           + 3*xi*(self.a[3]
                                                   + 2*xi*self.a[4])))

    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """
        return -2/((self.thickness/0.20)*self.a[0])**2


class Naca4DigitThicknessClassic(Naca4DigitThicknessBase):
    """Classic NACA 4-digit airfoil thickness."""

    def __init__(self, thickness: float) -> None:
        super().__init__(thickness=thickness, a=np.array([0.29690, -0.12600,
                                                          -0.35160, 0.28430,
                                                          -0.10150]))


class Naca4DigitThicknessEnhanced(Naca4DigitThicknessBase):
    """
    Enhanced NACA 4-digit airfoil thickness relation.

    This class extends the standard thickness distribution relations by
    - Solving for the coefficients based on the original constraints used to
      describe the thickness even with non-integer parameters,
    - Allowing the trailing edge to be closed instead of the default
      thickness, and
    - Allowing the setting of the leading edge radius value instead of the
      approximate way it is set in the original formulations.

    Attributes
    ----------
    closed_te : bool
        True if the thickness should be zero at the trailing edge
    le_radius : bool
        True if the leading edge radius should be same as classic airfoil or
        False if original method of setting the thickness near the leading
        edge should be used.
    """

    def __init__(self, thickness: float, closed_te: bool,
                 le_radius: bool) -> None:
        self.reset(closed_te, le_radius)
        super().__init__(thickness=thickness, a=self._a)

    def is_trailing_edge_closed(self) -> bool:
        """
        Return state of trailing edge condition.

        Returns
        -------
        bool
            True if the trailing edge is closed.
        """
        return self._closed_te

    def using_leading_edge_radius(self) -> bool:
        """
        Return state of leading edge treatment.

        Returns
        -------
        bool
            True if the leading edge radius is set otherwise the original
            approximate leading edge condition used.
        """
        return self._le_radius

    def reset(self, closed_te: bool, le_radius: bool) -> None:
        """
        Reset the flags for the leading edge and trailing edge shape.

        Parameters
        ----------
        closed_te : bool
            True if the thickness should be zero at the trailing edge
        le_radius : bool
            True if the leading edge radius should be same as classic airfoil
            or False if original method of setting the thickness near the
            leading edge should be used.
        """
        self._closed_te = closed_te
        self._le_radius = le_radius

        # solve for new values of the coefficients
        B = np.zeros([5,5])
        r = np.zeros([5,1])

        # first row is leading edge condition
        i = 0
        if le_radius:
            B[i, :] = [1, 0, 0, 0, 0]
            r[i] = 0.29690
        else:
            xi_1c = 0.1
            t_1c = 0.078
            B[i, :] = [np.sqrt(xi_1c), xi_1c, xi_1c**2, xi_1c**3, xi_1c**4]
            r[i] = t_1c

        # second row is the maximum thickness at 0.3c
        i = 1
        xi_max = 0.3
        B[i, :] = [0.5/np.sqrt(xi_max), 1, 2*xi_max, 3*xi_max**2, 4*xi_max**3]
        r[i] = 0.0

        # third row is the maximum thickness of 0.2c
        i = 2
        t_max = 0.1
        B[i, :] = [np.sqrt(xi_max), xi_max, xi_max**2, xi_max**3, xi_max**4]
        r[i] = t_max

        # fourth row is trailing edge slope
        i = 3
        te_slope = -0.234
        B[i, :] = [0.5, 1, 2, 3, 4]
        r[i] = te_slope

        # fith row is trailing edge thickness
        i = 4
        if closed_te:
            t_te = 0
        else:
            t_te = 0.002
        B[i, :] = [1, 1, 1, 1, 1]
        r[i] = t_te

        self._a = np.linalg.solve(B, r).transpose()[0]


class Naca4DigitModifiedThicknessBase:
    """
    Base class for the NACA modified 4-digit airfoil thickness.

    Attributes
    ----------
    thickness : float
        Maximum thickness per chord length.
    xi_m : float
        Location of end of fore section and start of aft section.
    a : numpy.ndarray
        Coefficients for fore equation.
    d : numpy.ndarray
        Coefficients for aft equation.
    """

    def __init__(self, thickness: float, xi_m: float, a: np_type.NDArray,
                 d: np_type.NDArray) -> None:
        self._t = thickness
        self._xi_m = xi_m
        self._a = a
        self._d = d

    @property
    def thickness(self) -> float:
        """Maximum thickness."""
        return self._t

    @thickness.setter
    def thickness(self, thickness: float) -> None:
        self._t = thickness

    @property
    def xi_m(self) -> float:
        """Location where fore and aft equations meet."""
        return self._xi_m

    @property
    def a(self) -> float:
        """Fore equation coefficients."""
        return self._a

    @property
    def d(self) -> float:
        """Aft equation coefficients."""
        return self._d

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return self.a[0]*np.sqrt(xi) + xi*(self.a[1]
                                               + xi*(self.a[2]
                                                     + (xi*self.a[3])))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return self.d[0] + (1-xi)*(self.d[1]
                                       + (1-xi)*(self.d[2]
                                                 + ((1-xi)*self.d[3])))

        return self.thickness*np.piecewise(xi, [xi <= self.xi_m,
                                                xi > self.xi_m],
                                           [lambda xi: fore(xi),
                                            lambda xi: aft(xi)])

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return 0.5*self.a[0]/np.sqrt(xi) + (self.a[1]
                                                + xi*(2*self.a[2]
                                                      + (3*xi*self.a[3])))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -self.d[1] + (1-xi)*(-2*self.d[2] + (-3*(1-xi)*self.d[3]))

        return self.thickness*np.piecewise(xi, [xi <= self.xi_m,
                                                xi > self.xi_m],
                                           [lambda xi: fore(xi),
                                            lambda xi: aft(xi)])

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (-0.25*self.a[0]/(xi*np.sqrt(xi)) + 2*self.a[2]
                    + 6*xi*self.a[3])

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*self.d[2] + 6*(1-xi)*self.d[3]

        return self.thickness*np.piecewise(xi, [xi <= self.xi_m,
                                                xi > self.xi_m],
                                           [lambda xi: fore(xi),
                                            lambda xi: aft(xi)])

    def _reset(self, le_radius: float, xi_m: float, eta: float) -> None:
        tau = self._tau(xi_m)
        k_m = self._k_m(eta=eta, tau=tau, xi_m=xi_m)
        self._xi_m = xi_m
        self._d = self._calc_d_terms(eta=eta, tau=tau, xi_m=xi_m)
        self._a = self._calc_a_terms(Iterm=le_radius, k_m=k_m, xi_m=xi_m)

    @staticmethod
    def _tau(xi_m: float) -> float:
        p = [1.0310900853, -2.7171508529, 4.8594083156]
        q = [1.0, -1.8252487562, 1.1771499645]
        return ((p[0] + p[1]*xi_m + p[2]*xi_m**2)
                / (q[0] + q[1]*xi_m + q[2]*xi_m**2))

    @staticmethod
    def _Q(Iterm: float) -> float:
        a_tilda2 = 0.08814961
        if Iterm < 9:
            return 25*a_tilda2/72
        else:
            return 25*a_tilda2/54

    @staticmethod
    def _calc_d_terms(eta: float, tau: float, xi_m: float) -> np_type.NDArray:
        d = np.zeros(4)
        d[0] = 0.5*eta
        d[1] = tau
        d[2] = (2*tau*(xi_m-1)-1.5*(eta-1))/(xi_m-1)**2
        d[3] = (tau*(xi_m-1)-(eta-1))/(xi_m-1)**3
        return d

    @staticmethod
    def _k_m(eta: float, tau: float, xi_m: float) -> float:
        return (3*(eta-1)-2*tau*(xi_m-1))/(xi_m-1)**2

    @staticmethod
    def _calc_a_terms(Iterm: float, k_m: float,
                      xi_m: float) -> np_type.NDArray:
        Q = Naca4DigitModifiedThicknessBase._Q(Iterm)
        a = np.zeros(4)
        sqrt_term = np.sqrt(2*Q)
        sqrt_term2 = np.sqrt(2*Q*xi_m)
        a[0] = Iterm*sqrt_term
        a[1] = 0.5*xi_m*(k_m + (3-3.75*Iterm*sqrt_term2)/xi_m**2)
        a[2] = -(k_m + (1.5-1.25*Iterm*sqrt_term2)/xi_m**2)
        a[3] = 0.5/xi_m*(k_m + (1-0.75*Iterm*sqrt_term2)/xi_m**2)
        return a

    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """
        return -2/(self.thickness*self.a[0])**2


class Naca4DigitModifiedThicknessClassic(Naca4DigitModifiedThicknessBase):
    """
    Classic NACA modified 4-digit airfoil thickness.

    Attributes
    ----------
    max_thickness_loc : int
        Location of maximum thickness in tenth of chord.
    le_radius : int
        Index to specify the radius of the leading edge.
    """

    def __init__(self, thickness: float, le_radius: int,
                 max_t_loc: int) -> None:
        super().__init__(thickness=thickness, xi_m=0.4, a=np.zeros(4),
                         d=np.zeros(4))
        self.reset(le_radius, max_t_loc)

    @property
    def max_thickness_loc(self) -> int:
        """Parameter specifying the location of maximum thickness."""
        return self._max_t_loc

    @property
    def le_radius(self) -> int:
        """Parameter specifying the leading edge radius."""
        return self._le_radius

    def reset(self, le_radius: int, max_t_loc: int) -> None:
        """
        Reset the thickness parameters to new values.

        Parameters
        ----------
        le_radius : int
            Indicator of leading edge radius.
        max_t_loc : int
            Indicator of location of maximum thickness.
        """
        self._le_radius = le_radius
        self._max_t_loc = max_t_loc
        self._reset(le_radius=le_radius, xi_m=max_t_loc/10.0, eta=0.02)


class Naca4DigitModifiedThicknessEnhanced(Naca4DigitModifiedThicknessBase):
    """
    Enhanced NACA modified 4-digit airfoil thickness relation.

    This class extends the standard modified thickness distribution relations
    by
    - Solving for the coefficients based on the original constraints used to
      describe the thickness for non-integer values of parameters
    - Allowing the trailing edge to be closed instead of the default thickness

    Attributes
    ----------
    closed_te : bool
        True if the thickness should be zero at the trailing edge
    max_thickness_loc : float
        Location of maximum thickness per chord.
    le_radius : float
        Parameter to specify the radius of the leading edge.
    """

    def __init__(self, thickness: float,  le_radius: float, max_t_loc: float,
                 closed_te: bool) -> None:
        super().__init__(thickness=thickness, xi_m=0.4, a=np.zeros(4),
                         d=np.zeros(4))
        self.reset(le_radius, max_t_loc, closed_te)

    @property
    def max_thickness_loc(self) -> float:
        """Location of maximum thickness per chord."""
        return self._xi_m

    @property
    def le_radius(self) -> float:
        """Parameter specifying the leading edge radius."""
        return self._le_radius

    @property
    def closed_te(self) -> bool:
        """Flag whether trailing edge should be closed."""
        return self._closed_te

    @closed_te.setter
    def closed_te(self, closed_te: bool) -> None:
        self.reset(le_radius=self.le_radius, max_t_loc=self.max_thickness_loc,
                   closed_te=closed_te)

    def reset(self, le_radius: float, max_t_loc: float,
              closed_te: bool) -> None:
        """
        Reset the thickness parameters to new values.

        Parameters
        ----------
        le_radius : int
            Indicator of leading edge radius.
        max_t_loc : int
            Indicator of location of maximum thickness.
        """
        self._le_radius = le_radius
        self._closed_te = closed_te
        if closed_te:
            eta = 0
        else:
            eta = 0.02
        self._reset(le_radius=le_radius, xi_m=max_t_loc, eta=eta)


class Naca4DigitAirfoilClassic(Geometry):
    """
    Classic NACA 4-digit airfoil.

    Attributes
    ----------
    max_camber : int
        Maximum camber parameter. 100 time the actual maximum camber per chord.
    max_camber_location : int
        Location of maximum camber parameter. 10 times the actual location per
        chord.
    max_thickness : int
        Maximum thickness parameter. 100 times the actual thickness per chord.
    scale : float
        Amount to scale the final airfoil.
    """

    def __init__(self, max_camber: int, max_camber_location: int,
                 max_thickness: int, scale: float) -> None:
        self._scale = scale
        self._max_camber = max_camber
        self._max_camber_loc = max_camber_location
        self._max_thickness = max_thickness
        self._delta_t = Naca4DigitThicknessClassic(max_thickness/100.0)
        self._yc = Naca4DigitCamber(m=max_camber/100.0,
                                    p=max_camber_location/10.0)

    @property
    def m(self) -> int:
        """Maximum camber parameter."""
        return self.max_camber

    @m.setter
    def m(self, m: int) -> None:
        self.max_camber = m

    @property
    def p(self) -> int:
        """Location of maximum camber parameter."""
        return self.max_camber_location

    @p.setter
    def p(self, p: int) -> None:
        self.max_camber_location = p

    @property
    def t(self) -> int:
        """Maximum thickness parameter."""
        return self.max_thickness

    @t.setter
    def t(self, t: int) -> None:
        self.max_thickness = t

    @property
    def max_camber(self) -> int:
        """Maximum camber parameter."""
        return self._max_camber

    @max_camber.setter
    def max_camber(self, max_camber: int) -> None:
        self._max_camber = max_camber
        self._yc.m = max_camber/100.0

    @property
    def max_camber_location(self) -> int:
        """Location of maximum camber parameter."""
        return self._max_camber_loc

    @max_camber_location.setter
    def max_camber_location(self, max_camber_loc: int) -> None:
        self._max_camber = max_camber_loc
        self._yc.p = max_camber_loc/10.0

    @property
    def max_thickness(self) -> int:
        """Maximum thickness parameter."""
        return self._max_thickness

    @max_thickness.setter
    def max_thickness(self, max_thickness: int) -> None:
        if max_thickness <= 0:
            raise ValueError("Maximum thickness must be non-zero and "
                             "positive.")
        self._delta_t.thickness(thickness=max_thickness/100.0)
        self._max_Dthickness = max_thickness

    @property
    def scale(self) -> float:
        """Scale factor for airfiol."""
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @staticmethod
    def _convert_xi(xi: np_type.NDArray) -> np_type.NDArray:
        xic = np.asarray(xi).copy()
        sgn = np.ones_like(xic)
        sgn[xic < 0] = -1.0
        xic[xic < 0] = -xic[xic < 0]
        return xic, sgn

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
        xic, _ = self._convert_xi(xi)
        return self.scale*self._yc.y(xic)

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
        xic, _ = self._convert_xi(xi)
        return self.scale*self._delta_t.y(xic)

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
        xic, sgn = self._convert_xi(xi)

        delta_t = self._delta_t.y(xic)
        yc = self._yc.y(xic)
        yc_p = self._yc.y_p(xic)
        denom = np.sqrt(1+yc_p**2)
        x = xic - sgn*delta_t*yc_p/denom
        y = yc + sgn*delta_t/denom
        return self.scale*x, self.scale*y

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

        delta_t = self._delta_t.y(xic)
        delta_tp = self._delta_t.y_p(xic)
        yc_p = self._yc.y_p(xic)
        yc_pp = self._yc.y_pp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_p = 1.0 - sgn/denom*(delta_tp*yc_p + delta_t*yc_pp/denom**2)
        y_p = yc_p + sgn/denom*(delta_tp - delta_t*yc_p*yc_pp/denom**2)
        return sgn*self.scale*x_p, sgn*self.scale*y_p

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

        delta_t = self._delta_t.y(xic)
        delta_tp = self._delta_t.y_p(xic)
        delta_tpp = self._delta_t.y_pp(xic)
        yc_p = self._yc.y_p(xic)
        yc_pp = self._yc.y_pp(xic)
        yc_ppp = self._yc.y_ppp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_pp = -sgn/denom*(delta_tpp*yc_p + (2*delta_tp*yc_pp
                                             + delta_t*yc_ppp)/denom**2
                           - 3*delta_t*yc_p*yc_pp**2/denom**4)
        y_pp = yc_pp + sgn/denom*(delta_tpp - (yc_p*(2*delta_tp*yc_pp
                                                     + delta_t*yc_ppp)
                                               - 2*delta_t*yc_pp**2)/denom**2
                                  - 3*delta_t*yc_pp**2/denom**4)
        return self.scale*x_pp, self.scale*y_pp
