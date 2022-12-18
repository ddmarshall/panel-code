#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils modeled in panel codes."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type
from scipy.optimize import root_scalar

from pyPC.geometry import Geometry


class Naca4DigitCamber:
    """
    Camber for the NACA 4-digit airfoils.

    Attributes
    ----------
    m : float
        Maximum amount of camber per chord length.
    p : float
        Relative chord location of maximum camber.
    """

    def __init__(self, m: float, p: float) -> None:
        self._m = m
        self._p = p

    @property
    def m(self) -> float:
        """Maximum amount of camber."""
        return self._m

    @m.setter
    def m(self, m: float) -> None:
        self._m = m

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

    @p.setter
    def p(self, p: float) -> None:
        self._p = p

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/self.p**2)*(2*self.p*xi - xi**2)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/(1-self.p)**2)*(1 + 2*self.p*(xi - 1) - xi**2)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/self.p**2)*(self.p - xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/(1-self.p)**2)*(self.p - xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/self.p**2)*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/(1-self.p)**2)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_ppp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])


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
      describe the thickness
    - Allowing the trailing edge to be closed instead of the default thickness
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


class Naca5DigitCamberBase:
    """
    Base class for the NACA 5-digit airfoil camber.

    Attributes
    ----------
    m : float
        Camber transition relative chord location.
    k1 : float
        Scale factor for camber.
    """

    def __init__(self, m: float, k1: float) -> None:
        self._m = m
        self._k1 = k1

    @property
    def m(self) -> float:
        """Camber transition relative chord location."""
        return self._m

    @property
    def k1(self) -> float:
        """Scale factor for camber."""
        return self._k1

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            m = self.m
            return (self.k1/6)*(xi**3 - 3*m*xi**2 + m**2*(3-m)*xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1*self.m**3/6)*(1 - xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        m = self.m

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1/6)*(3*xi**2 - 6*m*xi + m**2*(3-m))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -(self.k1*m**3/6)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= m, xi > m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1)*(xi - self.m)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_ppp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return self.k1*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])


class Naca5DigitCamberClassic(Naca5DigitCamberBase):
    """
    Camber for the classic NACA 5-digit airfoils.

    Attributes
    ----------
    camber_location : int
        Relative chord location of maximum camber. The only valid values are 1,
        2, 3, 4, and 5.
    """

    def __init__(self, camber_loc: int) -> None:
        self.camber_location = camber_loc
        super().__init__(m=self.m, k1=self.k1)

    @property
    def camber_location(self) -> int:
        """Relative chord location of maximum camber."""
        return self._camber_location

    @camber_location.setter
    def camber_location(self, camber_loc: int) -> None:
        if camber_loc == 1:
            self._m = 0.0580
            self._k1 = 361.4
        elif camber_loc == 2:
            self._m = 0.1260
            self._k1 = 51.64
        elif camber_loc == 3:
            self._m = 0.2025
            self._k1 = 15.957
        elif camber_loc == 4:
            self._m = 0.2900
            self._k1 = 6.643
        elif camber_loc == 5:
            self._m = 0.3910
            self._k1 = 3.230
        else:
            raise ValueError("Invalid NACA 5-Digit max. camber location: "
                             f"{camber_loc}.")

        self._camber_location = camber_loc


class Naca5DigitCamberEnhanced(Naca5DigitCamberBase):
    """
    Camber for the regular NACA 5-digit airfoils.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber.
    Cl_ideal : float
        Ideal lift coefficient.
    """

    def __init__(self, p: float, Cl_ideal: float) -> None:
        self._m = 0.0
        self._k1 = 0.0
        self._Cl_ideal = 1  # Need to bootstrap initialization
        self.p = p
        self.Cl_ideal = Cl_ideal

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

    @p.setter
    def p(self, p) -> float:
        root = root_scalar(lambda x: self._camber_slope(p, -6, x),
                           bracket=[p, 2*p])
        self._m = root.root
        self._p = p
        # This sets the new k1 since m changed
        self.Cl_ideal = self.Cl_ideal

    @property
    def Cl_ideal(self) -> float:
        """Ideal lift coefficient."""
        return self._Cl_ideal

    @Cl_ideal.setter
    def Cl_ideal(self, Cl_ideal: float) -> float:
        self._Cl_ideal = Cl_ideal
        self._k1 = self._determine_k1(Cl_ideal, self._m)

    @staticmethod
    def _camber_slope(xi: float, k1: float, m: float) -> float:
        return k1/6*(3*xi**2 - 6*m*xi + m**2*(3-m))

    @staticmethod
    def _determine_k1(Cl_ideal: float, m: float) -> float:
        return 6*Cl_ideal/(-3/2*(1-2*m)*np.arccos(1-2*m)
                           + (4*m**2-4*m+3)*np.sqrt(m*(1-m)))


class Naca5DigitCamberReflexedBase:
    """
    Base class for the NACA 5-digit reflexed airfoil camber.

    Attributes
    ----------
    m : float
        Camber transition relative chord location.
    k1 : float
        First scale factor for camber.
    k2 : float
        Second scale factor for camber.
    """

    def __init__(self, m: float, k1: float, k2: float) -> None:
        self._m = m
        self._k1 = k1
        self._k2 = k2

    @property
    def m(self) -> float:
        """Camber transition relative chord location."""
        return self._m

    @property
    def k1(self) -> float:
        """First scale factor for camber."""
        return self._k1

    @property
    def k2(self) -> float:
        """Second scale factor for camber."""
        return self._k2

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        m = self.m
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*((xi-m)**3 - k2ok1*(1-m)**3*xi + m**3*(1-xi))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(k2ok1*(xi-m)**3 - k2ok1*(1-m)**3*xi + m**3*(1-xi))

        return np.piecewise(xi, [xi <= m, xi > m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        m = self.m
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*(xi-m)**2 - k2ok1*(1-m)**3 - m**3)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*k2ok1*(xi-m)**2 - k2ok1*(1-m)**3 - m**3)

        return np.piecewise(xi, [xi <= m, xi > m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1)*(xi - self.m)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k2)*(xi - self.m)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_ppp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return self.k1*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return self.k2*np.ones_like(xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])


class Naca5DigitCamberReflexedClassic(Naca5DigitCamberReflexedBase):
    """
    Camber for the classic NACA 5-digit reflexed airfoils.

    Attributes
    ----------
    camber_location : int
        Relative chord location of maximum camber. The only valid values are 1,
        2, 3, 4, and 5.
    """

    def __init__(self, camber_loc: int) -> None:
        self.camber_location = camber_loc
        super().__init__(m=self.m, k1=self.k1, k2=self.k2)

    @property
    def camber_location(self) -> int:
        """Relative chord location of maximum camber."""
        return self._camber_location

    @camber_location.setter
    def camber_location(self, camber_loc: int) -> None:
        if camber_loc == 2:
            self._m = 0.1300
            self._k1 = 51.99
            self._k2 = 0.03972036
        elif camber_loc == 3:
            self._m = 0.2170
            self._k1 = 15.793
            self._k2 = 0.10691861
        elif camber_loc == 4:
            self._m = 0.3180
            self._k1 = 6.520
            self._k2 = 0.197556
        elif camber_loc == 5:
            self._m = 0.4410
            self._k1 = 3.191
            self._k2 = 0.4323805
        else:
            raise ValueError("Invalid NACA 5-Digit reflexed max. camber "
                             f"location: {camber_loc}.")

        self._camber_location = camber_loc


class Naca5DigitCamberReflexedEnhanced(Naca5DigitCamberReflexedBase):
    """
    Camber for the regular NACA 5-digit reflexed airfoils.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber.
    Cl_ideal : float
        Ideal lift coefficient.
    """

    def __init__(self, p: float, Cl_ideal: float) -> None:
        self._m = 0.0
        self._k1 = 0.0
        self._k2 = 1.0
        self._Cl_ideal = 1  # Need to bootstrap initialization
        self.p = p
        self.Cl_ideal = Cl_ideal

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

    @p.setter
    def p(self, p) -> float:
        def m_fun(m: float) -> float:
            k1 = 1.0
            k2ok1 = self._k2ok1(m, p)
            Cl_id = self._Cl_id(m, k1, k2ok1)
            return self._Cmc4(m, k1, k2ok1, Cl_id)

        root = root_scalar(m_fun, bracket=[p, 3*p])
        self._m = root.root
        self._p = p
        # This sets the new k1 and k2 since m and p changed
        self.Cl_ideal = self.Cl_ideal

    @property
    def Cl_ideal(self) -> float:
        """Ideal lift coefficient."""
        return self._Cl_ideal

    @Cl_ideal.setter
    def Cl_ideal(self, Cl_ideal: float) -> float:
        k2ok1 = self._k2ok1(self.m, self.p)
        self._k1 = Cl_ideal/self._Cl_id(self.m, 1, k2ok1)
        self._k2 = k2ok1*self._k1
        self._Cl_ideal = Cl_ideal

    @staticmethod
    def _Cl_id(m: float, k1: float, k2ok1: float) -> float:
        return (k1/12)*(3*k2ok1*(2*m-1)*np.pi
                        + 3*(1-k2ok1)*(2*m-1)*np.arccos(1-2*m)
                        + 2*(1-k2ok1)*(4*m**2-4*m+3)*np.sqrt(m*(1-m)))

    @staticmethod
    def _k2ok1(m: float, p: float) -> float:
        return (3*(p-m)**2 - m**3)/(1 - m)**3

    @staticmethod
    def _Cmc4(m: float, k1: float, k2ok1: float, Cl_id) -> float:
        return -0.25*Cl_id + (k1/192)*(3*k2ok1*np.pi
                                       + 3*(1-k2ok1)*np.arccos(1-2*m)
                                       + 2*(1-k2ok1)*(1-2*m)*(8*m**2-8*m-3)
                                       * np.sqrt(m*(1-m)))


class Airfoil(Geometry):
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
