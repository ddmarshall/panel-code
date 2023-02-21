#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes that use point elements to approximate behaviour of elements.

These classes use point elements to construct the approximate behaviour of
exact higher dimensional elements.
"""

from abc import abstractmethod

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import PointElement2D, LineElement2D
from pyPC.source_flow import PointSource2D
from pyPC.vortex_flow import PointVortex2D
from pyPC.doublet_flow import PointDoublet2D


class ApproxLineElement2D(LineElement2D):
    """
    Base class representing approximate line elements in two-dimensions.

    Attributes
    ----------
    ne : int
        The number of point elements to use to approximate line element.
    """

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 num_elements: int) -> None:
        super().__init__(xo=xo, yo=yo)

        self._ne = num_elements

    @property
    def ne(self) -> int:
        """Return number of elements."""
        return self._ne


class ApproxLineElementConstant2D(ApproxLineElement2D):
    """
    Approximate constant strength 2D line element.

    Attributes
    ----------
    strength : float
        The element strength.
    """

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 strength: float, num_elements: int) -> None:
        super().__init__(xo=xo, yo=yo, num_elements=num_elements)
        self._strength = strength

    @property
    def strength(self) -> float:
        """Return the element strength."""
        return self._strength

    def potential(self, xp: np_type.NDArray, yp: np_type.NDArray,
                  top: bool) -> np_type.NDArray:
        """
        Calculate the potential.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        element = self._setup_element()

        xpan = np.linspace(self._xo[0], self._xo[1], self.ne)
        ypan = np.linspace(self._yo[0], self._yo[1], self.ne)
        potential = np.zeros_like(xp)
        for xpa, ypa in zip(xpan, ypan):
            element.xo = xpa
            element.yo = ypa
            potential = potential + element.potential(xp, yp, top)

        return potential

    def stream_function(self, xp: np_type.NDArray, yp: np_type.NDArray,
                        top: bool) -> np_type.NDArray:
        """
        Calculate the stream function.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        element = self._setup_element()

        xpan = np.linspace(self._xo[0], self._xo[1], self.ne)
        ypan = np.linspace(self._yo[0], self._yo[1], self.ne)
        stream_function = np.zeros_like(xp)
        for xpa, ypa in zip(xpan, ypan):
            element.xo = xpa
            element.yo = ypa
            stream_function = (stream_function
                               + element.stream_function(xp, yp, top))

        return stream_function

    def velocity(self, xp: np_type.NDArray, yp: np_type.NDArray,
                 top: bool) -> Tuple[np_type.NDArray, np_type.NDArray]:
        """
        Calculate the velocity vector components.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        element = self._setup_element()

        xpan = np.linspace(self._xo[0], self._xo[1], self.ne)
        ypan = np.linspace(self._yo[0], self._yo[1], self.ne)
        u = np.zeros_like(xp)
        v = np.zeros_like(xp)
        for xpa, ypa in zip(xpan, ypan):
            element.xo = xpa
            element.yo = ypa
            utemp, vtemp = element.velocity(xp, yp, top)
            u = u + utemp
            v = v + vtemp

        return u, v

    @abstractmethod
    def _setup_element(self) -> PointElement2D:
        """
        Set up the point element for calculations.

        Returns
        -------
        PointSElement2D
            Point element class configured for analysis.
        """


class ApproxLineSourceConstant2D(ApproxLineElementConstant2D):
    """Approximate constant strength 2D line source."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 sigma: float, num_elements: int) -> None:
        super().__init__(xo=xo, yo=yo, strength=sigma,
                         num_elements=num_elements)

    def _setup_element(self) -> PointSource2D:
        """
        Set up the point source for calculations.

        Returns
        -------
        PointSource2D
            Point source class configured for analysis.
        """
        strength = self.strength*self.get_panel_length()/(self.ne-1)

        source = PointSource2D(xo=self._xo[0], yo=self._yo[0],
                               strength=strength,
                               angle=self.get_panel_angle())

        return source


class ApproxLineVortexConstant2D(ApproxLineElementConstant2D):
    """Approximate constant strength 2D line vortex."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 gamma: float, num_elements: int) -> None:
        super().__init__(xo=xo, yo=yo, strength=gamma,
                         num_elements=num_elements)

    def _setup_element(self) -> PointVortex2D:
        """
        Set up the point vortex for calculations.

        Returns
        -------
        PointVortex2D
            Point vortex class configured for analysis.
        """
        strength = self.strength*self.get_panel_length()/(self.ne-1)

        vortex = PointVortex2D(xo=self._xo[0], yo=self._yo[0],
                               strength=strength,
                               angle=self.get_panel_angle())

        return vortex


class ApproxLineDoubletConstant2D(ApproxLineElementConstant2D):
    """Approximate constant strength 2D line doublet."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 mu: float, num_elements: int) -> None:
        super().__init__(xo=xo, yo=yo, strength=mu, num_elements=num_elements)

    def _setup_element(self) -> PointDoublet2D:
        """
        Set up the point doublet for calculations.

        Returns
        -------
        PointDoublet2D
            Point doublet class configured for analysis.
        """
        strength = self.strength*self.get_panel_length()/(self.ne-1)

        # doublet is orientied in the xi-direction, so need to rotate angle
        doublet = PointDoublet2D(xo=self._xo[0], yo=self._yo[0],
                                 strength=strength,
                                 angle=0.5*np.pi+self.get_panel_angle())

        return doublet
