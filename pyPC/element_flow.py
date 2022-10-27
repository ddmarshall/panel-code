#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All of the base element classes for panel methods.

This module contains all of the base element classes for panel methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as np_type


@dataclass
class PointElement2D(ABC):
    """
    Base class for 2D point elements.

    Attributes
    ----------
    x0: float
        X-coorinate of source origin.
    y0: float
        Y-coorinate of source origin.
    angle: float
        Orientation angle of element in radians.
    """

    x0: float
    y0: float
    angle: float
    _strength_over_2pi: float = field(init=False)

    def __post_init__(self) -> None:
        """Configure the strength base parameter."""
        self._strength_over_2pi = 1/(2*np.pi)

    def get_strength(self) -> float:
        """
        Return the strength of the point element.

        Returns
        -------
        float
            Point element strength.
        """
        return (2*np.pi)*self._strength_over_2pi

    def set_strength(self, strength: float) -> None:
        """
        Set the strength of the point element.

        Parameters
        ----------
        strength : float
            Point element strength.
        """
        self._strength_over_2pi = strength/(2*np.pi)

    @abstractmethod
    def potential(self, xp: np_type.NDArray,
                  yp: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """

    @abstractmethod
    def stream_function(self, xp: np_type.NDArray,
                        yp: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """

    @abstractmethod
    def velocity(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """

    def _r_terms(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the radius terms needed for many calculations.

        Parameters
        ----------
        xp : np_type.NDArray
            X-coordinate of the point to calculate terms.
        yp : np_type.NDArray
            Y-coordinate of the point to calculate terms.

        Returns
        -------
        rx : np_type.NDArray
            X-component of the vector from element to point.
        ry : np_type.NDArray
            Y-component of the vector from element to point.
        rmag2 : np_type.NDArray
            Square of the distance from element to point.
        """
        rx = xp - self.x0
        ry = yp - self.y0
        rmag2 = rx**2 + ry**2
        return rx, ry, rmag2
