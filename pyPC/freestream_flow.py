#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All of the freestream type flows for panel methods.

This module contains all of the available freestream types for panel methods.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import Element2D


@dataclass
class FreestreamFlow2D(Element2D):
    """
    Represents two-dimensional freestream flows.

    Attributes
    ----------
    U_inf : float
        Magnitude of the freestream velocity.
        Angle of attack of the freestream velocity.
    """

    U_inf: float
    alpha: float

    def potential(self, xp: np_type.NDArray,
                  yp: np_type.NDArray, top: bool = True) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Branch cut flag that does not affect this class.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        return (self.U_inf*np.cos(self.alpha)*xp
                + self.U_inf*np.sin(self.alpha)*yp)

    def stream_function(self, xp: np_type.NDArray, yp: np_type.NDArray,
                        top: bool = True) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Branch cut flag that does not affect this class.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        return (-self.U_inf*np.sin(self.alpha)*xp
                + self.U_inf*np.cos(self.alpha)*yp)

    def velocity(self, xp: np_type.NDArray, yp: np_type.NDArray,
                 top: bool = True) -> Tuple[np_type.NDArray, np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Branch cut flag that does not affect this class.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        return (self.U_inf*np.cos(self.alpha)*np.ones_like(xp),
                self.U_inf*np.sin(self.alpha)*np.ones_like(yp))
