#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:34:48 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.freestream_flow import FreestreamFlow2D


class TestFreestream2D(unittest.TestCase):
    """Class to test the 2D point source."""

    def testPotential(self) -> None:
        """Test the calculation of the potential."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        fs = FreestreamFlow2D(U_inf=10, alpha=0.2)

        # test some hand calculations
        phi_ref: float | np_type.NDArray
        phi = fs.potential(xpt[0], ypt[0])
        phi_ref = 11.7873591
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

        phi = fs.potential(xpt, ypt)
        phi_ref = np.array([11.7873591, 23.5747182])
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

    def testStreamFunction(self) -> None:
        """Test the calculation of the stream function."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        fs = FreestreamFlow2D(U_inf=10, alpha=0.2)

        # test some hand calculations
        psi_ref: float | np_type.NDArray
        psi = fs.stream_function(xpt[0], ypt[0])
        psi_ref = 7.81397247
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

        psi = fs.stream_function(xpt, ypt)
        psi_ref = np.array([7.81397247, 15.6279449])
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

    def testVelocity(self) -> None:
        """Test the calculation of the velocity vector."""
        xp = np.array([1., 2.])
        yp = np.array([1., 2.])

        fs = FreestreamFlow2D(U_inf=10, alpha=0.2)

        # test some hand calculations
        u_ref: float | np_type.NDArray
        v_ref: float | np_type.NDArray
        u, v = fs.velocity(xp[1], yp[1])
        u_ref = 9.80066578
        v_ref = 1.98669331
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

        u, v = fs.velocity(xp, yp)
        u_ref = np.array([9.80066578, 9.80066578])
        v_ref = np.array([1.98669331, 1.98669331])
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))


if __name__ == "__main__":
    unittest.main(verbosity=1)
