#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:15:25 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.testing as npt

from approximate_elements import ApproxLineDoubletConstant2D

from pyPC.doublet_flow import LineDoubletConstant2D


class TestLineDoubletConstant2D(unittest.TestCase):
    """Class to test the constant strength 2D line doublet."""

    def testPotential(self) -> None:
        """Test the calculation of the potential."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        doublet = LineDoubletConstant2D(x0=(0.1, 0.3), y0=(0.2, 0.5))
        doublet.set_strength(0.5)

        # test some hand calculations
        phi = doublet.potential(xpt[0], ypt[0])
        phi_ref = 0.00846647437
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

        phi = doublet.potential(xpt, ypt)
        phi_ref = np.array([0.00846647437, 0.00281691205])
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

    def testStreamFunction(self) -> None:
        """Test the calculation of the stream function."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        doublet = LineDoubletConstant2D(x0=(0.1, 0.3), y0=(0.2, 0.5))
        doublet.set_strength(0.5)

        # test some hand calculations
        psi = doublet.stream_function(xpt[0], ypt[0])
        psi_ref = -0.0267646351
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

        psi = doublet.stream_function(xpt, ypt)
        psi_ref = np.array([-0.0267646351, -0.0114271488])
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

    def testVelocity(self) -> None:
        """Test the calculation of the velocity vector."""
        xp = np.array([1., 2.])
        yp = np.array([1., 2.])

        doublet = LineDoubletConstant2D(x0=(0.1, 0.3), y0=(0.2, 0.5))
        doublet.set_strength(0.5)

        # test some hand calculations
        u, v = doublet.velocity(xp[1], yp[1])
        u_ref = 0.00231212910
        v_ref = -0.00424681386
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

        u, v = doublet.velocity(xp, yp)
        u_ref = np.array([0.00986374997, 0.00231212910])
        v_ref = np.array([-0.0258830732, -0.00424681386])
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

    def testApproximateImplementation(self) -> None:
        """Test the calculations against a reference implementation."""
        # set mesh
        x, y = np.meshgrid(np.linspace(-1, 5, 100),
                           np.linspace(-1, 5, 100))

        # values
        doublet = LineDoubletConstant2D(x0=(1, 2), y0=(2, 4), strength=1)
        u, v = doublet.velocity(x, y)
        phi = doublet.potential(x, y)
        psi = doublet.stream_function(x, y)

        # approximate values
        ns = 6000
        doublet_app = ApproxLineDoubletConstant2D(x0=doublet.x0,
                                                  y0=doublet.y0,
                                                  mu=doublet.get_strength(),
                                                  num_elements=ns)
        u_app, v_app = doublet_app.velocity(x, y)
        phi_app = doublet_app.potential(x, y)
        psi_app = doublet_app.stream_function(x, y)

        self.assertIsNone(npt.assert_allclose(phi, phi_app, rtol=0, atol=7e-4))
        self.assertIsNone(npt.assert_allclose(psi, psi_app, rtol=0, atol=9e-4))
        self.assertIsNone(npt.assert_allclose(u, u_app, rtol=0, atol=3e-2))
        self.assertIsNone(npt.assert_allclose(v, v_app, rtol=0, atol=2e-2))


if __name__ == "__main__":
    unittest.main(verbosity=1)
