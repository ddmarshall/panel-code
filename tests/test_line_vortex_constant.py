#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:15:25 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.testing as npt

from pyPC.vortex_flow import LineVortexConstant2D
from approximate_elements import ApproximateLineVortexConstant2D


class TestLineVortexConstant2D(unittest.TestCase):
    """Class to test the constant strength 2D line vortex."""

    def testPotential(self) -> None:
        """Test the calculation of the potential."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        vortex = LineVortexConstant2D(x0=[0.1, 0.3], y0=[0.2, 0.5])
        vortex.set_strength(0.5)

        # test some hand calculations
        phi = vortex.potential(xpt[0], ypt[0])
        phi_ref = 0.00870528908
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

        phi = vortex.potential(xpt, ypt)
        phi_ref = np.array([0.00870528908, 0.00692249297])
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

    def testStreamFunction(self) -> None:
        """Test the calculation of the stream function."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        vortex = LineVortexConstant2D(x0=[0.1, 0.3], y0=[0.2, 0.5])
        vortex.set_strength(0.5)

        # test some hand calculations
        psi = vortex.stream_function(xpt[0], ypt[0])
        psi_ref = 0.000748597907
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

        psi = vortex.stream_function(xpt, ypt)
        psi_ref = np.array([0.000748597907, 0.0255915720])
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

    def testVelocity(self) -> None:
        """Test the calculation of the velocity vector."""
        xp = np.array([1., 2.])
        yp = np.array([1., 2.])

        vortex = LineVortexConstant2D(x0=[0.1, 0.3], y0=[0.2, 0.5])
        vortex.set_strength(0.5)

        # test some hand calculations
        u, v = vortex.velocity(xp[1], yp[1])
        u_ref = 0.00794542082
        v_ref = -0.00868245416
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

        u, v = vortex.velocity(xp, yp)
        u_ref = np.array([0.0175731676, 0.00794542082])
        v_ref = np.array([-0.0218908809, -0.00868245416])
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

    def testApproximateImplementation(self) -> None:
        """Test the calculations against a reference implementation."""
        # set mesh
        nptsx = 100
        nptsy = 100
        x, y = np.meshgrid(np.linspace(-1, 5, nptsx),
                           np.linspace(-1, 5, nptsy))

        # panel geometry
        xpan = [1, 2]
        ypan = [2, 4]
        gamma = 1

        # approximate values
        ns = 6000
        vortex_app = ApproximateLineVortexConstant2D(x0=xpan, y0=ypan,
                                                     gamma=gamma,
                                                     num_elements=ns)
        u_app, v_app = vortex_app.velocity(x, y)
        phi_app = vortex_app.potential(x, y)
        psi_app = vortex_app.stream_function(x, y)

        # values
        vortex = LineVortexConstant2D(x0=xpan, y0=ypan, strength=gamma)
        u, v = vortex.velocity(x, y)
        phi = vortex.potential(x, y)
        psi = vortex.stream_function(x, y)

        self.assertIsNone(npt.assert_allclose(phi, phi_app, rtol=0, atol=2e-4))
        self.assertIsNone(npt.assert_allclose(psi, psi_app, rtol=0, atol=2e-4))
        self.assertIsNone(npt.assert_allclose(u, u_app, rtol=0, atol=1e-3))
        self.assertIsNone(npt.assert_allclose(v, v_app, rtol=0, atol=5e-4))


if __name__ == "__main__":
    unittest.main(verbosity=1)
