#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:15:25 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from approximate_elements import ApproxLineSourceConstant2D

from pyPC.source_flow import LineSourceConstant2D


class TestLineSourceConstant2D(unittest.TestCase):
    """Class to test the constant strength 2D line source."""

    def testPotential(self) -> None:
        """Test the calculation of the potential."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        source = LineSourceConstant2D(xo=(0.1, 0.3), yo=(0.2, 0.5))
        source.set_strength(0.5)

        # test some hand calculations
        phi_ref: float | np_type.NDArray
        phi = source.potential(xpt[0], ypt[0], True)
        phi_ref = 0.000748597907
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

        phi = source.potential(xpt, ypt, True)
        phi_ref = np.array([0.000748597907, 0.0255915720])
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

    def testStreamFunction(self) -> None:
        """Test the calculation of the stream function."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        source = LineSourceConstant2D(xo=(0.1, 0.3), yo=(0.2, 0.5))
        source.set_strength(0.5)

        # test some hand calculations
        psi_ref: float | np_type.NDArray
        psi = source.stream_function(xpt[0], ypt[0], True)
        psi_ref = -0.00870528908
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

        psi = source.stream_function(xpt, ypt, True)
        psi_ref = np.array([-0.00870528908, -0.00692249297])
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

    def testVelocity(self) -> None:
        """Test the calculation of the velocity vector."""
        xp = np.array([1., 2.])
        yp = np.array([1., 2.])

        source = LineSourceConstant2D(xo=(0.1, 0.3), yo=(0.2, 0.5))
        source.set_strength(0.5)

        # test some hand calculations
        u_ref: float | np_type.NDArray
        v_ref: float | np_type.NDArray
        u, v = source.velocity(xp[1], yp[1], True)
        u_ref = 0.00868245416
        v_ref = 0.00794542082
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

        u, v = source.velocity(xp, yp, True)
        u_ref = np.array([0.0218908809, 0.00868245416])
        v_ref = np.array([0.0175731676, 0.00794542082])
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

    def testApproximateImplementation(self) -> None:
        """Test the calculations against a reference implementation."""
        # set mesh
        x, y = np.meshgrid(np.linspace(-1, 5, 100),
                           np.linspace(-1, 5, 100))

        # values
        source = LineSourceConstant2D(xo=(1, 2), yo=(2, 4), strength=1)
        u, v = source.velocity(x, y, True)
        phi = source.potential(x, y, True)
        psi = source.stream_function(x, y, True)

        # approximate values
        ns = 6000
        source_app = ApproxLineSourceConstant2D(xo=source.get_panel_xo(),
                                                yo=source.get_panel_yo(),
                                                sigma=source.get_strength(),
                                                num_elements=ns)
        u_app, v_app = source_app.velocity(x, y, True)
        phi_app = source_app.potential(x, y, True)
        psi_app = source_app.stream_function(x, y, True)

        self.assertIsNone(npt.assert_allclose(phi, phi_app, atol=2e-4))
        self.assertIsNone(npt.assert_allclose(psi, psi_app, atol=2e-4))
        self.assertIsNone(npt.assert_allclose(u, u_app, atol=5e-4))
        self.assertIsNone(npt.assert_allclose(v, v_app, atol=2e-3))


if __name__ == "__main__":
    unittest.main(verbosity=1)
