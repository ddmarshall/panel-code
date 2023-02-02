#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:30:33 2023

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.camber import NoCamber


class TestNoCamber(unittest.TestCase):
    """Class to test the no camber geometry."""

    def testCamber(self) -> None:
        """Test the camber relations."""
        af = NoCamber()

        def compare_values(xi: np_type.NDArray, af: NoCamber) -> None:
            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)

            # compare point values
            y = af.y(xi)
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            yp_ref = y_ref
            yp = af.y_p(xi)
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            ypp_ref = y_ref
            ypp = af.y_pp(xi)
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

            # compare third derivatives
            yppp_ref = y_ref
            yppp = af.y_ppp(xi)
            self.assertIsNone(npt.assert_allclose(yppp, yppp_ref, atol=1e-7))

        # test point on front
        xi = 0.25
        compare_values(xi, af)

        # test point on back
        xi = 0.6
        compare_values(xi, af)

        # test points on lower and upper surface
        xi = np.linspace(0, 1, 12)
        compare_values(xi, af)

    def testEndpoints(self) -> None:
        af = NoCamber()
        p, m = af.max_camber()

        # reference values
        y_ref = [0, 0]
        yp_ref = [0, 0]
        ypp_ref = [0, 0]
        yppp_ref = [0, 0]

        # test leading edge
        xi = 0
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref[0]))

        # test trailing edge
        xi = 1
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref[1]))

        # test both
        xi = np.array([0, 1])
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref))

    def testJoints(self) -> None:
        af = NoCamber()

        self.assertListEqual([0.0, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        af = NoCamber()

        self.assertTupleEqual((0.0, 0.0), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
