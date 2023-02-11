#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:28:41 2023

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.thickness import NoThickness


class TestNoThickness(unittest.TestCase):
    """Class to test the zero thickness geometry."""

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af = NoThickness()

        def compare_values(xi: np_type.NDArray, af: NoThickness) -> None:
            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)

            # compare point values
            t = np.sqrt(xi)
            x, y = af.xy(t)
            self.assertIsNone(npt.assert_allclose(x, xi))
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            yp_ref = y_ref
            xp, yp = af.xy_p(t)
            self.assertIsNone(npt.assert_allclose(xp, 2*t))
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            ypp_ref = y_ref
            xpp, ypp = af.xy_pp(t)
            self.assertIsNone(npt.assert_allclose(xpp, 2))
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        # test point on front
        xi = 0.25
        compare_values(xi, af)

        # test point on back
        xi = 0.6
        compare_values(xi, af)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        xi = np.linspace(0.001, 1, 12)
        compare_values(xi, af)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        af = NoThickness()

        # reference values
        x_ref = [0, 1]
        y_ref = [0, 0]
        xp_ref = [0, 2]
        yp_ref = [0, 0]
        xpp_ref = [2, 2]
        ypp_ref = [0, 0]
        k_ref = [0, 0]

        # test leading edge
        t = 0
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref[0], atol=1e-7))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        t = 1
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        t = np.array([0, 1])
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref))
        self.assertIsNone(npt.assert_allclose(k, k_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        af = NoThickness()

        self.assertListEqual([0.0, 1.0], af.joints())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        af = NoThickness()

        self.assertTupleEqual((0.0, 0.0), af.max_thickness())


if __name__ == "__main__":
    unittest.main(verbosity=1)
