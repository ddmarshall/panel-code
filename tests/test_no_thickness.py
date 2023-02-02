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
        y_ref = [0, 0]
        yp_ref = [0, 0]
        ypp_ref = [0, 0]
        k_ref = [0, 0]

        # test leading edge
        xi = 0
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        xi = 1
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        xi = np.array([0, 1])
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
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
