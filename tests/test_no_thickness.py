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
            yt_ref = y_ref
            xt, yt = af.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xt, 2*t))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref, atol=1e-7))

            # compare second derivatives
            ytt_ref = y_ref
            xtt, ytt = af.xy_tt(t)
            self.assertIsNone(npt.assert_allclose(xtt, 2))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref, atol=1e-7))

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
        xt_ref = [0, 2]
        yt_ref = [0, 0]
        xtt_ref = [2, 2]
        ytt_ref = [0, 0]
        k_ref = [0, 0]

        # test leading edge
        t = 0
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0], atol=2e-8))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        t = 1
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        t = np.array([0, 1])
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref, atol=2e-8))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))
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
