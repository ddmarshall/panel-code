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
            xi = np.asarray(xi)
            y_ref = np.zeros_like(xi)

            # compare point values
            x, y = af.xy(xi)
            self.assertIsNone(npt.assert_allclose(x, xi))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xt_ref = np.ones_like(xi)
            yt_ref = y_ref
            xt, yt = af.xy_t(xi)
            self.assertIsNone(npt.assert_allclose(xt, xt_ref))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref))

            # compare second derivatives
            xtt_ref = y_ref
            ytt_ref = y_ref
            xtt, ytt = af.xy_tt(xi)
            self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

            # compare third derivatives
            xttt_ref = y_ref
            yttt_ref = y_ref
            xttt, yttt = af.xy_ttt(xi)
            self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
            self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

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
        x_ref = [0, 1]
        y_ref = [0, 0]
        xt_ref = [1, 1]
        yt_ref = [0, 0]
        xtt_ref = [0, 0]
        ytt_ref = [0, 0]
        xttt_ref = [0, 0]
        yttt_ref = [0, 0]

        # test leading edge
        xi = 0
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[0]))

        # test trailing edge
        xi = 1
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[1]))

        # test both
        xi = np.array([0, 1])
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

    def testJoints(self) -> None:
        af = NoCamber()

        self.assertListEqual([0.0, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        af = NoCamber()

        self.assertTupleEqual((0.0, 0.0), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
