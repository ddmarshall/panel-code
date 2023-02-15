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
        ca = NoCamber()

        def compare_values(t: np_type.NDArray, af: NoCamber) -> None:
            t = np.asarray(t)

            # compare point values
            x_ref = t
            y_ref = np.zeros_like(t)
            x, y = af.xy(t)
            self.assertIsNone(npt.assert_allclose(x, x_ref))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xt_ref = np.ones_like(t)
            yt_ref = np.zeros_like(t)
            xt, yt = af.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xt, xt_ref))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref))

            # compare second derivatives
            xtt_ref = np.zeros_like(t)
            ytt_ref = np.zeros_like(t)
            xtt, ytt = af.xy_tt(t)
            self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

            # compare third derivatives
            xttt_ref = np.zeros_like(t)
            yttt_ref = np.zeros_like(t)
            xttt, yttt = af.xy_ttt(t)
            self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
            self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

            # compare curvature derivatives derivatives
            kt_ref = np.zeros_like(t)
            kt = ca.k_t(t)
            self.assertIsNone(npt.assert_allclose(kt, kt_ref, atol=1e-7))

        # test point on front
        t = 0.25
        compare_values(t, ca)

        # test point on back
        t = 0.6
        compare_values(t, ca)

        # test points on lower and upper surface
        t = np.linspace(0, 1, 12)
        compare_values(t, ca)

    def testEndpoints(self) -> None:
        af = NoCamber()

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
        t = 0
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        xttt, yttt = af.xy_ttt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[0]))

        # test trailing edge
        t = 1
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        xttt, yttt = af.xy_ttt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[1]))

        # test both
        t = np.array([0, 1])
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        xttt, yttt = af.xy_ttt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

    def testJoints(self) -> None:
        ca = NoCamber()

        self.assertListEqual([0.0, 1.0], ca.joints())

    def testMaxCamber(self) -> None:
        ca = NoCamber()

        self.assertEqual(0.0, ca.max_camber_parameter())


if __name__ == "__main__":
    unittest.main(verbosity=1)
