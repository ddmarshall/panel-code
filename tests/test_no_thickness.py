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

        def compare_values(t: np_type.NDArray, th: NoThickness) -> None:
            t = np.asarray(t)

            # compare point values
            delta_ref = np.zeros_like(t)
            delta = th.delta(t)
            self.assertIsNone(npt.assert_allclose(delta, delta_ref))

            # compare first derivatives
            deltat_ref = np.zeros_like(t)
            deltat = th.delta_t(t)
            self.assertIsNone(npt.assert_allclose(deltat, deltat_ref))

            # compare second derivatives
            deltatt_ref = np.zeros_like(t)
            deltatt = th.delta_tt(t)
            self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref))

        # test point on front
        t = 0.25
        compare_values(t, af)

        # test point on back
        t = 0.6
        compare_values(t, af)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        t = np.linspace(0.001, 1, 12)
        compare_values(t, af)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        th = NoThickness()

        # reference values
        delta_ref = [0, 0]
        deltat_ref = [0, 0]
        deltatt_ref = [0, 0]

        # test leading edge
        t = 0
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref[0]))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref[0]))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref[0]))

        # test trailing edge
        t = 1
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref[1]))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref[1]))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref[1]))

        # test both
        t = np.array([0, 1])
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref))

    def testDiscontinuities(self) -> None:
        """Test correct joints are being reported."""
        th = NoThickness()

        self.assertListEqual([], th.discontinuities())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        th = NoThickness()

        self.assertTupleEqual((0.0, 0.0), th.max_thickness())


if __name__ == "__main__":
    unittest.main(verbosity=1)
