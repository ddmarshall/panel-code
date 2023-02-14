#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:43:42 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.cylinder import Cylinder


class TestCylinder(unittest.TestCase):
    """Class to test the cylinder geometry."""

    def testJoints(self) -> None:
        """Test the joints determination."""
        R = 1.5
        surf = Cylinder(radius=R)

        self.assertIsNone(npt.assert_allclose(surf.joints(), [-1, 1]))

    def testAirfoilTerms(self) -> None:
        """Test the airfoil specific calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        # test leading edge, trailing edge, chord
        xle, yle = surf.leading_edge()
        xte, yte = surf.trailing_edge()
        self.assertIsNone(npt.assert_allclose(xle, 0))
        self.assertIsNone(npt.assert_allclose(yle, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(xte, 2*R))
        self.assertIsNone(npt.assert_allclose(yte, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(surf.chord(), 2*R))

        def compare_camber_thickness(t: np_type.NDArray,
                                     surf: Cylinder) -> None:
            # compare thickness and camber
            thick_ref = surf.radius*np.sin(np.pi*(1-t))
            xc, yc = surf.camber_location(t)
            self.assertIsNone(npt.assert_allclose(xc, t))
            self.assertIsNone(npt.assert_allclose(yc, 0))
            self.assertIsNone(npt.assert_allclose(surf.thickness_value(t),
                                                  thick_ref))

        # test point on lower surface
        t = -0.35
        compare_camber_thickness(t, surf)

        # test point on upper surface
        t = 0.4
        compare_camber_thickness(t, surf)

        # test points on lower and upper surface
        t = np.linspace(-1, 1, 11)
        compare_camber_thickness(t, surf)

    def testsurfaceUnitVectors(self) -> None:
        """Test the surface unit vectors."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_vectors(t: np_type.NDArray, surf: Cylinder) -> None:
            # compare unit tangent
            sx_ref, sy_ref = surf.xy_t(t)
            temp = np.sqrt(sx_ref**2 + sy_ref**2)
            sx_ref = sx_ref/temp
            sy_ref = sy_ref/temp
            sx, sy = surf.tangent(t)
            self.assertIsNone(npt.assert_allclose(sx, sx_ref))
            self.assertIsNone(npt.assert_allclose(sy, sy_ref))

            # compare unit normal
            nx_ref = -sy_ref
            ny_ref = sx_ref
            nx, ny = surf.normal(t)
            self.assertIsNone(npt.assert_allclose(nx, nx_ref))
            self.assertIsNone(npt.assert_allclose(ny, ny_ref))

        # test point on lower surface
        t = -0.35
        compare_vectors(t, surf)

        # test point on upper surface
        t = 0.4
        compare_vectors(t, surf)

        # test points on lower and upper surface
        t = np.linspace(-1, 1, 11)
        compare_vectors(t, surf)

    def testCurvature(self) -> None:
        """Test the cuvature calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_curvature(t: np_type.NDArray, surf: Cylinder) -> None:
            # compare curvatures
            k_ref = -1/surf.radius
            k = surf.k(t)
            self.assertIsNone(npt.assert_allclose(k, k_ref))

        # test point on lower surface
        t = -0.35
        compare_curvature(t, surf)

        # test point on upper surface
        t = 0.4
        compare_curvature(t, surf)

        # test points on lower and upper surface
        t = np.linspace(-1, 1, 11)
        compare_curvature(t, surf)

    def testArcLengthCalculations(self) -> None:
        """Test the arc-length calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_arclength(t_s: np_type.NDArray, t_e: np_type.NDArray,
                              surf: Cylinder) -> None:
            alen_ref = surf.radius*np.pi*(t_e - t_s)
            alen = surf.arc_length(t_s, t_e)
            self.assertIsNone(npt.assert_allclose(alen_ref, alen))

        # test point on lower surface (both directions)
        t_s = -0.97
        t_e = -0.35
        compare_arclength(t_s, t_e, surf)
        compare_arclength(t_e, t_s, surf)

        # test point on upper surface (both directions)
        t_s = 0.4
        t_e = 0.78
        compare_arclength(t_s, t_e, surf)
        compare_arclength(t_e, t_s, surf)

        # # test points on lower and upper surface
        t_e = np.linspace(-1, 1, 11)
        t_s = -1.0
        compare_arclength(t_s, t_e, surf)

    def testParametricCalculations(self) -> None:
        """Test the calculations when given parametric value."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_values(t: np_type.NDArray, surf: Cylinder) -> None:
            eps = 1e-7
            x_ref = surf.radius*(1+np.cos(np.pi*(1-t)))
            y_ref = surf.radius*np.sin(np.pi*(1-t))

            # compare point values
            x, y = surf.xy(t)
            self.assertIsNone(npt.assert_allclose(x, x_ref))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xpl, ypl = surf.xy(t+eps)
            xmi, ymi = surf.xy(t-eps)
            xp_ref = 0.5*(xpl-xmi)/eps
            yp_ref = 0.5*(ypl-ymi)/eps
            xp, yp = surf.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xp, xp_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(yp, yp_ref))

            # compare second derivatives
            xtp, ytp = surf.xy_t(t+eps)
            xtm, ytm = surf.xy_t(t-eps)
            xpp_ref = 0.5*(xtp-xtm)/eps
            ypp_ref = 0.5*(ytp-ytm)/eps
            xpp, ypp = surf.xy_tt(t)
            self.assertIsNone(npt.assert_allclose(xpp, xpp_ref))
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        # test point on lower surface
        t = -0.35
        compare_values(t, surf)

        # test point on upper surface
        t = 0.4
        compare_values(t, surf)

        # test points on lower and upper surface
        t = np.linspace(-1, 1, 11)
        compare_values(t, surf)

    def testArclengthCalculations(self) -> None:
        """Test the calculations when given arc-length value."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_values(t: np_type.NDArray, surf: Cylinder) -> None:
            eps = 1e-7
            s = surf.arc_length(-1, t)

            # compare point values
            self.assertIsNone(npt.assert_allclose(t, surf.t_from_s(s)))
            x, y = surf.xy_from_s(s)
            x_ref, y_ref = surf.xy_from_s(s)
            self.assertIsNone(npt.assert_allclose(x, x_ref))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xpl, ypl = surf.xy_from_s(s+eps)
            xmi, ymi = surf.xy_from_s(s-eps)
            xs_ref = 0.5*(xpl-xmi)/eps
            ys_ref = 0.5*(ypl-ymi)/eps
            xs, ys = surf.xy_s(s)
            self.assertIsNone(npt.assert_allclose(xs, xs_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(ys, ys_ref))

            # compare second derivatives
            xsp, ysp = surf.xy_s(s+eps)
            xsm, ysm = surf.xy_s(s-eps)
            xss_ref = 0.5*(xsp-xsm)/eps
            yss_ref = 0.5*(ysp-ysm)/eps
            xss, yss = surf.xy_ss(s)
            self.assertIsNone(npt.assert_allclose(xss, xss_ref))
            self.assertIsNone(npt.assert_allclose(yss, yss_ref, atol=1e-7))

        def theta_fun(t: np_type.NDArray) -> np_type.NDArray:
            return np.pi*(1-t)

        # test point on lower surface
        t = -0.35
        compare_values(t, surf)

        # test point on upper surface
        t = 0.4
        compare_values(t, surf)

        # test points on lower and upper surface
        # Note: because of finite differencing in test cannot go from [-1, 1]
        t = np.linspace(-0.999, 0.999, 11)
        compare_values(t, surf)

        # test invalid points
        s_total = surf.arc_length(-1, 1)
        self.assertRaises(ValueError, surf.t_from_s, -0.2*s_total)
        self.assertRaises(ValueError, surf.t_from_s, 1.2*s_total)


if __name__ == "__main__":
    unittest.main(verbosity=1)
