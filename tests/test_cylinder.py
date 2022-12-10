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

from pyPC.geometry import Cylinder


class TestCylinder(unittest.TestCase):
    """Class to test the cylinder geometry."""

    def testAirfoilTerms(self) -> None:
        """Test the airfoil specific calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        # test leading edge, trailing edge, chord
        xle, yle = surf.leading_edge()
        xte, yte = surf.trailing_edge()
        self.assertIsNone(npt.assert_allclose(xle, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(yle, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(xte, 2*R, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(yte, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(surf.chord(), 2*R, atol=1e-7))

        def compare_camber_thickness(xi: np_type.NDArray,
                                     surf: Cylinder) -> None:
            # compare thickness and camber
            thick_ref = surf.radius*np.sin(np.pi*(1-xi))
            self.assertIsNone(npt.assert_allclose(surf.camber(xi), 0,
                                                  atol=1e-7))
            self.assertIsNone(npt.assert_allclose(surf.thickness(xi),
                                                  thick_ref, atol=1e-7))

        # test point on lower surface
        xi = -0.35
        compare_camber_thickness(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_camber_thickness(xi, surf)

        # test points on lower and upper surface
        xi = np.linspace(-1, 1, 11)
        compare_camber_thickness(xi, surf)

    def testsurfaceUnitVectors(self) -> None:
        """Test the surface unit vectors."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_vectors(xi: np_type.NDArray, surf: Cylinder) -> None:
            # compare unit tangent
            sx_ref, sy_ref = surf.xy_p(xi)
            temp = np.sqrt(sx_ref**2 + sy_ref**2)
            sx_ref = sx_ref/temp
            sy_ref = sy_ref/temp
            sx, sy = surf.tangent(xi)
            self.assertIsNone(npt.assert_allclose(sx, sx_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(sy, sy_ref, atol=1e-7))

            # compare unit normal
            nx_ref = -sy_ref
            ny_ref = sx_ref
            nx, ny = surf.normal(xi)
            self.assertIsNone(npt.assert_allclose(nx, nx_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(ny, ny_ref, atol=1e-7))

        # test point on lower surface
        xi = -0.35
        compare_vectors(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_vectors(xi, surf)

        # test points on lower and upper surface
        xi = np.linspace(-1, 1, 11)
        compare_vectors(xi, surf)

    def testCurvature(self) -> None:
        """Test the cuvature calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_curvature(xi: np_type.NDArray, surf: Cylinder) -> None:
            # compare curvatures
            k_ref = -1/surf.radius
            k = surf.k(xi)
            self.assertIsNone(npt.assert_allclose(k, k_ref, atol=1e-7))

        # test point on lower surface
        xi = -0.35
        compare_curvature(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_curvature(xi, surf)

        # test points on lower and upper surface
        xi = np.linspace(-1, 1, 11)
        compare_curvature(xi, surf)

    def testArcLengthCalculations(self) -> None:
        """Test the arc-length calculations."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_arclength(xi_s: np_type.NDArray, xi_e: np_type.NDArray,
                              surf: Cylinder) -> None:
            alen_ref = surf.radius*np.pi*(xi_e - xi_s)
            alen = surf.arc_length(xi_s=xi_s, xi_e=xi_e)
            self.assertIsNone(npt.assert_allclose(alen_ref, alen, atol=1e-7))

        # test point on lower surface (both directions)
        xi_s = -0.97
        xi_e = -0.35
        compare_arclength(xi_s, xi_e, surf)
        compare_arclength(xi_e, xi_s, surf)

        # test point on upper surface (both directions)
        xi_s = 0.4
        xi_e = 0.78
        compare_arclength(xi_s, xi_e, surf)
        compare_arclength(xi_e, xi_s, surf)

        # # test points on lower and upper surface
        xi_e = np.linspace(-1, 1, 11)
        xi_s = -1.0
        compare_arclength(xi_s, xi_e, surf)

    def testParametricCalculations(self) -> None:
        """Test the calculations when given parametric value."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_values(xi: np_type.NDArray, surf: Cylinder) -> None:
            eps = 1e-7
            x_ref = surf.radius*(1+np.cos(np.pi*(1-xi)))
            y_ref = surf.radius*np.sin(np.pi*(1-xi))

            # compare point values
            x, y = surf.xy_from_xi(xi)
            self.assertIsNone(npt.assert_allclose(x, x_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            xpl, ypl = surf.xy_from_xi(xi+eps)
            xmi, ymi = surf.xy_from_xi(xi-eps)
            xp_ref = 0.5*(xpl-xmi)/eps
            yp_ref = 0.5*(ypl-ymi)/eps
            xp, yp = surf.xy_p(xi)
            self.assertIsNone(npt.assert_allclose(xp, xp_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            xtp, ytp = surf.xy_p(xi+eps)
            xtm, ytm = surf.xy_p(xi-eps)
            xpp_ref = 0.5*(xtp-xtm)/eps
            ypp_ref = 0.5*(ytp-ytm)/eps
            xpp, ypp = surf.xy_pp(xi)
            self.assertIsNone(npt.assert_allclose(xpp, xpp_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        # test point on lower surface
        xi = -0.35
        compare_values(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_values(xi, surf)

        # test points on lower and upper surface
        xi = np.linspace(-1, 1, 11)
        compare_values(xi, surf)

    def testFunctionalCalculations(self) -> None:
        """Test the calculations when given x-coordinate."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_values(xi: np_type.NDArray, surf: Cylinder) -> None:
            eps = 1e-7
            upper = (np.asarray(xi) > 0).all()
            x, y = surf.xy_from_xi(xi)

            # compare point values
            self.assertIsNone(npt.assert_allclose(xi,
                                                  surf._xi_from_x(x, upper),
                                                  atol=1e-7))
            self.assertIsNone(npt.assert_allclose(y, surf.y(x, upper),
                                                  atol=1e-7))

            # compare slopes
            dydx_ref = 0.5*(surf.y(x+eps, upper)-surf.y(x-eps, upper))/eps
            self.assertIsNone(npt.assert_allclose(surf.dydx(x, upper),
                                                  dydx_ref, atol=1e-7))

            # compare second derivatives
            d2ydx2_ref = 0.5*(surf.dydx(x+eps, upper)
                              - surf.dydx(x-eps, upper))/eps
            self.assertIsNone(npt.assert_allclose(surf.d2ydx2(x, upper),
                                                  d2ydx2_ref, atol=1e-7))

        # test point on lower surface
        xi = -0.35
        compare_values(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_values(xi, surf)

        # test points on lower surface
        xi = np.linspace(-0.95, -0.05, 11)
        compare_values(xi, surf)

        # test points on upper surface
        xi = np.linspace(0.05, 0.95, 11)
        compare_values(xi, surf)

        # test invalid points
        self.assertRaises(ValueError, surf._xi_from_x, -0.2*R, False)
        self.assertRaises(ValueError, surf._xi_from_x, 2.2*R, False)

    def testArclengthCalculations(self) -> None:
        """Test the calculations when given arc-length value."""
        R = 1.5
        surf = Cylinder(radius=R)

        def compare_values(xi: np_type.NDArray, surf: Cylinder) -> None:
            eps = 1e-7
            s = surf.arc_length(-1, xi)

            # compare point values
            self.assertIsNone(npt.assert_allclose(xi, surf._xi_from_s(s),
                                                  atol=1e-7))
            x, y = surf.xy_from_s(s)
            x_ref, y_ref = surf.xy_from_s(s)
            self.assertIsNone(npt.assert_allclose(x, x_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            xpl, ypl = surf.xy_from_s(s+eps)
            xmi, ymi = surf.xy_from_s(s-eps)
            xs_ref = 0.5*(xpl-xmi)/eps
            ys_ref = 0.5*(ypl-ymi)/eps
            xs, ys = surf.xy_dot(s)
            self.assertIsNone(npt.assert_allclose(xs, xs_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(ys, ys_ref, atol=1e-7))

            # compare second derivatives
            xsp, ysp = surf.xy_dot(s+eps)
            xsm, ysm = surf.xy_dot(s-eps)
            xss_ref = 0.5*(xsp-xsm)/eps
            yss_ref = 0.5*(ysp-ysm)/eps
            xss, yss = surf.xy_ddot(s)
            self.assertIsNone(npt.assert_allclose(xss, xss_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(yss, yss_ref, atol=1e-7))

        def theta_fun(xi: np_type.NDArray) -> np_type.NDArray:
            return np.pi*(1-xi)

        # test point on lower surface
        xi = -0.35
        compare_values(xi, surf)

        # test point on upper surface
        xi = 0.4
        compare_values(xi, surf)

        # test points on lower and upper surface
        # Note: because of finite differencing in test cannot go from [-1, 1]
        xi = np.linspace(-0.999, 0.999, 11)
        compare_values(xi, surf)

        # test invalid points
        s_total = surf.arc_length(-1, 1)
        self.assertRaises(ValueError, surf._xi_from_s, -0.2*s_total)
        self.assertRaises(ValueError, surf._xi_from_s, 1.2*s_total)


if __name__ == "__main__":
    unittest.main(verbosity=1)
