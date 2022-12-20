#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:37:52 2022

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil import Naca4DigitModifiedThicknessBase
from pyPC.airfoil import Naca4DigitModifiedThicknessClassic
from pyPC.airfoil import Naca4DigitModifiedThicknessEnhanced

from theory_of_wing_sections import read_thickness_data


class TestNaca4DigitModified(unittest.TestCase):
    """Class to test the NACA modified 4-digit airfoil geometry."""

    def testSetters(self) -> None:
        """Test the setting of thickness parameters."""
        af = Naca4DigitModifiedThicknessClassic(thickness=0.20, le_radius=4,
                                                max_t_loc=5)

        # test static methods
        xi_ref = [0.2,  0.3,   0.4,   0.5,   0.6]
        tau_ref = [1.0, 1.170, 1.575, 2.325, 3.500]
        for xit, tauit in np.nditer([xi_ref, tau_ref]):
            tau = af._tau(xit)
            self.assertIsNone(npt.assert_allclose(tau, tauit))

        # sharp trailing edge
        M_ref = 4
        d_ref = [0, 1.575, -1.0833333333, -0.2546296297]
        km_ref = -3.0833333333
        d = af._calc_d_terms(eta=0.0, tau=tau_ref[M_ref-2],
                             xi_m=xi_ref[M_ref-2])
        km = af._k_m(eta=0.0, tau=tau_ref[M_ref-2], xi_m=xi_ref[M_ref-2])
        self.assertIsNone(npt.assert_allclose(d, d_ref))
        self.assertIsNone(npt.assert_allclose(km, km_ref))

        I_ref = 3
        a_ref = [0.74225, 0.9328327771, -2.6241657396, 1.2077076380]
        a = af._calc_a_terms(Iterm=I_ref, k_m=km_ref, xi_m=xi_ref[M_ref-2])
        self.assertIsNone(npt.assert_allclose(a, a_ref))

        # blunt trailing edge
        M_ref = 4
        d_ref = [0.01, 1.575, -1.1666666667, -0.1620370371]
        km_ref = -2.9166666667
        d = af._calc_d_terms(eta=0.02, tau=tau_ref[M_ref-2],
                             xi_m=xi_ref[M_ref-2])
        km = af._k_m(eta=0.02, tau=tau_ref[M_ref-2], xi_m=xi_ref[M_ref-2])
        self.assertIsNone(npt.assert_allclose(d, d_ref))
        self.assertIsNone(npt.assert_allclose(km, km_ref))

        I_ref = [4.2,          9.1]
        Q_ref = [0.0306075035, 0.0408100046]
        for Iit, Qit in np.nditer([I_ref, Q_ref]):
            Q = af._Q(Iit)
            self.assertIsNone(npt.assert_allclose(Q, Qit))

        I_ref = 3
        a_ref = [0.74225, 0.9661661252, -2.7908324797, 1.4160410631]
        a = af._calc_a_terms(Iterm=I_ref, k_m=km_ref, xi_m=xi_ref[M_ref-2])
        self.assertIsNone(npt.assert_allclose(a, a_ref))

        # test initializing parameters
        af.thickness = 0.12
        af.reset(le_radius=I_ref, max_t_loc=M_ref)
        self.assertIsNone(npt.assert_allclose(af.a, a_ref))
        self.assertIsNone(npt.assert_allclose(af.d, d_ref))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))

        # NACA 0008-34
        af = Naca4DigitModifiedThicknessClassic(thickness=0.08, le_radius=3,
                                                max_t_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=5e-5))

        # NACA 0010-34
        af = Naca4DigitModifiedThicknessClassic(thickness=0.10, le_radius=3,
                                                max_t_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=5e-5))

        # NACA 0010-35
        af = Naca4DigitModifiedThicknessClassic(thickness=0.10, le_radius=3,
                                                max_t_loc=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=6e-5))

        # NACA 0010-64
        af = Naca4DigitModifiedThicknessClassic(thickness=0.10, le_radius=6,
                                                max_t_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=6e-5))

        # NACA 0010-65
        af = Naca4DigitModifiedThicknessClassic(thickness=0.10, le_radius=6,
                                                max_t_loc=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=5e-5))

        # NACA 0010-66
        af = Naca4DigitModifiedThicknessClassic(thickness=0.10, le_radius=6,
                                                max_t_loc=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=6e-5))

        # NACA 0012-34
        af = Naca4DigitModifiedThicknessClassic(thickness=0.12, le_radius=3,
                                                max_t_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=6e-5))

        # NACA 0012-64
        af = Naca4DigitModifiedThicknessClassic(thickness=0.12, le_radius=6,
                                                max_t_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}"
                    + f"-{af.le_radius}{af.max_thickness_loc}.dat")
        x_ref, y_ref = read_thickness_data(filename)
        y = af.y(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=8e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca4DigitModifiedThicknessEnhanced(thickness=0.20, le_radius=4.3,
                                                 max_t_loc=0.56,
                                                 closed_te=False)

        # test the settings with open trailing edge
        xi_m = af.max_thickness_loc
        xi_te = 1.0
        y_m = af.y(xi_m)
        yp_m = af.y_p(xi_m)
        y_te = af.y(xi_te)
        self.assertIsNone(npt.assert_allclose(y_m, 0.5*af.thickness))
        self.assertIsNone(npt.assert_allclose(yp_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, 0.02*0.10))

        # test the settings with close trailing edge
        af.closed_te = True
        y_m = af.y(xi_m)
        yp_m = af.y_p(xi_m)
        y_te = af.y(xi_te)
        self.assertIsNone(npt.assert_allclose(y_m, 0.5*af.thickness))
        self.assertIsNone(npt.assert_allclose(yp_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, 0, atol=1e-7))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca4DigitModifiedThicknessClassic(thickness=0.10,
                                                     le_radius=6, max_t_loc=4)
        af_closed = Naca4DigitModifiedThicknessEnhanced(thickness=0.152,
                                                        le_radius=4.3,
                                                        max_t_loc=0.56,
                                                        closed_te=True)

        def compare_values(xi: np_type.NDArray,
                           af: Naca4DigitModifiedThicknessBase) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    if xir <= af.xi_m:
                        yr[...] = af.thickness*(af.a[0]*np.sqrt(xir)
                                                + af.a[1]*xir
                                                + af.a[2]*xir**2
                                                + af.a[3]*xir**3)
                    else:
                        yr[...] = af.thickness*(af.d[0] + af.d[1]*(1-xir)
                                                + af.d[2]*(1-xir)**2
                                                + af.d[3]*(1-xir)**3)

            # compare point values
            y = af.y(xi)
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            ypl = af.y(xi+eps)
            ymi = af.y(xi-eps)
            yp_ref = 0.5*(ypl-ymi)/eps
            yp = af.y_p(xi)
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            ypl = af.y_p(xi+eps)
            ymi = af.y_p(xi-eps)
            ypp_ref = 0.5*(ypl-ymi)/eps
            ypp = af.y_pp(xi)
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        # test point on front
        xi = 0.25
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test point on back
        xi = 0.60
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        xi = np.linspace(0.001, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)


if __name__ == "__main__":
    unittest.main(verbosity=1)
