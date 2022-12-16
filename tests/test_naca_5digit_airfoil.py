#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:19:40 2022

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil import Naca5DigitCamberClassic, Naca5DigitCamberEnhanced

from theory_of_wing_sections import read_camber_data


class TestNaca5Digit(unittest.TestCase):
    """Class to test the NACA 5-digit airfoil geometry."""

    def testCamberClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))

        # NACA 210xx
        af = Naca5DigitCamberClassic(camber_loc=1)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        x_ref, y_ref, yp_ref = read_camber_data(filename)
        y = af.y(x_ref)
        yp = af.y_p(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=3e-5))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref, rtol=0, atol=2e-5))

        # NACA 220xx
        af = Naca5DigitCamberClassic(camber_loc=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        x_ref, y_ref, yp_ref = read_camber_data(filename)
        y = af.y(x_ref)
        yp = af.y_p(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref, rtol=0, atol=1e-5))

        # NACA 230xx
        af = Naca5DigitCamberClassic(camber_loc=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        x_ref, y_ref, yp_ref = read_camber_data(filename)
        y = af.y(x_ref)
        yp = af.y_p(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref, rtol=0, atol=1e-5))

        # NACA 240xx
        af = Naca5DigitCamberClassic(camber_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        x_ref, y_ref, yp_ref = read_camber_data(filename)
        y = af.y(x_ref)
        yp = af.y_p(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref, rtol=0, atol=1e-5))

        # NACA 250xx
        af = Naca5DigitCamberClassic(camber_loc=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        x_ref, y_ref, yp_ref = read_camber_data(filename)
        y = af.y(x_ref)
        yp = af.y_p(x_ref)
        self.assertIsNone(npt.assert_allclose(y, y_ref, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref, rtol=0, atol=2e-5))

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamberEnhanced(0.05, 0.30)

        # Note: while published data from Jacobs and Pinkerton (1936) has
        #       values, they are noticable off from actual values. These
        #       reference values come from previous Matlab implementation.
        p = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        Cl_ideal = 0.3
        # m from ref.    [0.0580,       0.1260,       0.2025,
        #                 0.2900,       0.3910]
        m_ref = np.array([0.0580815972, 0.1257435084, 0.2026819782,
                          0.2903086448, 0.3913440104])
        # k1 from ref.    [361.4000,       51.6400,       15.9570,
        #                  6.6430,       3.2300]
        k1_ref = np.array([350.3324130671, 51.5774898103, 15.9196523168,
                            6.6240446646, 3.2227606900])

        # test the static methods
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            k1 = af._determine_k1(Cl_ideal, mit)
            y_p = af._camber_slope(pit, k1it, mit)
            self.assertIsNone(npt.assert_allclose(k1, k1it))
            self.assertIsNone(npt.assert_allclose(y_p, 0, atol=1e-7))

        # test the initialization of camber
        af.Cl_ideal = Cl_ideal
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            af.p = pit
            self.assertIsNone(npt.assert_allclose(af._m, mit))
            self.assertIsNone(npt.assert_allclose(af._k1, k1it))

    # def testCamber(self) -> None:
    #     """Test the camber relations."""
    #     af = Naca4DigitCamber(m=0.03, p=0.4)

    #     def compare_values(xi: np_type.NDArray, af: Naca4DigitCamber) -> None:
    #         eps = 1e-7

    #         xi_a = np.asarray(xi)
    #         y_ref = np.zeros_like(xi_a)
    #         it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
    #                                                 ["writeonly"]])
    #         with it:
    #             for xir, yr in it:
    #                 if xir <= af.p:
    #                     yr[...] = (af.m/af.p**2)*(2*af.p*xir - xir**2)
    #                 else:
    #                     yr[...] = (af.m/(1-af.p)**2)*(1-2*af.p + 2*af.p*xir
    #                                                   - xir**2)

    #         # compare point values
    #         y = af.y(xi)
    #         self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

    #         # compare first derivatives
    #         ypl = af.y(xi+eps)
    #         ymi = af.y(xi-eps)
    #         yp_ref = 0.5*(ypl-ymi)/eps
    #         yp = af.y_p(xi)
    #         self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

    #         # compare second derivatives
    #         ypl = af.y_p(xi+eps)
    #         ymi = af.y_p(xi-eps)
    #         ypp_ref = 0.5*(ypl-ymi)/eps
    #         ypp = af.y_pp(xi)
    #         self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

    #         # compare third derivatives
    #         ypl = af.y_pp(xi+eps)
    #         ymi = af.y_pp(xi-eps)
    #         yppp_ref = 0.5*(ypl-ymi)/eps
    #         yppp = af.y_ppp(xi)
    #         self.assertIsNone(npt.assert_allclose(yppp, yppp_ref, atol=1e-7))

    #     # test point on front
    #     xi = 0.25
    #     compare_values(xi, af)

    #     # test point on back
    #     xi = 0.6
    #     compare_values(xi, af)

    #     # test points on lower and upper surface
    #     xi = np.linspace(0, 1, 12)
    #     compare_values(xi, af)

    # def testThickness(self) -> None:
    #     """Test the thickness relations."""
    #     af = Naca4DigitThickness(t_max=0.3, closed_te=False)

    #     def compare_values(xi: np_type.NDArray,
    #                         af: Naca4DigitThickness) -> None:
    #         eps = 1e-7

    #         xi_a = np.asarray(xi)
    #         y_ref = np.zeros_like(xi_a)
    #         it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
    #                                                 ["writeonly"]])
    #         with it:
    #             for xir, yr in it:
    #                 yr[...] = (af.t_max/0.20)*(af._a[0]*np.sqrt(xir)
    #                                             + af._a[1]*xir
    #                                             + af._a[2]*xir**2
    #                                             + af._a[3]*xir**3
    #                                             + af._a[4]*xir**4)

    #         # compare point values
    #         y = af.y(xi)
    #         self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

    #         # compare first derivatives
    #         ypl = af.y(xi+eps)
    #         ymi = af.y(xi-eps)
    #         yp_ref = 0.5*(ypl-ymi)/eps
    #         yp = af.y_p(xi)
    #         self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

    #         # compare second derivatives
    #         ypl = af.y_p(xi+eps)
    #         ymi = af.y_p(xi-eps)
    #         ypp_ref = 0.5*(ypl-ymi)/eps
    #         ypp = af.y_pp(xi)
    #         self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

    #     # test point on front
    #     xi = 0.25
    #     compare_values(xi, af)

    #     # test point on back
    #     xi = 0.6
    #     compare_values(xi, af)

    #     # test points on lower and upper surface
    #     xi = np.linspace(0.001, 1, 12)
    #     compare_values(xi, af)


if __name__ == "__main__":
    unittest.main(verbosity=1)
