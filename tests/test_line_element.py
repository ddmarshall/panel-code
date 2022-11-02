#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:49:39 2022

@author: ddmarshall
"""
from typing import Tuple

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.element_flow import LineElement2D


class LineElementTest(LineElement2D):
    """Represents a generic line element in 2 dimensions."""

    def __init__(self, xo: Tuple[float, float],
                 yo: Tuple[float, float]) -> None:
        super().__init__(xo=xo, yo=yo)

    def potential(self, xp: np_type.NDArray, yp: np_type.NDArray,
                  top: bool) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        return 0*xp

    def stream_function(self, xp: np_type.NDArray, yp: np_type.NDArray,
                        top: bool) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        return 0*xp

    def velocity(self, xp: np_type.NDArray, yp: np_type.NDArray,
                 top: bool) -> Tuple[np_type.NDArray, np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        return 0*xp, 0*yp


class TestLineElement2D(unittest.TestCase):
    """Class to test the constant strength 2D line doublet."""

    def testCoordinateSetting(self) -> None:
        """Test the setting of panel coordinates."""
        # test construction and that normal, tangent, and length are correct
        le = LineElementTest((0, 1), (0, 0))
        x_i, y_i = le.get_panel_start()
        x_ip1, y_ip1 = le.get_panel_end()
        ell = le.get_panel_length()
        sx, sy = le.get_panel_tangent()
        nx, ny = le.get_panel_normal()
        ell_ref = np.sqrt((x_ip1-x_i)**2 + (y_ip1-y_i)**2)
        sx_ref = (x_ip1-x_i)/ell_ref
        sy_ref = (y_ip1-y_i)/ell_ref
        nx_ref = -sy_ref
        ny_ref = sx_ref

        self.assertIsNone(npt.assert_allclose(ell, ell_ref))
        self.assertIsNone(npt.assert_allclose(sx, sx_ref))
        self.assertIsNone(npt.assert_allclose(sy, sy_ref))
        self.assertIsNone(npt.assert_allclose(nx, nx_ref))
        self.assertIsNone(npt.assert_allclose(ny, ny_ref))

        # test resetting and that normal, tangent, and length are correct
        le.set_panel_coordinates((1, 2), (2, 1))
        x_i, y_i = le.get_panel_start()
        x_ip1, y_ip1 = le.get_panel_end()
        ell = le.get_panel_length()
        sx, sy = le.get_panel_tangent()
        nx, ny = le.get_panel_normal()
        ell_ref = np.sqrt((x_ip1-x_i)**2 + (y_ip1-y_i)**2)
        sx_ref = (x_ip1-x_i)/ell_ref
        sy_ref = (y_ip1-y_i)/ell_ref
        nx_ref = -sy_ref
        ny_ref = sx_ref

        self.assertIsNone(npt.assert_allclose(ell, ell_ref))
        self.assertIsNone(npt.assert_allclose(sx, sx_ref))
        self.assertIsNone(npt.assert_allclose(sy, sy_ref))
        self.assertIsNone(npt.assert_allclose(nx, nx_ref))
        self.assertIsNone(npt.assert_allclose(ny, ny_ref))
        pass

    def test_XiEta(self) -> None:
        """Test the calculation of xi- and eta-coordinates."""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        x_i, y_i = le.get_panel_start()
        sx, sy = le.get_panel_tangent()
        nx, ny = le.get_panel_normal()
        xi, eta = le._get_xi_eta(x_test, y_test)

        x_rel = x_test - x_i
        y_rel = y_test - y_i
        xi_ref = x_rel*sx + y_rel*sy
        eta_ref = x_rel*nx + y_rel*ny
        self.assertIsNone(npt.assert_allclose(xi, xi_ref))
        self.assertIsNone(npt.assert_allclose(eta, eta_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        x_i, y_i = le.get_panel_start()
        sx, sy = le.get_panel_tangent()
        nx, ny = le.get_panel_normal()
        xi, eta = le._get_xi_eta(x_test, y_test)

        x_rel = x_test - x_i
        y_rel = y_test - y_i
        xi_ref = x_rel*sx + y_rel*sy
        eta_ref = x_rel*nx + y_rel*ny
        self.assertIsNone(npt.assert_allclose(xi, xi_ref))
        self.assertIsNone(npt.assert_allclose(eta, eta_ref))

    def test_GetITerms(self) -> None:
        """Test the calculation of the terms needed for integration."""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        ell = le.get_panel_length()
        xi_ref, eta_ref = le._get_xi_eta(x_test, y_test)
        r2_i_ref = xi_ref**2 + eta_ref**2
        r2_ip1_ref = (xi_ref - ell)**2 + eta_ref**2
        beta_i_ref = np.arctan2(eta_ref, xi_ref)
        beta_ip1_ref = np.arctan2(eta_ref, xi_ref-ell)

        # get top of branch cut
        beta_i_ref[4] = 0
        beta_ip1_ref[4] = np.pi
        beta_i_ref[5] = np.pi
        beta_ip1_ref[5] = np.pi

        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xi_ref, eta_ref,
                                                         True)
        self.assertIsNone(npt.assert_allclose(r2_i, r2_i_ref))
        self.assertIsNone(npt.assert_allclose(r2_ip1, r2_ip1_ref))
        self.assertIsNone(npt.assert_allclose(beta_i, beta_i_ref))
        self.assertIsNone(npt.assert_allclose(beta_ip1, beta_ip1_ref))

        # get bottom of branch cut
        beta_i_ref[4] = 0
        beta_ip1_ref[4] = -np.pi
        beta_i_ref[5] = -np.pi
        beta_ip1_ref[5] = -np.pi

        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xi_ref, eta_ref,
                                                         False)
        self.assertIsNone(npt.assert_allclose(r2_i, r2_i_ref))
        self.assertIsNone(npt.assert_allclose(r2_ip1, r2_ip1_ref))
        self.assertIsNone(npt.assert_allclose(beta_i, beta_i_ref))
        self.assertIsNone(npt.assert_allclose(beta_ip1, beta_ip1_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        ell = le.get_panel_length()
        xi_ref, eta_ref = le._get_xi_eta(x_test, y_test)
        r2_i_ref = xi_ref**2 + eta_ref**2
        r2_ip1_ref = (xi_ref - ell)**2 + eta_ref**2
        beta_i_ref = np.arctan2(eta_ref, xi_ref)
        beta_ip1_ref = np.arctan2(eta_ref, xi_ref-ell)

        # get top of branch cut
        beta_i_ref[1] = np.pi
        beta_ip1_ref[1] = np.pi

        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xi_ref, eta_ref,
                                                         True)
        self.assertIsNone(npt.assert_allclose(r2_i, r2_i_ref))
        self.assertIsNone(npt.assert_allclose(r2_ip1, r2_ip1_ref))
        self.assertIsNone(npt.assert_allclose(beta_i, beta_i_ref))
        self.assertIsNone(npt.assert_allclose(beta_ip1, beta_ip1_ref))

        # get bottom of branch cut
        beta_i_ref[1] = -np.pi
        beta_ip1_ref[1] = -np.pi

        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xi_ref, eta_ref,
                                                         False)
        self.assertIsNone(npt.assert_allclose(r2_i, r2_i_ref))
        self.assertIsNone(npt.assert_allclose(r2_ip1, r2_ip1_ref))
        self.assertIsNone(npt.assert_allclose(beta_i, beta_i_ref))
        self.assertIsNone(npt.assert_allclose(beta_ip1, beta_ip1_ref))

    def test_I00(self) -> None:
        """Test the calculation of I0,0"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I00_ref = 0.5*np.log(r2_i/r2_ip1)
        I00 = le._get_I00(r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I00, I00_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I00_ref = 0.5*np.log(r2_i/r2_ip1)
        I00 = le._get_I00(r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I00, I00_ref))

    def test_I01(self) -> None:
        """Test the calculation of I0,1"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))

        # get top of branch cut
        xip, etap = le._get_xi_eta(x_test, y_test)
        _, _, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I01_ref = beta_ip1 - beta_i
        I01 = le._get_I01(beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I01, I01_ref))

        # get top of branch cut
        xip, etap = le._get_xi_eta(x_test, y_test)
        _, _, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I01_ref = beta_ip1 - beta_i
        I01 = le._get_I01(beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I01, I01_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))

        # get top of branch cut
        xip, etap = le._get_xi_eta(x_test, y_test)
        _, _, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I01_ref = beta_ip1 - beta_i
        I01 = le._get_I01(beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I01, I01_ref))

        # get top of branch cut
        xip, etap = le._get_xi_eta(x_test, y_test)
        _, _, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I01_ref = beta_ip1 - beta_i
        I01 = le._get_I01(beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I01, I01_ref))

    def test_I02(self) -> None:
        """Test the calculation of I0,2"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I02_ref = ell*((xip/ell)*I00 + (etap/ell)*I01 + 0.5*np.log(r2_ip1) - 1)
        I02 = le._get_I02(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I02, I02_ref))

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I02_ref = ell*((xip/ell)*I00 + (etap/ell)*I01 + 0.5*np.log(r2_ip1) - 1)
        I02 = le._get_I02(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I02, I02_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I02_ref = ell*((xip/ell)*I00 + (etap/ell)*I01 + 0.5*np.log(r2_ip1) - 1)
        I02 = le._get_I02(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I02, I02_ref))

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I02_ref = ell*((xip/ell)*I00 + (etap/ell)*I01 + 0.5*np.log(r2_ip1) - 1)
        I02 = le._get_I02(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I02, I02_ref))

    def test_I03(self) -> None:
        """Test the calculation of I0,3"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I03_ref = ell*((etap/ell)*I00 - (xip/ell)*I01 + beta_ip1)
        I03 = le._get_I03(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I03, I03_ref))

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I03_ref = ell*((etap/ell)*I00 - (xip/ell)*I01 + beta_ip1)
        I03 = le._get_I03(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I03, I03_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, True)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I03_ref = ell*((etap/ell)*I00 - (xip/ell)*I01 + beta_ip1)
        I03 = le._get_I03(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I03, I03_ref))

        # get top of branch cut
        r2_i, r2_ip1, beta_i, beta_ip1 = le._get_I_terms(xip, etap, False)
        I00 = le._get_I00(r2_i, r2_ip1)
        I01 = le._get_I01(beta_i, beta_ip1)
        I03_ref = ell*((etap/ell)*I00 - (xip/ell)*I01 + beta_ip1)
        I03 = le._get_I03(xip, etap, r2_i, r2_ip1, beta_i, beta_ip1)
        self.assertIsNone(npt.assert_allclose(I03, I03_ref))

    def test_I04(self) -> None:
        """Test the calculation of I0,4"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I04_ref = etap/r2_ip1 - etap/r2_i
        I04 = le._get_I04(etap, r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I04, I04_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I04_ref = etap/r2_ip1 - etap/r2_i
        I04 = le._get_I04(etap, r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I04, I04_ref))

    def test_I05(self) -> None:
        """Test the calculation of I0,5"""
        le = LineElementTest((0, 1), (0, 0))
        x_test = np.array([2.3, -0.2, 0, 3, 0.5, -4])
        y_test = np.array([-3.1, 1.9, 2, 0, 0.0, 0])

        # test unrotated panel
        le.set_panel_coordinates((0, 1), (0, 0))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I05_ref = -((xip-ell)/r2_ip1 - xip/r2_i)
        I05 = le._get_I05(xip, r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I05, I05_ref))

        # test rotated panel this should be in line with 2nd test point
        le.set_panel_coordinates((0.8, 1.8), (0.9, -0.1))
        ell = le.get_panel_length()
        xip, etap = le._get_xi_eta(x_test, y_test)
        r2_i, r2_ip1, _, _ = le._get_I_terms(xip, etap, True)
        I05_ref = -((xip-ell)/r2_ip1 - xip/r2_i)
        I05 = le._get_I05(xip, r2_i, r2_ip1)
        self.assertIsNone(npt.assert_allclose(I05, I05_ref))


if __name__ == "__main__":
    unittest.main(verbosity=1)
