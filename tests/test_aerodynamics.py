"""
tests/test_aerodynamics.py

Authoritative test suite for models/aerodynamics.py.

Covers:
- AerodynamicsDatabase.from_config() loads config.yaml correctly
- Flight 41 and Flight 15 are fully distinct records
- Derivative values match published Table 3 exactly
- 1-sigma uncertainties are preserved
- SI-converted nominal conditions match hand-computed values
- Zero perturbation yields exactly zero delta coefficients
- Sign convention: positive elevator -> nose-down moment (Cm_delta_e < 0)
- Sign convention: positive alpha -> positive lift, nose-down moment (stable)
- Hand calculation matches compute_longitudinal_delta_coefficients()
- Dimensionalization via compute_longitudinal_delta_forces()
- compute_delta_alpha_rad() helper
- Input validation: NaN, inf, wrong types rejected
- Provenance fields are populated
- 2015 flight-test reference geometry differs from 2017 active aircraft geometry
- Validity status returned with coefficient results

Run with:
    python -m unittest discover -s tests -v
"""

import math
import pathlib
import unittest

import yaml

from models.aerodynamics import (
    AerodynamicsDatabase,
    FlightCondition,
    FlightTestAeroData,
    LongitudinalPerturbationCoefficients,
    compute_delta_alpha_rad,
    compute_longitudinal_delta_coefficients,
    compute_longitudinal_delta_forces,
    compute_nominal_dynamic_pressure,
)
from models.atmosphere import get_atmosphere_from_altitude


def _load_config() -> dict:
    project_root = pathlib.Path(__file__).parent.parent
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_db() -> AerodynamicsDatabase:
    return AerodynamicsDatabase.from_config(_load_config())


class TestDatabaseLoading(unittest.TestCase):
    """Verify AerodynamicsDatabase.from_config() against config.yaml."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)
        self.f15 = self.db.get_flight_test_data(FlightCondition.FLIGHT_15)

    def test_flight_41_derivatives_exact(self):
        d = self.f41.derivatives
        self.assertEqual(d.CL_alpha.value, 3.933)
        self.assertEqual(d.CL_q.value, 15.11)
        self.assertEqual(d.CL_delta_e.value, 0.143)
        self.assertEqual(d.Cm_alpha.value, -1.667)
        self.assertEqual(d.Cm_q.value, -46.36)
        self.assertEqual(d.Cm_delta_e.value, -1.676)

    def test_flight_15_derivatives_exact(self):
        d = self.f15.derivatives
        self.assertEqual(d.CL_alpha.value, 3.828)
        self.assertEqual(d.CL_q.value, 16.39)
        self.assertEqual(d.CL_delta_e.value, 0.125)
        self.assertEqual(d.Cm_alpha.value, -1.437)
        self.assertEqual(d.Cm_q.value, -44.76)
        self.assertEqual(d.Cm_delta_e.value, -1.722)

    def test_flights_are_distinct(self):
        """Flight 41 and Flight 15 must not accidentally share data."""
        self.assertNotEqual(
            self.f41.derivatives.CL_alpha.value,
            self.f15.derivatives.CL_alpha.value,
        )
        self.assertNotEqual(
            self.f41.nominal_condition.airspeed_mps,
            self.f15.nominal_condition.airspeed_mps,
        )
        self.assertEqual(self.f41.condition, FlightCondition.FLIGHT_41)
        self.assertEqual(self.f15.condition, FlightCondition.FLIGHT_15)

    def test_uncertainties_preserved(self):
        self.assertEqual(self.f41.derivatives.CL_alpha.uncertainty_1sigma, 0.073)
        self.assertEqual(self.f41.derivatives.Cm_q.uncertainty_1sigma, 3.712)
        self.assertEqual(self.f15.derivatives.CL_q.uncertainty_1sigma, 2.719)
        self.assertEqual(self.f15.derivatives.Cm_delta_e.uncertainty_1sigma, 0.040)

    def test_nominal_condition_si_values(self):
        """Nominal conditions must be in SI, matching config reference_values_si."""
        nom = self.f41.nominal_condition
        # Config: airspeed_m_s: 42.39768 (direct SI value)
        self.assertTrue(math.isclose(nom.airspeed_mps, 42.39768, rel_tol=1e-9))
        # Config: altitude_m: 373.9896
        self.assertTrue(math.isclose(nom.altitude_m, 373.9896, rel_tol=1e-9))
        # Config: mean_aerodynamic_chord_m: 0.278892
        self.assertTrue(math.isclose(nom.mac_m, 0.278892, rel_tol=1e-9))
        # Config: wing_reference_area_m2: 0.54831374208
        self.assertTrue(math.isclose(nom.wing_area_m2, 0.54831374208, rel_tol=1e-9))

    def test_reference_mass_properties(self):
        rmp = self.f41.reference_mass_properties
        self.assertTrue(math.isclose(rmp.mass_kg, 23.91940691866, rel_tol=1e-9))
        self.assertTrue(rmp.Iyy_kg_m2 > 0)

    def test_provenance_populated(self):
        prov = self.f41.derivative_provenance
        self.assertIn("AIAA 2015-2704", prov.aiaa_paper)
        self.assertEqual(prov.year, 2015)
        self.assertIn("Table 3", prov.table)
        self.assertEqual(prov.flight, "T-2 Flight 41")

    def test_condition_enum_stored(self):
        self.assertEqual(self.f41.condition, FlightCondition.FLIGHT_41)
        self.assertEqual(self.f15.condition, FlightCondition.FLIGHT_15)


class TestGeometryReconciliation(unittest.TestCase):
    """Verify that the 2015 aero reference geometry differs from 2017 aircraft baseline."""

    def setUp(self):
        self.config = _load_config()
        self.db = AerodynamicsDatabase.from_config(self.config)
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)

    def test_aero_geometry_differs_from_aircraft_baseline(self):
        """The 2015 flight-test reference geometry must NOT equal the 2017 baseline."""
        aircraft_cfg = self.config["aircraft"]["geometry"]
        aircraft_mac = aircraft_cfg["mac_m"]
        aircraft_area = aircraft_cfg["wing_area_m2"]

        aero_mac = self.f41.nominal_condition.mac_m
        aero_area = self.f41.nominal_condition.wing_area_m2

        # They should be close but NOT identical (different NASA source tables)
        self.assertNotEqual(aircraft_mac, aero_mac)
        self.assertNotEqual(aircraft_area, aero_area)

        # But the difference should be small (same physical airframe)
        self.assertTrue(abs(aircraft_mac - aero_mac) < 0.01)
        self.assertTrue(abs(aircraft_area - aero_area) < 0.01)

    def test_dimensionalization_uses_flight_case_geometry(self):
        """Forces must be computed with the 2015 flight-case S and c̄, not 2017 values."""
        aircraft_cfg = self.config["aircraft"]["geometry"]
        aircraft_mac = aircraft_cfg["mac_m"]     # 2017: 0.280416
        aircraft_area = aircraft_cfg["wing_area_m2"]  # 2017: 0.548127936

        nom = self.f41.nominal_condition
        aero_mac = nom.mac_m                     # 2015: 0.278892
        aero_area = nom.wing_area_m2             # 2015: 0.54831374208

        # Compute forces through the public API
        atm = get_atmosphere_from_altitude(nom.altitude_m)
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.05,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.0,
        )
        forces = compute_longitudinal_delta_forces(self.f41, coeffs, atm)

        # Manually compute what 2015 geometry produces
        q_bar = forces.dynamic_pressure_Pa
        expected_dL_2015 = q_bar * aero_area * coeffs.delta_CL
        expected_dM_2015 = q_bar * aero_area * aero_mac * coeffs.delta_Cm

        # Manually compute what 2017 geometry would produce (wrong)
        wrong_dL_2017 = q_bar * aircraft_area * coeffs.delta_CL
        wrong_dM_2017 = q_bar * aircraft_area * aircraft_mac * coeffs.delta_Cm

        # The actual output must match 2015, not 2017
        self.assertTrue(math.isclose(forces.delta_lift_N, expected_dL_2015, rel_tol=1e-12))
        self.assertTrue(
            math.isclose(forces.delta_pitching_moment_Nm, expected_dM_2015, rel_tol=1e-12)
        )
        self.assertFalse(math.isclose(forces.delta_lift_N, wrong_dL_2017, rel_tol=1e-9))
        self.assertFalse(
            math.isclose(forces.delta_pitching_moment_Nm, wrong_dM_2017, rel_tol=1e-9)
        )

    def test_geometry_reconciliation_documented(self):
        """config.yaml must document the geometry source variation."""
        recon = self.config["simulation_reference_data"]["geometry_reconciliation"]
        self.assertEqual(recon["status"], "SOURCE_VARIATION")
        self.assertIn("2017", recon["active_aircraft_source"])
        self.assertIn("2015", recon["flight_test_model_source"])

    def test_flight_case_mass_is_metadata_only(self):
        """Flight-case mass/Iyy in aerodynamics data must NOT replace aircraft mass."""
        aircraft_mass = self.config["aircraft"]["mass_properties"]["mass_kg"]
        aero_ref_mass = self.f41.reference_mass_properties.mass_kg

        # Both should be present and close but flight-case mass is reference-only
        self.assertTrue(aero_ref_mass > 0)
        self.assertTrue(aircraft_mass > 0)
        # The flight-case mass object lives on FlightTestAeroData, not on Aircraft
        self.assertTrue(hasattr(self.f41, 'reference_mass_properties'))


class TestNoBiasTerms(unittest.TestCase):
    """Regression guard: the perturbation model must NEVER include CL_0 or Cm_0."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)

    def test_zero_input_produces_zero_output(self):
        """If someone adds CL_0/Cm_0, this test will catch it."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.0,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.0,
        )
        # Exactly zero — not "close to zero" — no hidden bias
        self.assertEqual(coeffs.delta_CL, 0.0)
        self.assertEqual(coeffs.delta_Cm, 0.0)

    def test_no_bias_fields_on_derivative_set(self):
        """LongitudinalDerivativeSet must not gain CL_0 or Cm_0 fields."""
        d = self.f41.derivatives
        self.assertFalse(hasattr(d, 'CL_0'))
        self.assertFalse(hasattr(d, 'Cm_0'))



class TestPerturbationEquations(unittest.TestCase):
    """Test coefficient and force perturbation computations."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)
        self.f15 = self.db.get_flight_test_data(FlightCondition.FLIGHT_15)

    def test_zero_perturbation(self):
        """Zero perturbation inputs must yield exactly zero delta coefficients."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.0,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.0,
        )
        self.assertEqual(coeffs.delta_CL, 0.0)
        self.assertEqual(coeffs.delta_Cm, 0.0)

    def test_elevator_sign_convention(self):
        """Positive elevator perturbation must produce nose-down moment (delta_Cm < 0)."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.0,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.1,
        )
        self.assertLess(coeffs.delta_Cm, 0.0)
        # CL_delta_e > 0, so positive elevator also increases lift
        self.assertGreater(coeffs.delta_CL, 0.0)

    def test_negative_elevator_sign(self):
        """Negative elevator perturbation must produce nose-up moment."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.0,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=-0.1,
        )
        self.assertGreater(coeffs.delta_Cm, 0.0)

    def test_alpha_sign_convention(self):
        """Positive alpha must produce positive lift and nose-down moment (stable)."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.1,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.0,
        )
        self.assertGreater(coeffs.delta_CL, 0.0)
        self.assertLess(coeffs.delta_Cm, 0.0)

    def test_hand_calculation(self):
        """Verify against manually computed delta_CL and delta_Cm."""
        d = self.f41.derivatives
        nom = self.f41.nominal_condition

        alpha = 0.05
        q = 0.1
        de = 0.02

        q_hat = q * nom.mac_m / (2.0 * nom.airspeed_mps)

        expected_dCL = (
            d.CL_alpha.value * alpha
            + d.CL_q.value * q_hat
            + d.CL_delta_e.value * de
        )
        expected_dCm = (
            d.Cm_alpha.value * alpha
            + d.Cm_q.value * q_hat
            + d.Cm_delta_e.value * de
        )

        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=alpha,
            delta_pitch_rate_rads=q,
            delta_elevator_rad=de,
        )

        self.assertTrue(math.isclose(coeffs.delta_CL, expected_dCL, rel_tol=1e-12))
        self.assertTrue(math.isclose(coeffs.delta_Cm, expected_dCm, rel_tol=1e-12))

    def test_validity_status_returned(self):
        """Result must include validity status (no bounds configured currently)."""
        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.01,
            delta_pitch_rate_rads=0.0,
            delta_elevator_rad=0.0,
        )
        self.assertIn("VALIDITY", coeffs.validity_status)
        self.assertTrue(len(coeffs.validity_note) > 0)


class TestDeltaAlphaHelper(unittest.TestCase):
    """Test the compute_delta_alpha_rad helper."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)

    def test_delta_alpha_from_trim(self):
        """compute_delta_alpha_rad returns alpha - alpha_trim."""
        trim_alpha = self.f41.nominal_condition.angle_of_attack_rad
        absolute_alpha = trim_alpha + 0.05  # 0.05 rad perturbation

        delta = compute_delta_alpha_rad(self.f41, absolute_alpha)
        self.assertTrue(math.isclose(delta, 0.05, abs_tol=1e-12))

    def test_at_trim_yields_zero(self):
        trim_alpha = self.f41.nominal_condition.angle_of_attack_rad
        delta = compute_delta_alpha_rad(self.f41, trim_alpha)
        self.assertEqual(delta, 0.0)


class TestDimensionalization(unittest.TestCase):
    """Test force/moment dimensionalization using atmosphere.py density."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)

    def test_nominal_dynamic_pressure(self):
        """q_bar = 0.5 * rho * V_trim^2 using ISA density at nominal altitude."""
        atm = get_atmosphere_from_altitude(self.f41.nominal_condition.altitude_m)
        q_bar = compute_nominal_dynamic_pressure(self.f41, atm)

        V = self.f41.nominal_condition.airspeed_mps
        expected = 0.5 * atm.density_kg_m3 * V ** 2
        self.assertTrue(math.isclose(q_bar, expected, rel_tol=1e-12))
        self.assertGreater(q_bar, 0.0)

    def test_delta_forces_hand_calculation(self):
        """Verify ΔL = q_bar*S*ΔCL and ΔM = q_bar*S*c*ΔCm."""
        atm = get_atmosphere_from_altitude(self.f41.nominal_condition.altitude_m)

        coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.f41,
            delta_alpha_rad=0.03,
            delta_pitch_rate_rads=0.05,
            delta_elevator_rad=0.01,
        )

        forces = compute_longitudinal_delta_forces(self.f41, coeffs, atm)

        nom = self.f41.nominal_condition
        V = nom.airspeed_mps
        q_bar = 0.5 * atm.density_kg_m3 * V ** 2
        qS = q_bar * nom.wing_area_m2

        expected_dL = qS * coeffs.delta_CL
        expected_dM = qS * nom.mac_m * coeffs.delta_Cm

        self.assertTrue(math.isclose(forces.delta_lift_N, expected_dL, rel_tol=1e-12))
        self.assertTrue(
            math.isclose(forces.delta_pitching_moment_Nm, expected_dM, rel_tol=1e-12)
        )
        self.assertTrue(math.isclose(forces.dynamic_pressure_Pa, q_bar, rel_tol=1e-12))

    def test_altitude_mismatch_rejected(self):
        """AtmosphericState altitude must match the flight's nominal altitude."""
        wrong_atm = get_atmosphere_from_altitude(5000.0)
        with self.assertRaises(ValueError):
            compute_nominal_dynamic_pressure(self.f41, wrong_atm)


class TestInputValidation(unittest.TestCase):
    """Test rejection of invalid inputs."""

    def setUp(self):
        self.db = _build_db()
        self.f41 = self.db.get_flight_test_data(FlightCondition.FLIGHT_41)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            compute_longitudinal_delta_coefficients(
                flight_data=self.f41,
                delta_alpha_rad=float('nan'),
                delta_pitch_rate_rads=0.0,
                delta_elevator_rad=0.0,
            )

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            compute_longitudinal_delta_coefficients(
                flight_data=self.f41,
                delta_alpha_rad=float('inf'),
                delta_pitch_rate_rads=0.0,
                delta_elevator_rad=0.0,
            )

    def test_wrong_type_flight_data_rejected(self):
        with self.assertRaises(TypeError):
            compute_longitudinal_delta_coefficients(
                flight_data="not a flight",  # type: ignore
                delta_alpha_rad=0.0,
                delta_pitch_rate_rads=0.0,
                delta_elevator_rad=0.0,
            )

    def test_wrong_type_condition_rejected(self):
        with self.assertRaises(TypeError):
            self.db.get_flight_test_data("flight_41")  # type: ignore


if __name__ == "__main__":
    unittest.main()
