"""
tests/test_atmosphere.py

Authoritative test suite for models/atmosphere.py.

Covers:
  - Valid happy-path ISA reference values (sea level, tropopause, stratosphere)
  - Internal consistency (rho = p/(R*T), a = sqrt(gamma*R*T))
  - Layer-boundary continuity at 11,000 m (T, p, rho, a)
  - Monotonicity of T (troposphere), p, and rho across both layers
  - NED conversion round-trip (get_atmosphere_from_ned_z)
  - Invalid-input rejection: NaN, ±infinity, negative altitude,
    altitude above 20,000 m, wrong types (str, bool, None, list)

Run with:
    python -m pytest tests/test_atmosphere.py -v
or:
    python -m unittest tests/test_atmosphere.py
"""

import math
import unittest

from models.atmosphere import (
    TROPOPAUSE_ALT_M,
    MAX_SUPPORTED_ALTITUDE_M,
    MIN_SUPPORTED_ALTITUDE_M,
    R_SPECIFIC_AIR,
    GAMMA_AIR,
    RHO0_KG_M3,
    get_atmosphere_from_altitude,
    get_atmosphere_from_ned_z,
)


# =============================================================================
# Helper
# =============================================================================

def _assert_internal_consistency(tc: unittest.TestCase, state) -> None:
    """rho == p/(R*T)  and  a == sqrt(gamma*R*T) must hold for any state."""
    expected_rho = state.pressure_Pa / (R_SPECIFIC_AIR * state.temperature_K)
    tc.assertTrue(
        math.isclose(state.density_kg_m3, expected_rho, rel_tol=1e-9),
        f"rho mismatch at {state.altitude_m} m: "
        f"got {state.density_kg_m3}, expected {expected_rho}",
    )
    expected_a = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * state.temperature_K)
    tc.assertTrue(
        math.isclose(state.speed_of_sound_mps, expected_a, rel_tol=1e-9),
        f"speed-of-sound mismatch at {state.altitude_m} m: "
        f"got {state.speed_of_sound_mps}, expected {expected_a}",
    )


# =============================================================================
# Happy-path / reference-value tests
# =============================================================================

class TestISAReferenceValues(unittest.TestCase):
    """ISA tabulated reference values (ICAO Doc 7488 / NASA TM-X-74335)."""

    def test_sea_level_temperature(self):
        s = get_atmosphere_from_altitude(0.0)
        self.assertTrue(math.isclose(s.temperature_K, 288.15, rel_tol=1e-6))

    def test_sea_level_pressure(self):
        s = get_atmosphere_from_altitude(0.0)
        self.assertTrue(math.isclose(s.pressure_Pa, 101325.0, rel_tol=1e-6))

    def test_sea_level_density(self):
        s = get_atmosphere_from_altitude(0.0)
        self.assertTrue(math.isclose(s.density_kg_m3, RHO0_KG_M3, rel_tol=2e-3))

    def test_sea_level_speed_of_sound(self):
        s = get_atmosphere_from_altitude(0.0)
        self.assertTrue(math.isclose(s.speed_of_sound_mps, 340.3, rel_tol=2e-3))

    def test_sea_level_internal_consistency(self):
        _assert_internal_consistency(self, get_atmosphere_from_altitude(0.0))

    def test_tropopause_temperature(self):
        s = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M)
        self.assertTrue(math.isclose(s.temperature_K, 216.65, rel_tol=1e-4))

    def test_tropopause_internal_consistency(self):
        _assert_internal_consistency(self, get_atmosphere_from_altitude(TROPOPAUSE_ALT_M))

    def test_stratosphere_sample_internal_consistency(self):
        _assert_internal_consistency(self, get_atmosphere_from_altitude(15000.0))

    def test_top_of_modeled_range_internal_consistency(self):
        _assert_internal_consistency(self, get_atmosphere_from_altitude(MAX_SUPPORTED_ALTITUDE_M))

    def test_integer_altitude_accepted(self):
        """int inputs should be accepted and produce the same result as float."""
        s_int = get_atmosphere_from_altitude(1000)
        s_flt = get_atmosphere_from_altitude(1000.0)
        self.assertEqual(s_int.temperature_K, s_flt.temperature_K)


# =============================================================================
# Layer-boundary continuity at 11,000 m
# =============================================================================

class TestTropopauseContinuity(unittest.TestCase):
    """All four state variables must be continuous across the 11 km boundary."""

    def setUp(self):
        self.below = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M - 0.01)
        self.above = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M + 0.01)

    def test_pressure_continuity(self):
        self.assertTrue(
            math.isclose(self.below.pressure_Pa, self.above.pressure_Pa, rel_tol=1e-4)
        )

    def test_temperature_continuity(self):
        self.assertTrue(
            math.isclose(self.below.temperature_K, self.above.temperature_K, rel_tol=1e-4)
        )

    def test_density_continuity(self):
        self.assertTrue(
            math.isclose(self.below.density_kg_m3, self.above.density_kg_m3, rel_tol=1e-4)
        )

    def test_speed_of_sound_continuity(self):
        self.assertTrue(
            math.isclose(
                self.below.speed_of_sound_mps, self.above.speed_of_sound_mps, rel_tol=1e-4
            )
        )


# =============================================================================
# Monotonicity
# =============================================================================

class TestMonotonicity(unittest.TestCase):
    """Physical quantities must be strictly monotone with altitude."""

    def setUp(self):
        altitudes = [0.0, 5000.0, TROPOPAUSE_ALT_M, 15000.0, MAX_SUPPORTED_ALTITUDE_M]
        self.states = [get_atmosphere_from_altitude(h) for h in altitudes]

    def test_pressure_strictly_decreasing(self):
        pressures = [s.pressure_Pa for s in self.states]
        self.assertTrue(all(a > b for a, b in zip(pressures, pressures[1:])))

    def test_density_strictly_decreasing(self):
        densities = [s.density_kg_m3 for s in self.states]
        self.assertTrue(all(a > b for a, b in zip(densities, densities[1:])))

    def test_temperature_strictly_decreasing_in_troposphere(self):
        """Temperature decreases only in the troposphere (states[:3])."""
        temperatures = [s.temperature_K for s in self.states[:3]]
        self.assertTrue(all(a > b for a, b in zip(temperatures, temperatures[1:])))

    def test_temperature_constant_in_isothermal_layer(self):
        """Temperature is constant above the tropopause."""
        tropo = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M)
        strat_low = get_atmosphere_from_altitude(15000.0)
        strat_top = get_atmosphere_from_altitude(MAX_SUPPORTED_ALTITUDE_M)
        self.assertTrue(
            math.isclose(tropo.temperature_K, strat_low.temperature_K, rel_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(strat_low.temperature_K, strat_top.temperature_K, rel_tol=1e-9)
        )


# =============================================================================
# NED conversion
# =============================================================================

class TestNEDConversion(unittest.TestCase):
    """get_atmosphere_from_ned_z(z) must exactly match get_atmosphere_from_altitude(-z)."""

    def _assert_states_equal(self, ned_state, alt_state):
        for field in ("temperature_K", "pressure_Pa", "density_kg_m3", "speed_of_sound_mps"):
            self.assertTrue(
                math.isclose(
                    getattr(ned_state, field),
                    getattr(alt_state, field),
                    rel_tol=1e-12,
                ),
                f"{field} mismatch: ned={getattr(ned_state, field)}, "
                f"alt={getattr(alt_state, field)}",
            )

    def test_ned_z_negative_1000(self):
        """z_ned = -1000 m  ->  altitude = 1000 m."""
        self._assert_states_equal(
            get_atmosphere_from_ned_z(-1000.0),
            get_atmosphere_from_altitude(1000.0),
        )

    def test_ned_z_zero(self):
        """z_ned = 0  ->  altitude = 0 (sea level)."""
        self._assert_states_equal(
            get_atmosphere_from_ned_z(0.0),
            get_atmosphere_from_altitude(0.0),
        )

    def test_ned_z_negative_at_tropopause(self):
        """z_ned = -TROPOPAUSE_ALT_M  ->  altitude = TROPOPAUSE_ALT_M."""
        self._assert_states_equal(
            get_atmosphere_from_ned_z(-TROPOPAUSE_ALT_M),
            get_atmosphere_from_altitude(TROPOPAUSE_ALT_M),
        )

    def test_ned_z_integer_accepted(self):
        ned = get_atmosphere_from_ned_z(-5000)
        alt = get_atmosphere_from_altitude(5000.0)
        self.assertEqual(ned.temperature_K, alt.temperature_K)


# =============================================================================
# Invalid-input rejection -- get_atmosphere_from_altitude
# =============================================================================

class TestInvalidAltitude(unittest.TestCase):
    """_validate_altitude_m must reject every invalid input with the right exception."""

    # --- NaN ---
    def test_nan_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(float("nan"))

    # --- Infinity ---
    def test_positive_infinity_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(float("inf"))

    def test_negative_infinity_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(float("-inf"))

    # --- Out-of-range: below minimum ---
    def test_negative_altitude_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(-1.0)

    def test_minus_one_millimetre_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(-0.001)

    # --- Out-of-range: above maximum ---
    def test_above_20000_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(20000.1)

    def test_way_above_max_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_altitude(100_000.0)

    # --- Wrong types ---
    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_altitude("1000")  # type: ignore[arg-type]

    def test_bool_raises_type_error(self):
        """bool is a subclass of int; it must still be rejected."""
        with self.assertRaises(TypeError):
            get_atmosphere_from_altitude(True)  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_altitude(None)  # type: ignore[arg-type]

    def test_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_altitude([1000.0])  # type: ignore[arg-type]


# =============================================================================
# Invalid-input rejection -- get_atmosphere_from_ned_z
# =============================================================================

class TestInvalidNEDZ(unittest.TestCase):
    """get_atmosphere_from_ned_z must propagate type/value errors correctly."""

    def test_nan_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_ned_z(float("nan"))

    def test_positive_infinity_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_ned_z(float("inf"))

    def test_negative_infinity_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_atmosphere_from_ned_z(float("-inf"))

    def test_z_positive_beyond_max_raises_value_error(self):
        """z_ned > 0 means below MSL -> altitude < 0 -> out of range."""
        with self.assertRaises(ValueError):
            get_atmosphere_from_ned_z(1.0)  # altitude = -1 m

    def test_z_very_negative_raises_value_error(self):
        """altitude would exceed MAX_SUPPORTED_ALTITUDE_M."""
        with self.assertRaises(ValueError):
            get_atmosphere_from_ned_z(-20000.1)

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_ned_z("0")  # type: ignore[arg-type]

    def test_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_ned_z(False)  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_atmosphere_from_ned_z(None)  # type: ignore[arg-type]


# =============================================================================
# Boundary values that MUST succeed (fence-post)
# =============================================================================

class TestBoundaryValues(unittest.TestCase):
    """Exact boundary altitudes must be accepted without error."""

    def test_minimum_altitude_accepted(self):
        s = get_atmosphere_from_altitude(MIN_SUPPORTED_ALTITUDE_M)
        self.assertGreater(s.pressure_Pa, 0.0)

    def test_maximum_altitude_accepted(self):
        s = get_atmosphere_from_altitude(MAX_SUPPORTED_ALTITUDE_M)
        self.assertGreater(s.pressure_Pa, 0.0)

    def test_exact_tropopause_accepted(self):
        s = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M)
        self.assertGreater(s.pressure_Pa, 0.0)


if __name__ == "__main__":
    unittest.main()
