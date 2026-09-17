"""
tests/test_propulsion.py

Test suite for the NASA AirSTAR T-2 propulsion subsystem.

Verifies:
    - Two-engine architecture and JetCat P70 identity
    - 16 lbf / 32 lbf SI conversions
    - Rated thrust is NOT used as constant flight thrust
    - Input validation (RPM, throttle, airspeed, lambda, pump voltage)
    - NaN / infinity rejection
    - Independent left/right engine state
    - MODEL_NOT_IDENTIFIED when no thrust model is available
    - Plugin thrust model with health-factor application
    - Plugin error rejection (NaN, inf, negative)
    - Provenance tracking
    - No fabricated coefficients or maps
    - No mutable config exposure
    - Aircraft identity guard
"""

import unittest
import math
import os
import yaml

from models.propulsion import (
    PropulsionModel,
    PropulsionConfigError,
    EngineState,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')


def _make_model():
    """Helper: create a PropulsionModel from the project config file."""
    return PropulsionModel.from_config_file(CONFIG_PATH)


def _load_config():
    """Helper: load the raw config dict for mutation-based tests."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


# =========================================================================
# Initialization and identity
# =========================================================================


class TestInitialization(unittest.TestCase):

    def test_two_engines_exist(self):
        model = _make_model()
        self.assertIsNotNone(model.left_engine)
        self.assertIsNotNone(model.right_engine)
        self.assertEqual(model.num_engines, 2)

    def test_from_config_dict(self):
        """from_config() accepts a pre-loaded dict."""
        config = _load_config()
        model = PropulsionModel.from_config(config)
        self.assertEqual(model.num_engines, 2)
        self.assertEqual(model.rated_thrust_lbf, 16)

    def test_no_mutable_config_stored(self):
        """The model must NOT expose the complete mutable config dict."""
        model = _make_model()
        self.assertFalse(hasattr(model, 'config'))

    def test_identity_rejects_non_t2(self):
        """Non-T-2 aircraft configurations must be rejected."""
        config = _load_config()
        config['configuration_identity']['aircraft'] = "Boeing 737"
        with self.assertRaises(PropulsionConfigError):
            PropulsionModel.from_config(config)

    def test_identity_rejects_wrong_manufacturer(self):
        config = _load_config()
        config['configuration_identity']['engine_manufacturer'] = "Pratt"
        with self.assertRaises(PropulsionConfigError):
            PropulsionModel.from_config(config)

    def test_identity_rejects_wrong_model(self):
        config = _load_config()
        config['configuration_identity']['engine_model'] = "P200"
        with self.assertRaises(PropulsionConfigError):
            PropulsionModel.from_config(config)


# =========================================================================
# Thrust rating conversions
# =========================================================================


class TestThrustRatings(unittest.TestCase):

    def test_rated_thrust_lbf_is_16(self):
        model = _make_model()
        self.assertEqual(model.rated_thrust_lbf, 16)

    def test_16_lbf_to_newtons(self):
        """16 lbf -> 71.171545844168 N (exact conversion)."""
        model = _make_model()
        self.assertTrue(
            math.isclose(model.rated_thrust_N, 71.171545844168, rel_tol=1e-9)
        )

    def test_32_lbf_to_newtons(self):
        """32 lbf -> 142.343091688336 N (exact conversion)."""
        model = _make_model()
        self.assertTrue(
            math.isclose(
                model.rated_thrust_N * 2, 142.343091688336, rel_tol=1e-9
            )
        )

    def test_rated_thrust_not_used_as_constant_flight_thrust(self):
        """Without a plugin, rated thrust must NOT appear as output."""
        model = _make_model()
        state = model.calculate_thrust(
            left_rpm=50000, left_throttle=1.0, left_lambda=0.0,
            right_rpm=50000, right_throttle=1.0, right_lambda=0.0,
            airspeed=20.0,
        )
        # Must NOT silently return the rated value
        self.assertTrue(math.isnan(state.left_thrust_N))
        self.assertEqual(state.model_status, "MODEL_NOT_IDENTIFIED")


# =========================================================================
# Input validation — RPM
# =========================================================================


class TestInputValidationRPM(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_negative_rpm_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'left_rpm': -100})

    def test_nan_rpm_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'left_rpm': float('nan')})

    def test_inf_rpm_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'right_rpm': float('inf')})

    def test_neg_inf_rpm_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'left_rpm': float('-inf')})


# =========================================================================
# Input validation — throttle
# =========================================================================


class TestInputValidationThrottle(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_throttle_above_1_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'left_throttle': 1.5})

    def test_throttle_below_0_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'right_throttle': -0.1})

    def test_nan_throttle_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'left_throttle': float('nan')}
            )

    def test_inf_throttle_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'right_throttle': float('inf')}
            )


# =========================================================================
# Input validation — airspeed
# =========================================================================


class TestInputValidationAirspeed(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_negative_airspeed_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'airspeed': -1.0})

    def test_nan_airspeed_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'airspeed': float('nan')}
            )

    def test_inf_airspeed_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'airspeed': float('inf')}
            )

    def test_neg_inf_airspeed_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'airspeed': float('-inf')}
            )


# =========================================================================
# Input validation — health factor (lambda)
# =========================================================================


class TestInputValidationLambda(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_negative_lambda_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'left_lambda': -0.1})

    def test_lambda_above_1_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(**{**self.base, 'right_lambda': 1.1})

    def test_nan_lambda_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'left_lambda': float('nan')}
            )

    def test_inf_lambda_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'right_lambda': float('inf')}
            )


# =========================================================================
# Input validation — pump voltage
# =========================================================================


class TestInputValidationPumpVoltage(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_negative_pump_voltage_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'left_pump_voltage': -1.0}
            )

    def test_nan_pump_voltage_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'right_pump_voltage': float('nan')}
            )

    def test_inf_pump_voltage_rejected(self):
        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **{**self.base, 'left_pump_voltage': float('inf')}
            )


# =========================================================================
# Engine independence
# =========================================================================


class TestEngineIndependence(unittest.TestCase):

    def test_left_right_independent_state(self):
        model = _make_model()
        state = model.calculate_thrust(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.1,
            right_rpm=60000, right_throttle=0.8, right_lambda=0.2,
            airspeed=20.0,
            left_pump_voltage=3.0, right_pump_voltage=4.5,
        )
        self.assertEqual(state.left_engine.rpm, 50000)
        self.assertEqual(state.right_engine.rpm, 60000)
        self.assertEqual(state.left_engine.throttle, 0.5)
        self.assertEqual(state.right_engine.throttle, 0.8)
        self.assertEqual(state.left_engine.lambda_factor, 0.1)
        self.assertEqual(state.right_engine.lambda_factor, 0.2)
        self.assertEqual(state.left_engine.pump_voltage, 3.0)
        self.assertEqual(state.right_engine.pump_voltage, 4.5)


# =========================================================================
# MODEL_NOT_IDENTIFIED behavior
# =========================================================================


class TestModelNotIdentified(unittest.TestCase):

    def test_explicit_status_when_no_plugin(self):
        model = _make_model()
        state = model.calculate_thrust(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )
        self.assertEqual(state.model_status, "MODEL_NOT_IDENTIFIED")
        self.assertTrue(math.isnan(state.left_thrust_N))
        self.assertTrue(math.isnan(state.right_thrust_N))
        self.assertTrue(math.isnan(state.total_thrust_N))


# =========================================================================
# Plugin thrust model — valid and invalid
# =========================================================================


class TestPluginThrustModel(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.base = dict(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )

    def test_valid_plugin(self):
        def constant_10(engine_state, airspeed):
            return 10.0

        state = self.model.calculate_thrust(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.1,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.2,
            airspeed=20.0,
            thrust_model_plugin=constant_10,
        )
        self.assertEqual(state.model_status, "PLUGIN_MODEL_EVALUATED")
        # left: 10 * (1 - 0.1) = 9.0
        self.assertTrue(math.isclose(state.left_thrust_N, 9.0))
        # right: 10 * (1 - 0.2) = 8.0
        self.assertTrue(math.isclose(state.right_thrust_N, 8.0))
        # total: 9 + 8 = 17
        self.assertTrue(math.isclose(state.total_thrust_N, 17.0))

    def test_plugin_returning_nan_rejected(self):
        def bad_nan(engine_state, airspeed):
            return float('nan')

        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **self.base, thrust_model_plugin=bad_nan,
            )

    def test_plugin_returning_inf_rejected(self):
        def bad_inf(engine_state, airspeed):
            return float('inf')

        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **self.base, thrust_model_plugin=bad_inf,
            )

    def test_plugin_returning_negative_rejected(self):
        def bad_neg(engine_state, airspeed):
            return -5.0

        with self.assertRaises(ValueError):
            self.model.calculate_thrust(
                **self.base, thrust_model_plugin=bad_neg,
            )


# =========================================================================
# Provenance and fabrication guard
# =========================================================================


class TestProvenance(unittest.TestCase):

    def test_provenance_tracked(self):
        model = _make_model()
        self.assertIn("thrust_rating_per_engine", model.provenance)
        self.assertEqual(
            model.provenance["thrust_rating_per_engine"]["value_lbf"], 16
        )
        self.assertEqual(
            model.provenance["thrust_rating_per_engine"]["status"],
            "RATED_HEADLINE_VALUE_NOT_CONSTANT_FLIGHT_THRUST",
        )

    def test_no_fabricated_coefficients(self):
        """If no coefficients are fabricated, the model must return
        MODEL_NOT_IDENTIFIED without a plugin (i.e. it does not
        silently produce a thrust value from invented parameters).
        """
        model = _make_model()
        state = model.calculate_thrust(
            left_rpm=50000, left_throttle=0.5, left_lambda=0.0,
            right_rpm=50000, right_throttle=0.5, right_lambda=0.0,
            airspeed=20.0,
        )
        self.assertEqual(state.model_status, "MODEL_NOT_IDENTIFIED")
        self.assertTrue(math.isnan(state.left_thrust_N))


if __name__ == '__main__':
    unittest.main()
