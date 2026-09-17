"""
tests/test_sensor.py

Tests for sensors/sensor.py.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sensors.sensor_model import (
    SensorModel,
    SensorSuiteParameters,
    SensorChannelNoiseModel,
    default_sensor_suite,
)


def test_zero_bias_zero_noise_equals_truth():
    zero_channel = SensorChannelNoiseModel(
        name="zero_test", units="unit", bias=0.0, noise_std=0.0
    )
    suite_zero = SensorSuiteParameters(
        w_sensor=zero_channel, rate_gyro_q=zero_channel, accelerometer_az=zero_channel
    )
    model_zero = SensorModel(suite=suite_zero, seed=1234)

    result = model_zero.measure(delta_w=2.0, delta_q=0.1, perturbation_az=-0.15)
    assert result["measured"]["delta_w_mps"] == 2.0
    assert result["measured"]["delta_q_rads"] == 0.1
    assert result["measured"]["perturbation_az_mps2"] == -0.15
    assert result["truth"]["delta_w_mps"] == 2.0
    assert result["truth"]["delta_q_rads"] == 0.1
    assert result["truth"]["perturbation_az_mps2"] == -0.15


def test_bias_only_hand_calculation():
    zero_channel = SensorChannelNoiseModel(
        name="zero_test", units="unit", bias=0.0, noise_std=0.0
    )
    biased_q = SensorChannelNoiseModel(
        name="rate_gyro_q", units="rad/s", bias=0.02, noise_std=0.0
    )
    suite_bias = SensorSuiteParameters(
        w_sensor=zero_channel, rate_gyro_q=biased_q, accelerometer_az=zero_channel
    )
    model_bias = SensorModel(suite=suite_bias, seed=99)
    r = model_bias.measure(delta_w=0.0, delta_q=0.1, perturbation_az=0.0)
    assert math.isclose(r["measured"]["delta_q_rads"], 0.12, rel_tol=1e-12)


def test_fixed_seed_reproducibility():
    noisy_channel = SensorChannelNoiseModel(
        name="noisy", units="unit", bias=0.5, noise_std=0.2
    )
    suite_noisy = SensorSuiteParameters(
        w_sensor=noisy_channel, rate_gyro_q=noisy_channel, accelerometer_az=noisy_channel
    )
    model_a = SensorModel(suite=suite_noisy, seed=42)
    model_b = SensorModel(suite=suite_noisy, seed=42)
    for _ in range(20):
        ra = model_a.measure(delta_w=1.0, delta_q=1.0, perturbation_az=1.0)
        rb = model_b.measure(delta_w=1.0, delta_q=1.0, perturbation_az=1.0)
        assert ra["measured"] == rb["measured"]


def test_noise_statistics_converge():
    zero_channel = SensorChannelNoiseModel(
        name="zero_test", units="unit", bias=0.0, noise_std=0.0
    )
    stats_channel = SensorChannelNoiseModel(
        name="stats", units="unit", bias=0.3, noise_std=0.05
    )
    suite_stats = SensorSuiteParameters(
        w_sensor=stats_channel, rate_gyro_q=zero_channel, accelerometer_az=zero_channel
    )
    model_stats = SensorModel(suite=suite_stats, seed=7)
    n_samples = 20000
    samples = [
        model_stats.measure(delta_w=0.0, delta_q=0.0, perturbation_az=0.0)["measured"]["delta_w_mps"]
        for _ in range(n_samples)
    ]
    sample_mean = sum(samples) / n_samples
    sample_var = sum((s - sample_mean) ** 2 for s in samples) / n_samples
    sample_std = math.sqrt(sample_var)

    standard_error = stats_channel.noise_std / math.sqrt(n_samples)
    assert abs(sample_mean - stats_channel.bias) < 8 * standard_error
    assert math.isclose(sample_std, stats_channel.noise_std, rel_tol=0.1)


def test_input_validation():
    zero_channel = SensorChannelNoiseModel(
        name="zero_test", units="unit", bias=0.0, noise_std=0.0
    )
    suite_zero = SensorSuiteParameters(
        w_sensor=zero_channel, rate_gyro_q=zero_channel, accelerometer_az=zero_channel
    )
    model_zero = SensorModel(suite=suite_zero, seed=1234)

    with pytest.raises(ValueError):
        model_zero.measure(delta_w=float('nan'), delta_q=0.0, perturbation_az=0.0)

    with pytest.raises(ValueError):
        SensorChannelNoiseModel(name="bad", units="unit", bias=0.0, noise_std=-1.0)

    with pytest.raises(ValueError):
        SensorChannelNoiseModel(name="bad", units="unit", bias=float('inf'), noise_std=0.0)

    with pytest.raises(TypeError):
        SensorModel(suite="not_a_suite")  # type: ignore

    with pytest.raises(TypeError):
        SensorModel.from_config("not_a_dict")  # type: ignore


def test_config_driven_construction():
    cfg_with_sensors = {
        "sensors": {
            "rate_gyro_q": {"bias": 0.01, "noise_std": 0.0},
        }
    }
    model_cfg = SensorModel.from_config(cfg_with_sensors, seed=5)
    assert model_cfg.suite.rate_gyro_q.bias == 0.01
    assert model_cfg.suite.rate_gyro_q.model_origin == "PROJECT_DEFINED_CONFIG_SOURCED"
    assert model_cfg.suite.w_sensor.model_origin == "PROJECT_DEFINED_PLACEHOLDER"

    cfg_without_sensors = {"aircraft": {}}
    model_no_cfg = SensorModel.from_config(cfg_without_sensors, seed=5)
    default_suite = default_sensor_suite()
    assert model_no_cfg.suite.w_sensor.bias == default_suite.w_sensor.bias
    assert model_no_cfg.suite.w_sensor.noise_std == default_suite.w_sensor.noise_std


def test_overflow_measurement_rejected():
    zero_channel = SensorChannelNoiseModel(
        name="zero_test", units="unit", bias=0.0, noise_std=0.0
    )
    overflow_channel = SensorChannelNoiseModel(
        name="overflow", units="unit", bias=1e308, noise_std=0.0
    )
    suite_overflow = SensorSuiteParameters(
        w_sensor=overflow_channel, rate_gyro_q=zero_channel, accelerometer_az=zero_channel
    )
    model_overflow = SensorModel(suite=suite_overflow, seed=1)
    
    with pytest.raises(ValueError):
        model_overflow.measure(delta_w=1e308, delta_q=0.0, perturbation_az=0.0)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
