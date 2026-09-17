"""Minimum tests for the redesigned NASA AirSTAR T-2 dashboard (dashboard/app.py).

Scope is intentionally narrow, per the redesign task:
  - the aircraft visualization exists and is the dashboard's main visual
  - the time slider's selected index maps to the *exact* stored sample
    (no interpolation / smoothing / generated values)
  - the qualitative pitch-rate visualization is driven directly by the
    stored ``delta_q_rads`` column, is bounded, and is never labeled
    "aircraft attitude"
  - existing scenario-metadata association / mismatch-protection /
    missing-data behavior is unchanged

These tests do not touch simulation, physics, estimator, health, scenario,
or integration logic; they exercise only the dashboard's data-preparation
and HTML-rendering functions using small, synthetic CSV fixtures.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import app as dash  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_run_csv(path: Path, rows, include_delta_e: bool = True) -> None:
    """Write a minimal, schema-valid Simulator CSV result for testing."""

    fieldnames = ["time_s", "delta_w_mps", "delta_q_rads"]
    if include_delta_e:
        fieldnames.append("delta_e_rad")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def sample_rows():
    # Deliberately non-trivial, non-monotonic-looking floats so any
    # accidental rounding/interpolation would be caught by exact comparison.
    return [
        {"time_s": 0.0, "delta_w_mps": 0.123456, "delta_q_rads": 0.0, "delta_e_rad": 0.01},
        {"time_s": 0.05, "delta_w_mps": 0.987654, "delta_q_rads": 0.271828, "delta_e_rad": 0.02},
        {"time_s": 0.10, "delta_w_mps": -0.55555, "delta_q_rads": -0.314159, "delta_e_rad": -0.015},
        {"time_s": 0.15, "delta_w_mps": 1.414213, "delta_q_rads": 0.05, "delta_e_rad": 0.0},
    ]


@pytest.fixture
def bundle(tmp_path, sample_rows):
    csv_path = tmp_path / "unit_test_run.csv"
    _write_run_csv(csv_path, sample_rows, include_delta_e=True)
    run = dash.load_run(csv_path)
    return dash.prepare_dashboard_data(run, scenario=None, integrated_results=None)


@pytest.fixture
def bundle_no_delta_e(tmp_path, sample_rows):
    csv_path = tmp_path / "unit_test_run_no_de.csv"
    rows = [{k: v for k, v in r.items() if k != "delta_e_rad"} for r in sample_rows]
    _write_run_csv(csv_path, rows, include_delta_e=False)
    run = dash.load_run(csv_path)
    return dash.prepare_dashboard_data(run, scenario=None, integrated_results=None)


def _rendered_page(bundle_obj, runs=(), scenarios=(), **kwargs):
    return dash.render_dashboard_page(
        bundle_obj,
        runs=runs,
        scenarios=scenarios,
        scenario_registry_error=None,
        scenario_user_supplied=False,
        **kwargs,
    )


def _extract_payload(html_text: str) -> dict:
    match = re.search(r"var PAYLOAD = (\{.*?\});\n", html_text, re.S)
    assert match, "Time-slider PAYLOAD JSON was not found in the rendered page."
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# 1. Aircraft visualization exists
# ---------------------------------------------------------------------------

def test_aircraft_visualization_present_and_is_main_visual(bundle):
    page = _rendered_page(bundle)

    assert '<h2>Aircraft</h2>' in page
    assert 'class="aircraft-hero-svg"' in page
    # The aircraft SVG must be present exactly once and be the large,
    # dedicated visual (not the old small 120x56 header icon).
    assert page.count('class="aircraft-hero-svg"') == 1
    assert 'viewBox="0 0 640 260"' in page
    assert 'width="120" height="56"' not in page  # old small header schematic is gone




# ---------------------------------------------------------------------------
# 2. Slider index maps to the exact stored sample (no interpolation)
# ---------------------------------------------------------------------------

def test_slider_payload_matches_exact_stored_samples(bundle, sample_rows):
    page = _rendered_page(bundle)
    payload = _extract_payload(page)
    samples = payload["samples"]

    assert len(samples) == len(sample_rows)
    for expected, actual in zip(sample_rows, samples):
        assert actual["time_s"] == pytest.approx(expected["time_s"], abs=0.0)
        assert actual["delta_w_mps"] == pytest.approx(expected["delta_w_mps"], abs=0.0)
        assert actual["delta_q_rads"] == pytest.approx(expected["delta_q_rads"], abs=0.0)
        assert actual["delta_e_rad"] == pytest.approx(expected["delta_e_rad"], abs=0.0)

    # Slider bounds must exactly span the stored samples: 0 .. len(rows)-1.
    assert f'max="{len(sample_rows) - 1}"' in page
    assert 'id="time-slider"' in page


def test_slider_readout_ids_present_for_all_required_fields(bundle):
    page = _rendered_page(bundle)
    for element_id in ("ts-time", "ts-dw", "ts-dq", "ts-de", "time-index-display"):
        assert f'id="{element_id}"' in page


# ---------------------------------------------------------------------------
# 3. Qualitative pitch-rate visualization driven by stored delta_q_rads
# ---------------------------------------------------------------------------

def test_pitch_rate_label_is_exact_and_not_called_attitude(bundle):
    page = _rendered_page(bundle)
    assert "Qualitative pitch-rate visualization" in page
    # The only permitted appearance of "attitude" is the explicit negation.
    lowered = page.lower()
    assert "aircraft attitude" in lowered
    assert lowered.count("aircraft attitude") == lowered.count("not aircraft attitude")


def test_pitch_rate_gauge_scale_derived_from_stored_delta_q_rads(bundle, sample_rows):
    page = _rendered_page(bundle)
    payload = _extract_payload(page)

    expected_q_scale = max(abs(r["delta_q_rads"]) for r in sample_rows)
    assert payload["q_scale"] == pytest.approx(expected_q_scale)

    # The gauge's rotating needle element must exist and be JS-driven.
    assert 'id="q-indicator"' in page
    assert 'id="q-indicator-raw"' in page
    assert "PAYLOAD.q_scale" in page
    assert "Math.tanh" in page  # bounded, saturating transform


def test_pitch_rate_gauge_is_bounded_for_extreme_values():
    # Mirror the client-side transform used in render_dashboard_page's JS to
    # confirm it is bounded and stable for any finite stored delta_q_rads,
    # independent of the run's own scale.
    max_angle_deg = 75.0
    for q, q_scale in [(1e12, 0.4), (-1e12, 0.4), (0.0, 1.0), (1e-9, 1e-9), (-3.0, 3.0)]:
        normalized = math.tanh(q / q_scale)
        angle = normalized * max_angle_deg
        assert -max_angle_deg <= angle <= max_angle_deg
        assert math.isfinite(angle)


def test_aircraft_visualization_and_slider_use_same_stored_sample(bundle):
    page = _rendered_page(bundle)
    # Exactly one per-index sample lookup must drive both the numeric
    # time-slider readouts and the qualitative pitch-rate visualization --
    # i.e. the aircraft graphic is not fed from a separate/duplicated data
    # source or a different index than the slider's own readouts.
    assert page.count("PAYLOAD.samples[i]") == 1
    assert "qIndicator.style.transform" in page
    assert "tsDq.textContent = fmt(d.delta_q_rads)" in page
    assert "qIndicatorRaw.textContent = fmt(d.delta_q_rads)" in page


def test_pitch_rate_visualization_does_not_fabricate_pitch_angle(bundle):
    page = _rendered_page(bundle)
    assert "pitch angle" not in page.lower() or "no pitch angle is computed" in page.lower()
    assert "delta_theta" not in page
    assert "pitch_angle" not in page


# ---------------------------------------------------------------------------
# 4. Existing scenario metadata / mismatch-protection / missing-data
#    behavior is unchanged
# ---------------------------------------------------------------------------

class _FakeScenario:
    def __init__(self, name):
        self.name = name
        self.description = "unit test scenario"
        self.classification = "TEST"
        self.reference_condition_provenance = "PROJECT_DEFINED"
        self.control_schedule_provenance = "PROJECT_DEFINED"
        self.source = "unit-test"
        self.flight_condition = "FC1"
        self.assumptions = ()
        self.notes = ""


def test_scenario_metadata_mismatch_protection_unchanged(tmp_path, sample_rows):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    run_name = "known_run"
    _write_run_csv(data_dir / f"{run_name}.csv", sample_rows, include_delta_e=True)
    (data_dir / f"{run_name}.meta.json").write_text(
        json.dumps({"scenario_name": "a_different_known_scenario"})
    )

    def fake_registry():
        return [_FakeScenario(run_name), _FakeScenario("a_different_known_scenario")]

    application = dash.DashboardApplication.__new__(dash.DashboardApplication)
    application.repository = dash.RunRepository(data_dir)
    application.default_run = None
    application.default_scenario = None
    application.scenarios = dash.load_scenarios(registry_func=fake_registry)
    application.scenario_registry_error = None

    page = application.render({})
    assert "Scenario metadata mismatch" in page
    assert "SCENARIO: Not associated" in page


def test_missing_delta_e_shows_not_available_and_is_not_fabricated(bundle_no_delta_e):
    page = _rendered_page(bundle_no_delta_e)
    payload = _extract_payload(page)

    assert "delta_e_rad" not in payload["samples"][0]
    assert "NOT AVAILABLE IN SAVED RESULT" in page


def test_dashboard_error_still_raised_for_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "bad_run.csv"
    bad_csv.write_text("time_s,delta_w_mps\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(dash.ResultSchemaError):
        dash.load_run(bad_csv)
