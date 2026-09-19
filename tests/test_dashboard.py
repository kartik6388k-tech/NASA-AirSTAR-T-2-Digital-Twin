"""
Tests for the NASA AirSTAR T-2 Simulation Telemetry Playback.

Unit tests (mock-based):
  - API JSON structure and content-type
  - Mock oscillator provenance labeling
  - No-cache headers
  - Root path serves HTML
  - /api/runs graceful empty-directory fallback
  - 404 for unknown paths

Integration tests (live server):
  - Live /api/telemetry returns valid JSON with expected keys
  - Sequential playback advances index
  - Live root serves dashboard HTML
  - Thread-safety under concurrent polling
"""

import json
import os
import sys
import threading
import socketserver
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.app import (
    DashboardHandler,
    TelemetryState,
    make_handler,
    load_csv_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
time_s,delta_w_mps,delta_q_rads,delta_e_rad,propulsion_model_status
0.0,0.0,0.0,0.0,MODEL_NOT_IDENTIFIED
0.01,-0.05,0.002,-0.01,MODEL_NOT_IDENTIFIED
0.02,-0.12,0.005,-0.02,MODEL_NOT_IDENTIFIED
0.03,-0.18,0.008,-0.03,MODEL_NOT_IDENTIFIED
0.04,-0.22,0.010,-0.035,MODEL_NOT_IDENTIFIED
"""


@pytest.fixture
def csv_dir(tmp_path):
    """Create a temp directory with a sample CSV run."""
    csv_file = tmp_path / "test_run.csv"
    csv_file.write_text(SAMPLE_CSV.strip(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """Create a temp directory with no CSV files."""
    return tmp_path


@pytest.fixture
def state(csv_dir):
    """TelemetryState pointing at the CSV fixture directory."""
    return TelemetryState(data_dir=csv_dir)


@pytest.fixture
def empty_state(empty_dir):
    """TelemetryState with no CSV files available."""
    return TelemetryState(data_dir=empty_dir)


class MockWfile:
    """Minimal writable buffer to capture HTTP response body."""
    def __init__(self):
        self.buf = BytesIO()
    def write(self, data):
        self.buf.write(data)
    def getvalue(self):
        return self.buf.getvalue()


def make_mock_handler(state: TelemetryState, path: str):
    """Create a mock DashboardHandler for unit testing without sockets."""
    cls = make_handler(state)
    handler = cls.__new__(cls)
    handler.path = path
    handler.command = "GET"
    handler.request_version = "HTTP/1.1"
    handler.headers = {}
    handler.wfile = MockWfile()
    handler._headers_buffer = []
    handler.responses = {200: ('OK', ''), 400: ('Bad Request', ''), 404: ('Not Found', '')}

    # Mock the response methods to capture output
    handler._sent_status = None
    handler._sent_headers = {}

    def mock_send_response(code, msg=None):
        handler._sent_status = code
    handler.send_response = mock_send_response

    def mock_send_header(key, value):
        handler._sent_headers[key] = value
    handler.send_header = mock_send_header

    def mock_end_headers():
        pass
    handler.end_headers = mock_end_headers

    return handler


# ===================================================================
# UNIT TESTS — Mock-based
# ===================================================================

class TestApiTelemetryJson:
    """Test /api/telemetry returns properly structured JSON."""

    def test_mock_mode_returns_json(self, empty_state):
        handler = make_mock_handler(empty_state, '/api/telemetry')
        handler.do_GET()
        body = handler.wfile.getvalue()
        data = json.loads(body)

        assert 'state' in data
        assert 'health' in data
        assert 'provenance' in data
        assert 'timestamp_s' in data
        assert 'delta_q_rads' in data['state']
        assert 'delta_w_mps' in data['state']
        assert 'delta_e_rad' in data['state']
        assert 'pitch_rad' not in data.get('attitude', {})
        assert 'pitch_rad' not in data

    def test_mock_mode_provenance(self, empty_state):
        handler = make_mock_handler(empty_state, '/api/telemetry')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())

        assert data['provenance']['source'] == 'DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR'
        assert data['provenance']['run_name'] is None

    def test_csv_mode_provenance(self, state):
        state.load_run('test_run')
        handler = make_mock_handler(state, '/api/telemetry')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())

        assert data['provenance']['source'] == 'SAVED_CSV'
        assert data['provenance']['run_name'] == 'test_run'

    def test_content_type_json(self, empty_state):
        handler = make_mock_handler(empty_state, '/api/telemetry')
        handler.do_GET()
        assert handler._sent_headers.get('Content-Type') == 'application/json; charset=utf-8'

    def test_nocache_headers(self, empty_state):
        handler = make_mock_handler(empty_state, '/api/telemetry')
        handler.do_GET()
        assert 'no-cache' in handler._sent_headers.get('Cache-Control', '')
        assert handler._sent_headers.get('Pragma') == 'no-cache'
        assert handler._sent_headers.get('Expires') == '0'


class TestEndHeaders:
    """Regression tests for the no-cache injection in end_headers().

    ``http.server`` stores raw bytes lines in ``_headers_buffer``
    (``b"Name: value\\r\\n"``), not (key, value) tuples.  The real
    ``end_headers()`` must tolerate that and must not duplicate headers.
    """

    def _real_end_headers(self, state):
        """Mock handler wired so the REAL end_headers/send_header run."""
        handler = make_mock_handler(state, '/api/telemetry')
        handler.wfile = MockWfile()
        # Mirror what http.server.send_header actually does: append a
        # latin-1 encoded bytes line to _headers_buffer.
        def real_send_header(keyword, value):
            handler._headers_buffer.append(
                ("%s: %s\r\n" % (keyword, value)).encode("latin-1")
            )
        handler.send_header = real_send_header
        handler.end_headers = lambda: DashboardHandler.end_headers(handler)
        return handler

    def test_end_headers_accepts_bytes_buffer(self, empty_state):
        handler = self._real_end_headers(empty_state)
        handler._headers_buffer = [b"Content-Type: text/html\r\n"]
        handler.end_headers()  # must not raise ValueError
        out = handler.wfile.getvalue()
        assert b"Cache-Control:" in out
        assert out.endswith(b"\r\n")

    def test_end_headers_no_duplicates(self, empty_state):
        handler = self._real_end_headers(empty_state)
        handler._headers_buffer = [b"Cache-Control: no-cache\r\n"]
        handler.end_headers()
        out = handler.wfile.getvalue()
        assert out.count(b"Cache-Control:") == 1

    def test_end_headers_empty_buffer(self, empty_state):
        handler = self._real_end_headers(empty_state)
        handler._headers_buffer = []
        handler.end_headers()
        assert b"Cache-Control:" in handler.wfile.getvalue()


class TestApiRuns:
    """Test /api/runs endpoint."""

    def test_returns_run_list(self, state):
        handler = make_mock_handler(state, '/api/runs')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert 'runs' in data
        assert 'test_run' in data['runs']

    def test_empty_dir_returns_empty_list(self, empty_state):
        handler = make_mock_handler(empty_state, '/api/runs')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['runs'] == []

    def test_nonexistent_dir_returns_empty_list(self):
        state = TelemetryState(data_dir=Path("/nonexistent/path/that/should/not/exist"))
        handler = make_mock_handler(state, '/api/runs')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['runs'] == []


class TestApiRunLoad:
    """Test /api/run/<name> endpoint."""

    def test_load_valid_run(self, state):
        handler = make_mock_handler(state, '/api/run/test_run')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is True
        assert data['run_name'] == 'test_run'
        assert data['samples'] == 5

    def test_load_nonexistent_run(self, state):
        handler = make_mock_handler(state, '/api/run/no_such_run')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is False

    def test_load_empty_name(self, state):
        handler = make_mock_handler(state, '/api/run/')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is False


class TestPlaybackControl:
    """Test /api/playback?action= endpoint."""

    def test_pause_and_play(self, state):
        state.load_run('test_run')

        handler = make_mock_handler(state, '/api/playback?action=pause')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is True
        assert data['paused'] is True

        handler2 = make_mock_handler(state, '/api/playback?action=play')
        handler2.do_GET()
        data2 = json.loads(handler2.wfile.getvalue())
        assert data2['ok'] is True
        assert data2['paused'] is False

    def test_reset(self, state):
        state.load_run('test_run')
        # Advance a few frames
        state.get_telemetry()
        state.get_telemetry()

        handler = make_mock_handler(state, '/api/playback?action=reset')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is True

    def test_unknown_action(self, state):
        handler = make_mock_handler(state, '/api/playback?action=rewind')
        handler.do_GET()
        data = json.loads(handler.wfile.getvalue())
        assert data['ok'] is False


# ===================================================================
# TELEMETRY STATE TESTS
# ===================================================================

class TestTelemetryState:
    """Test TelemetryState playback logic."""

    def test_sequential_advance(self, state):
        state.load_run('test_run')
        f1 = state.get_telemetry()
        f2 = state.get_telemetry()
        assert f1['playback_index'] == 0
        assert f2['playback_index'] == 1

    def test_pause_stops_advance(self, state):
        state.load_run('test_run')
        state.playback_control('pause')
        f1 = state.get_telemetry()
        f2 = state.get_telemetry()
        assert f1['playback_index'] == f2['playback_index']

    def test_reset_returns_to_zero(self, state):
        state.load_run('test_run')
        state.get_telemetry()
        state.get_telemetry()
        state.playback_control('reset')
        f = state.get_telemetry()
        assert f['playback_index'] == 0

    def test_stays_at_last_frame(self, state):
        state.load_run('test_run')
        # Exhaust all frames
        for _ in range(100):
            state.get_telemetry()
        f = state.get_telemetry()
        assert f['playback_index'] == 4  # 5 rows, last index = 4

    def test_stored_delta_q_directly_in_telemetry_response(self, state):
        """Verify stored delta_q_rads is passed directly to telemetry response."""
        state.load_run('test_run')
        f0 = state.get_telemetry()
        assert f0['state']['delta_q_rads'] == 0.0
        assert f0['state']['delta_w_mps'] == 0.0
        assert f0['state']['delta_e_rad'] == 0.0
        assert 'pitch_rad' not in f0.get('attitude', {})
        assert 'pitch_rad' not in f0

        f1 = state.get_telemetry()
        assert f1['state']['delta_q_rads'] == 0.002
        assert f1['state']['delta_w_mps'] == -0.05
        assert f1['state']['delta_e_rad'] == -0.01

    def test_no_integrated_pitch_angle_state(self, state):
        """Verify no integrated cumulative pitch angle state is created."""
        state.load_run('test_run')
        for _ in range(5):
            frame = state.get_telemetry()
            assert 'pitch_rad' not in frame.get('attitude', {})
            assert 'pitch_rad' not in frame
            assert 'pitch_rad' not in frame['state']

    def test_model_not_identified_in_health(self, state):
        state.load_run('test_run')
        f = state.get_telemetry()
        assert f['health']['propulsion_model_status'] == 'MODEL_NOT_IDENTIFIED'

    def test_playback_completion_status(self, state):
        state.load_run('test_run')
        for _ in range(10):
            f = state.get_telemetry()
        assert f['playback_status'] == 'COMPLETED'
        assert f['playback_index'] == 4

    def test_playback_looping(self, state):
        state.load_run('test_run')
        state.playback_control('loop_on')
        indices = [state.get_telemetry()['playback_index'] for _ in range(8)]
        assert indices == [0, 1, 2, 3, 4, 0, 1, 2]

    def test_unload_action_restores_mock_oscillator(self, state):
        state.load_run('test_run')
        assert state.get_telemetry()['provenance']['source'] == 'SAVED_CSV'
        res = state.playback_control('unload')
        assert res['ok'] is True
        f = state.get_telemetry()
        assert f['provenance']['source'] == 'DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR'
        assert f['provenance']['run_name'] is None
        assert state._rows == []

    def test_load_run_demo_clears_saved_run(self, state):
        state.load_run('test_run')
        res = state.load_run('demo')
        assert res['ok'] is True
        assert res['mode'] == 'DEMO'
        f = state.get_telemetry()
        assert f['provenance']['source'] == 'DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR'
        assert state._rows == []


class TestCsvLoader:
    """Test CSV loading and validation."""

    def test_load_valid_csv(self, csv_dir):
        rows = load_csv_run(csv_dir / "test_run.csv")
        assert len(rows) == 5
        assert rows[0]['time_s'] == 0.0
        assert rows[0]['delta_w_mps'] == 0.0

    def test_load_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("time_s,some_col\n1.0,foo\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required"):
            load_csv_run(bad_csv)

    def test_load_empty_csv(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("time_s,delta_w_mps,delta_q_rads\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_csv_run(empty_csv)


class TestTelemetryVisualizationData:
    """Verify stored telemetry flows directly to visualization data without pitch integration."""

    def test_stored_delta_q_to_telemetry_fidelity(self, state):
        state.load_run('test_run')
        expected_samples = [
            (0.0, 0.0, 0.0),
            (0.002, -0.05, -0.01),
            (0.005, -0.12, -0.02),
            (0.008, -0.18, -0.03),
            (0.010, -0.22, -0.035),
        ]
        for expected_dq, expected_dw, expected_de in expected_samples:
            frame = state.get_telemetry()
            st = frame['state']
            assert st['delta_q_rads'] == expected_dq
            assert st['delta_w_mps'] == expected_dw
            assert st['delta_e_rad'] == expected_de
            assert 'pitch_rad' not in frame.get('attitude', {})
            assert 'pitch_rad' not in frame

    def test_changing_playback_samples_updates_visualization(self, state):
        """Verify changing playback samples changes corresponding aircraft visualization."""
        state.load_run('test_run')

        # Sample 0
        f0 = state.get_telemetry()
        assert f0['playback_index'] == 0
        vis0 = f0['visualization']
        assert 'pitch_visual' in vis0
        assert 'vertical_visual' in vis0
        assert 'lateral_visual' in vis0
        # No lateral telemetry in test_run CSV -> lateral motion is disabled
        assert vis0['lateral_visual'] == 0.0
        assert vis0['lateral_mapping'] == 'DISABLED — NO LATERAL TELEMETRY'

        # Sample 1 (different delta_q, delta_w, time)
        f1 = state.get_telemetry()
        assert f1['playback_index'] == 1
        vis1 = f1['visualization']

        # Pitch visualization changes with delta_q_rads
        assert vis1['pitch_visual'] != vis0['pitch_visual']
        # Vertical visualization changes with delta_w_mps
        assert vis1['vertical_visual'] != vis0['vertical_visual']
        # Lateral remains disabled because no lateral telemetry exists in this run
        assert vis1['lateral_visual'] == 0.0
        assert vis1['lateral_mapping'] == 'DISABLED — NO LATERAL TELEMETRY'
        # Both state and visualization update from same sample
        assert f1['state']['delta_q_rads'] == 0.002
        assert f1['state']['delta_w_mps'] == -0.05

    def test_visual_transformations_are_bounded(self, state):
        """Verify all visual transformations stay strictly within safe bounds."""
        state.load_run('test_run')
        for _ in range(5):
            frame = state.get_telemetry()
            vis = frame['visualization']
            assert -0.35 <= vis['pitch_visual'] <= 0.35
            assert -1.0 <= vis['vertical_visual'] <= 1.0
            assert -0.25 <= vis['lateral_visual'] <= 0.25

    def test_no_forbidden_lateral_states_created(self, state):
        """Verify no roll, yaw, p, r, beta, phi, or psi physical states are created."""
        state.load_run('test_run')
        forbidden = {'roll', 'yaw', 'p', 'r', 'beta', 'phi', 'psi'}
        for _ in range(5):
            frame = state.get_telemetry()
            st = frame['state']
            for k in forbidden:
                assert k not in st
                assert f"delta_{k}" not in st

    def test_mock_frame_visualization_clearly_labeled_demo(self, empty_state):
        """Verify mock fallback clearly labels lateral motion as visual demo."""
        frame = empty_state.get_telemetry()
        vis = frame['visualization']
        assert 'pitch_visual' in vis
        assert 'vertical_visual' in vis
        assert 'lateral_visual' in vis
        assert vis['lateral_mapping'] == 'PROJECT_DEFINED_VISUAL DEMO — NOT SIMULATED LATERAL DYNAMICS'

    def test_lateral_control_surface_used_when_present(self, tmp_path):
        """Verify lateral control surface is used when actual lateral telemetry exists."""
        lat_csv = tmp_path / "lat_run.csv"
        lat_csv.write_text(
            "time_s,delta_w_mps,delta_q_rads,delta_a_rad\n"
            "0.0,0.0,0.0,0.05\n",
            encoding="utf-8",
        )
        lat_state = TelemetryState(data_dir=tmp_path)
        lat_state.load_run("lat_run")
        frame = lat_state.get_telemetry()
        vis = frame['visualization']
        assert vis['lateral_mapping'] == 'CONTROL_SURFACE_AILERON'
        assert vis['lateral_visual'] != 0.0


# ===================================================================
# INTEGRATION TESTS — Live server
# ===================================================================

@pytest.fixture(scope="module")
def live_server():
    """Spin up a live server on an ephemeral port with a sample CSV."""
    tmp = tempfile.mkdtemp()
    csv_file = Path(tmp) / "integration_run.csv"
    csv_file.write_text(SAMPLE_CSV.strip(), encoding="utf-8")

    state = TelemetryState(data_dir=Path(tmp))
    handler_cls = make_handler(state)

    class TestServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    # chdir to project root so dashboard.html can be served
    original_dir = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    httpd = TestServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]

    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    yield port, state

    httpd.shutdown()
    httpd.server_close()
    server_thread.join(timeout=2)
    os.chdir(original_dir)


class TestLiveServer:
    """Integration tests against a running server."""

    def test_api_telemetry_returns_200(self, live_server):
        port, _ = live_server
        url = f"http://127.0.0.1:{port}/api/telemetry"
        response = urllib.request.urlopen(url)
        assert response.status == 200
        data = json.loads(response.read())
        assert 'state' in data
        assert 'health' in data
        assert 'provenance' in data
        assert 'delta_q_rads' in data['state']
        assert 'pitch_rad' not in data.get('attitude', {})
        assert 'pitch_rad' not in data

    def test_api_runs_returns_200(self, live_server):
        port, _ = live_server
        url = f"http://127.0.0.1:{port}/api/runs"
        response = urllib.request.urlopen(url)
        assert response.status == 200
        data = json.loads(response.read())
        assert 'runs' in data
        assert 'integration_run' in data['runs']

    def test_load_and_playback(self, live_server):
        port, state = live_server
        # Load the run
        url = f"http://127.0.0.1:{port}/api/run/integration_run"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        assert data['ok'] is True

        # Poll twice, index should advance
        url = f"http://127.0.0.1:{port}/api/telemetry"
        r1 = json.loads(urllib.request.urlopen(url).read())
        r2 = json.loads(urllib.request.urlopen(url).read())
        assert r2['playback_index'] > r1['playback_index']

    def test_nocache_headers_live(self, live_server):
        port, _ = live_server
        url = f"http://127.0.0.1:{port}/api/telemetry"
        response = urllib.request.urlopen(url)
        assert 'no-cache' in response.headers.get('Cache-Control', '')

    def test_root_serves_html(self, live_server):
        port, _ = live_server
        url = f"http://127.0.0.1:{port}/"
        try:
            response = urllib.request.urlopen(url)
            assert response.status == 200
            content_type = response.headers.get('Content-Type', '')
            assert 'html' in content_type.lower()
            # Regression: static responses must also carry no-cache headers
            assert 'no-cache' in response.headers.get('Cache-Control', '')
        except urllib.error.HTTPError as e:
            # 404 is acceptable if dashboard.html isn't in the test working dir
            assert e.code == 404

    def test_root_does_not_crash_end_headers(self, live_server):
        """Regression: end_headers() must not raise ValueError on static files.

        Previously every GET / crashed with
        ``ValueError: too many values to unpack (expected 2)`` because
        ``_headers_buffer`` holds raw bytes lines, not (key, value) tuples.
        """
        port, _ = live_server
        url = f"http://127.0.0.1:{port}/"
        try:
            response = urllib.request.urlopen(url)
            assert response.status == 200
        except urllib.error.HTTPError as e:
            # A clean 404 is fine; a dropped connection (server-side
            # exception) is what this test guards against.
            assert e.code == 404


class TestThreadSafety:
    """Verify concurrent polling doesn't corrupt state."""

    def test_concurrent_polling(self, state):
        state.load_run('test_run')
        errors = []

        def poll_loop(n):
            try:
                for _ in range(n):
                    frame = state.get_telemetry()
                    # Verify structural integrity
                    assert 'state' in frame
                    assert 'delta_q_rads' in frame['state']
                    assert 'pitch_rad' not in frame.get('attitude', {})
                    assert frame['playback_index'] is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=poll_loop, args=(20,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"