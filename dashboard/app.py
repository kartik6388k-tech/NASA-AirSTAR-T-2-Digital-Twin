"""
NASA AirSTAR T-2 — Simulation Telemetry Playback Backend.

Architecture
------------
Subclasses ``http.server.SimpleHTTPRequestHandler`` to act as both a static
file server (for ``dashboard.html``) and a REST-style JSON API for telemetry
playback.

Endpoints
---------
GET /                → serves dashboard/dashboard.html
GET /api/telemetry   → current telemetry frame (JSON)
GET /api/runs        → list of available saved CSV run names (JSON)
GET /api/run/<name>  → load a specific run for playback (JSON acknowledgement)
GET /api/playback    → playback control (play/pause/reset via ?action=...)
Everything else      → 404

Thread Safety
-------------
All mutable playback state (loaded CSV rows, playback index, playback mode)
is guarded by ``threading.Lock`` so rapid 50 ms polling from the browser
cannot corrupt shared state.

Data Provenance
---------------
The dashboard is read-only and does not execute simulations.  When no saved
CSV run is loaded the API falls back to a deterministic mathematical
oscillator labeled ``DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR``.
Stored ``delta_q_rads`` is served directly for qualitative visualization
without attitude integration or newly created physical states.
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# CSV field constants (mirror the Simulator output schema)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ("time_s", "delta_w_mps", "delta_q_rads")
NUMERIC_FIELDS = (
    "time_s", "delta_w_mps", "delta_q_rads", "delta_e_rad",
    "delta_w_dot", "delta_q_dot", "perturbation_az_mps2",
    "delta_alpha_rad", "delta_CL", "delta_Cm",
    "delta_Lift_N", "delta_PitchMom_Nm", "total_thrust_N",
    "w_gust_mps", "delta_w_aero_mps",
)
NAN_PERMITTED = frozenset({"total_thrust_N"})


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def _parse_float(col: str, raw: str) -> Optional[float]:
    """Parse a numeric CSV cell; return None for empty optional fields."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if math.isnan(v) and col in NAN_PERMITTED:
        return v  # NaN is legitimate for total_thrust_N
    if not math.isfinite(v) and col not in NAN_PERMITTED:
        return None
    return v


def load_csv_run(path: Path) -> List[Dict[str, Any]]:
    """Load and validate one saved Simulator CSV file.

    Returns a list of row dicts with numeric fields parsed to float.
    Raises ``ValueError`` if the file is missing required columns or is empty.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        headers = [h.strip() for h in reader.fieldnames]
        missing = [c for c in REQUIRED_FIELDS if c not in headers]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            parsed: Dict[str, Any] = {}
            for col in headers:
                raw = (row.get(col) or "").strip()
                if col in NUMERIC_FIELDS:
                    parsed[col] = _parse_float(col, raw)
                else:
                    parsed[col] = raw if raw else None
            rows.append(parsed)

    if not rows:
        raise ValueError(f"CSV is empty: {path.name}")
    return rows


# ---------------------------------------------------------------------------
# Qualitative visual kinematics
# ---------------------------------------------------------------------------
def _compute_visual_kinematics(
    row: Dict[str, Any],
    scenario_name: Optional[str] = None,
    is_demo: bool = False,
) -> Dict[str, Any]:
    """Compute bounded visual transformations from a telemetry sample.

    Visual mapping:
      - delta_q_rads -> bounded pitch rotation (nose up/down)
      - delta_w_mps  -> bounded vertical movement
      - Left/right   -> disabled when no lateral telemetry exists in the saved run;
                        in demo mode, clearly labeled
                        PROJECT_DEFINED_VISUAL DEMO — NOT SIMULATED LATERAL DYNAMICS.

    Note:
      Visualization is qualitative and does not represent simulated lateral
      dynamics. No roll, yaw, p, r, beta, phi, or psi physical states are created.
    """
    t = row.get("time_s")
    t_val = float(t) if t is not None else 0.0

    dq = row.get("delta_q_rads")
    dq_val = float(dq) if dq is not None else 0.0

    dw = row.get("delta_w_mps")
    dw_val = float(dw) if dw is not None else 0.0

    # delta_q_rads -> bounded qualitative pitch rate visual cue
    pitch_visual = round(max(-0.25, min(0.25, dq_val * 2.5)), 6)

    # delta_w_mps -> controlled visual heave displacement cue
    vertical_visual = round(max(-0.4, min(0.4, -dw_val * 0.15)), 6)

    da = row.get("delta_a_rad")
    if da is not None:
        try:
            da_val = float(da)
            lateral_val = max(-0.25, min(0.25, da_val * 2.0))
            lateral_mapping = "CONTROL_SURFACE_AILERON"
        except (ValueError, TypeError):
            lateral_val = 0.0
            lateral_mapping = "DISABLED — NO LATERAL TELEMETRY"
    elif is_demo:
        # No lateral dynamics exist in the short-period model; disabled in DEMO too
        lateral_val = 0.0
        lateral_mapping = "DISABLED — NO LATERAL DYNAMICS IN SHORT-PERIOD MODEL"
    else:
        # Saved telemetry playback: strictly disabled (no lateral states in longitudinal twin)
        lateral_val = 0.0
        lateral_mapping = "DISABLED — NO LATERAL TELEMETRY"

    lateral_visual = round(max(-0.25, min(0.25, lateral_val)), 6)

    return {
        "pitch_visual": pitch_visual,
        "vertical_visual": vertical_visual,
        "lateral_visual": lateral_visual,
        "lateral_mapping": lateral_mapping,
    }



# ---------------------------------------------------------------------------
# Thread-safe playback state
# ---------------------------------------------------------------------------
class TelemetryState:
    """Thread-safe mutable state for CSV playback.

    All reads and writes go through the lock so concurrent HTTP handler
    threads cannot observe torn state during rapid 50 ms polling.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self._lock = threading.Lock()
        self.data_dir = Path(data_dir).resolve()
        self._rows: List[Dict[str, Any]] = []
        self._index: int = 0
        self._run_name: Optional[str] = None
        self._scenario_name: Optional[str] = None
        self._paused: bool = False
        self._loop: bool = True
        self._source: str = "DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR"
        self._start_time: float = _time.monotonic()

    # -- discovery ----------------------------------------------------------

    def discover_runs(self) -> List[str]:
        """Return sorted list of available CSV run names (graceful fallback)."""
        if not self.data_dir.is_dir():
            return []
        names = []
        for p in sorted(self.data_dir.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() == ".csv":
                names.append(p.stem)
        return names

    # -- loading ------------------------------------------------------------

    def load_run(self, run_name: str) -> Dict[str, Any]:
        """Load a saved CSV run for playback.  Thread-safe."""
        if not run_name or run_name.lower() in ("demo", "mock"):
            with self._lock:
                self._rows = []
                self._index = 0
                self._run_name = None
                self._scenario_name = None
                self._paused = False
                self._source = "DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR"
                self._start_time = _time.monotonic()
            return {"ok": True, "run_name": None, "mode": "DEMO", "samples": 0}

        path = (self.data_dir / f"{run_name}.csv").resolve()
        # Path traversal guard
        try:
            path.relative_to(self.data_dir)
        except ValueError:
            return {"ok": False, "error": "Invalid run name"}
        if not path.is_file():
            return {"ok": False, "error": f"Run not found: {run_name}"}

        try:
            rows = load_csv_run(path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        scenario_name: Optional[str] = None
        meta_path = self.data_dir / f"{run_name}.meta.json"
        if meta_path.is_file():
            try:
                with meta_path.open("r", encoding="utf-8") as mfh:
                    mdata = json.load(mfh)
                    scenario_name = mdata.get("scenario_name")
            except Exception:
                pass

        with self._lock:
            self._rows = rows
            self._index = 0
            self._run_name = run_name
            self._scenario_name = scenario_name
            self._paused = False
            self._source = "SAVED_CSV"

        return {"ok": True, "run_name": run_name, "samples": len(rows)}

    # -- playback control ---------------------------------------------------

    def playback_control(self, action: str) -> Dict[str, Any]:
        with self._lock:
            if action == "pause":
                self._paused = True
            elif action == "play":
                self._paused = False
            elif action == "reset":
                self._index = 0
                self._paused = False
                self._start_time = _time.monotonic()
            elif action in ("unload", "demo"):
                self._rows = []
                self._index = 0
                self._run_name = None
                self._scenario_name = None
                self._paused = False
                self._source = "DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR"
                self._start_time = _time.monotonic()
            elif action == "loop_on":
                self._loop = True
            elif action == "loop_off":
                self._loop = False
            elif action == "loop_toggle":
                self._loop = not self._loop
            else:
                return {"ok": False, "error": f"Unknown action: {action}"}
            return {
                "ok": True,
                "action": action,
                "paused": self._paused,
                "loop": self._loop,
                "source": self._source,
            }

    # -- mock oscillator ----------------------------------------------------

    def _mock_frame(self) -> Dict[str, Any]:
        """Generate a deterministic short-period oscillation frame."""
        elapsed = _time.monotonic() - self._start_time
        omega = 4.5  # short-period natural frequency (rad/s), PROJECT_DEFINED
        zeta = 0.35  # damping ratio, PROJECT_DEFINED
        dq = 0.05 * math.exp(-zeta * omega * elapsed) * math.sin(
            omega * math.sqrt(1 - zeta ** 2) * elapsed
        )
        dw = -0.8 * math.exp(-zeta * omega * elapsed) * math.cos(
            omega * math.sqrt(1 - zeta ** 2) * elapsed
        )
        de = 0.035 * math.sin(0.5 * elapsed)  # slow elevator sweep

        vk = _compute_visual_kinematics(
            {"time_s": elapsed, "delta_q_rads": dq, "delta_w_mps": dw},
            is_demo=True,
        )

        return {
            "timestamp_s": round(elapsed, 4),
            "playback_index": None,
            "playback_total": None,
            "playback_status": "RUNNING",
            "loop": True,
            "state": {
                "delta_w_mps": round(dw, 6),
                "delta_q_rads": round(dq, 6),
                "delta_e_rad": round(de, 6),
            },
            "visualization": vk,
            "forces": {
                "delta_Lift_N": None,
                "delta_PitchMom_Nm": None,
                "total_thrust_N": None,
            },
            "health": {
                "status": "NOMINAL",
                "propulsion_model_status": "MODEL_NOT_IDENTIFIED",
            },
            "environment": {
                "w_gust_mps": 0.0,
                "provenance": "NONE",
            },
            "provenance": {
                "source": "DEMO — PROJECT_DEFINED_MOCK_OSCILLATOR",
                "run_name": None,
            },
        }

    # -- telemetry frame ----------------------------------------------------

    def get_telemetry(self) -> Dict[str, Any]:
        """Return the current telemetry frame.  Thread-safe."""
        with self._lock:
            if not self._rows:
                return self._mock_frame()

            idx = self._index
            row = self._rows[idx]

            vk = _compute_visual_kinematics(row, self._scenario_name)

            is_end = (idx >= len(self._rows) - 1)
            is_looping = self._loop
            if self._paused:
                playback_status = "PAUSED"
            elif is_end and not is_looping:
                playback_status = "COMPLETED"
            else:
                playback_status = "RUNNING"

            w_gust = row.get("w_gust_mps")
            frame: Dict[str, Any] = {
                "timestamp_s": row.get("time_s"),
                "playback_index": idx,
                "playback_total": len(self._rows),
                "playback_status": playback_status,
                "loop": is_looping,
                "state": {
                    "delta_w_mps": row.get("delta_w_mps"),
                    "delta_q_rads": row.get("delta_q_rads"),
                    "delta_e_rad": row.get("delta_e_rad"),
                },
                "visualization": vk,
                "forces": {
                    "delta_Lift_N": row.get("delta_Lift_N"),
                    "delta_PitchMom_Nm": row.get("delta_PitchMom_Nm"),
                    "total_thrust_N": row.get("total_thrust_N"),
                },
                "health": {
                    "status": "NOMINAL",
                    "propulsion_model_status": (
                        row.get("propulsion_model_status") or "UNKNOWN"
                    ),
                },
                "environment": {
                    "w_gust_mps": float(w_gust) if w_gust is not None else 0.0,
                    "provenance": "PROJECT_DEFINED_TURBULENCE" if w_gust is not None else "NONE",
                },
                "provenance": {
                    "source": "SAVED_CSV",
                    "run_name": self._run_name,
                },
            }

            # Advance index if not paused
            if not self._paused:
                if idx < len(self._rows) - 1:
                    self._index += 1
                elif is_looping:
                    self._index = 0

            return frame



# ---------------------------------------------------------------------------
# Custom JSON encoder (handles NaN → null for JSON compliance)
# ---------------------------------------------------------------------------
class _TelemetryEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler routing API calls and serving the dashboard HTML.

    The ``telemetry_state`` attribute is injected by ``make_handler()``.
    """

    server_version = "AirSTAR-Dashboard/2.0"
    telemetry_state: TelemetryState  # injected at class creation time

    def do_GET(self) -> None:  # noqa: N802
        # --- Root → serve dashboard.html ---
        if self.path == "/" or self.path == "/dashboard":
            self.path = "/dashboard/dashboard.html"
            return super().do_GET()

        # --- API: telemetry frame ---
        if self.path == "/api/telemetry":
            frame = self.telemetry_state.get_telemetry()
            return self._json_response(frame)

        # --- API: list available runs ---
        if self.path == "/api/runs":
            runs = self.telemetry_state.discover_runs()
            return self._json_response({"runs": runs})

        # --- API: load a specific run ---
        if self.path.startswith("/api/run/"):
            run_name = self.path[len("/api/run/"):]
            if not run_name:
                return self._json_response({"ok": False, "error": "No run name"}, 400)
            result = self.telemetry_state.load_run(run_name)
            status = 200 if result.get("ok") else 400
            return self._json_response(result, status)

        # --- API: playback control ---
        if self.path.startswith("/api/playback"):
            # Parse ?action=play|pause|reset
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            action = (qs.get("action") or [""])[0]
            if not action:
                return self._json_response(
                    {"ok": False, "error": "Missing ?action=play|pause|reset"}, 400
                )
            result = self.telemetry_state.playback_control(action)
            return self._json_response(result)

        # --- Everything else → 404 ---
        self.send_error(404, "Not Found")

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, cls=_TelemetryEncoder).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # Inject no-cache on ALL responses (including static HTML).
        #
        # ``self._headers_buffer`` is a list of raw *bytes* lines in the
        # form ``b"Header-Name: value\r\n"`` (see ``http.server``'s
        # ``send_header``).  It is NOT a list of (key, value) tuples, so
        # unpacking it as tuples raises ``ValueError: too many values to
        # unpack`` on every response.  Scan the joined bytes instead.
        buffered = b"".join(self._headers_buffer) if self._headers_buffer else b""
        if b"Cache-Control:" not in buffered:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep server log concise; suppress per-poll noise for /api/telemetry
        if args and isinstance(args[0], str) and "/api/telemetry" in args[0]:
            return
        sys.stderr.write(f"[dashboard] {fmt % args}\n")


def make_handler(state: TelemetryState):
    """Create a handler class bound to the given telemetry state."""
    return type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {"telemetry_state": state},
    )


# ---------------------------------------------------------------------------
# Server entry points
# ---------------------------------------------------------------------------
def run_server(host: str = "127.0.0.1", port: int = 8000,
               data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Start the dashboard server (blocking)."""
    os.chdir(str(PROJECT_ROOT))

    state = TelemetryState(data_dir=data_dir)
    handler_cls = make_handler(state)

    class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with ReusableThreadingTCPServer((host, port), handler_cls) as httpd:
        runs = state.discover_runs()
        print("=" * 62)
        print("  NASA AirSTAR T-2 · Simulation Telemetry Playback")
        print("=" * 62)
        print(f"  URL:           http://{host}:{port}/")
        print(f"  Data dir:      {state.data_dir}")
        print(f"  Available runs: {len(runs)}")
        if runs:
            for r in runs[:5]:
                print(f"    • {r}")
            if len(runs) > 5:
                print(f"    … and {len(runs) - 5} more")
        else:
            print("  (no saved runs — mock oscillator active)")
        print("  Mode:          Read-only qualitative playback · no simulation feedback")
        print("=" * 62)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        finally:
            httpd.server_close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NASA AirSTAR T-2 Simulation Telemetry Playback"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Listen port (default: 8000)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing Simulator CSV results")
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port, data_dir=args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())