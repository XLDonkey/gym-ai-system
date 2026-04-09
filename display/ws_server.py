"""
XL Fitness AI — Machine WebSocket Server
Broadcasts live inference state to the tablet mounted on each machine.

Runs as a background thread inside pi/main.py (alongside the inference loop).
The tablet (display/tablet.html) connects to ws://[pi-ip]:8788 and receives
JSON state updates every 100ms.

Broadcast payload:
{
  "machine_id":   "xlf-pi-001",
  "machine_name": "Nautilus Lat Pulldown",
  "member_name":  "Matthew",        ← from user_tracking / PersonDB
  "activity":     "good_rep",       ← from ActivityStateMachine
  "activity_id":  3,
  "phase":        "ENGAGED",
  "rep_count":    8,                ← current set rep count
  "weight_kg":    52.5,             ← from WeightDetector or e_weight StackClient
  "form_score":   0.85,             ← good_reps / total_reps this set
  "session_reps": 24,               ← total reps this session (all sets)
  "session_sets": 3,
  "confidence":   0.92,
  "timestamp":    "2026-04-09T..."
}

Usage in pi/main.py:
    from display.ws_server import MachineWSServer
    ws = MachineWSServer(machine_id=cfg.MACHINE_ID, machine_name=cfg.MACHINE_NAME)
    ws.start()
    # In main loop (every frame):
    ws.update_state(member_name=..., activity=..., rep_count=..., weight_kg=...)
"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Optional

try:
    import websockets
    _WS = True
except ImportError:
    _WS = False
    print("[ws_server] INFO: websockets not installed — pip install websockets")


class MachineWSServer:
    """
    WebSocket server that broadcasts machine state to the mounted tablet.

    Args:
        machine_id:   Unique machine ID (from config.py)
        machine_name: Human-readable name (e.g. "Nautilus Lat Pulldown")
        port:         WebSocket port (default 8788)
        enabled:      Set False to disable entirely
    """

    BROADCAST_HZ = 10   # 10 updates/sec to tablet — enough for smooth display

    def __init__(
        self,
        machine_id:   str  = "",
        machine_name: str  = "",
        port:         int  = 8788,
        enabled:      bool = True,
    ):
        self.machine_id   = machine_id
        self.machine_name = machine_name
        self.port         = port
        self.enabled      = enabled and _WS

        self._state: dict             = self._blank_state()
        self._lock                    = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._clients: set            = set()

        if enabled and not _WS:
            print("[ws_server] DISABLED — install: pip install websockets")
        elif enabled:
            print(f"[ws_server] Ready  ws://0.0.0.0:{self.port}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start broadcasting in a daemon thread. Non-blocking."""
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="WSServer")
        self._thread.start()

    def update_state(
        self,
        member_name:   Optional[str]   = None,
        member_id:     Optional[str]   = None,
        activity:      str             = "no_person",
        activity_id:   int             = 0,
        phase:         str             = "IDLE",
        rep_count:     int             = 0,
        weight_kg:     Optional[float] = None,
        form_score:    float           = 0.0,
        session_reps:  int             = 0,
        session_sets:  int             = 0,
        confidence:    float           = 0.0,
    ):
        """
        Update broadcast state. Thread-safe. Call every frame from the main loop.
        Only connected tablet clients actually receive updates.
        """
        with self._lock:
            self._state.update({
                "member_name":  member_name,
                "member_id":    member_id,
                "activity":     activity,
                "activity_id":  activity_id,
                "phase":        phase,
                "rep_count":    rep_count,
                "weight_kg":    weight_kg,
                "form_score":   round(form_score, 2),
                "session_reps": session_reps,
                "session_sets": session_sets,
                "confidence":   round(confidence, 3),
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            })

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── Async internals ────────────────────────────────────────────────────────

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._on_connect, "0.0.0.0", self.port):
            print(f"[ws_server] Listening on ws://0.0.0.0:{self.port}")
            await self._broadcast_loop()

    async def _on_connect(self, ws):
        self._clients.add(ws)
        print(f"[ws_server] Tablet connected  ({len(self._clients)} client(s))")
        # Send current state immediately on connect
        with self._lock:
            await ws.send(json.dumps(self._state))
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    async def _broadcast_loop(self):
        interval = 1.0 / self.BROADCAST_HZ
        while True:
            if self._clients:
                with self._lock:
                    payload = json.dumps(self._state)
                dead = set()
                for ws in list(self._clients):
                    try:
                        await ws.send(payload)
                    except Exception:
                        dead.add(ws)
                self._clients -= dead
            await asyncio.sleep(interval)

    def _blank_state(self) -> dict:
        return {
            "machine_id":   self.machine_id,
            "machine_name": self.machine_name,
            "member_name":  None,
            "member_id":    None,
            "activity":     "no_person",
            "activity_id":  0,
            "phase":        "IDLE",
            "rep_count":    0,
            "weight_kg":    None,
            "form_score":   0.0,
            "session_reps": 0,
            "session_sets": 0,
            "confidence":   0.0,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }
