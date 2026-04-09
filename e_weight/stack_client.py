"""
XL Fitness AI — Electric Weight Stack Client  [PHASE 2 — PLACEHOLDER]
Retrieves the current weight from a custom brushless motor weight stack.

Status: Phase 2 — hardware not yet built.

The electric stack replaces the traditional pin-loaded iron weight stack.
A brushless motor drives the selector to the chosen weight position and locks.
The motor controller runs on the same local network as the Pi.

Why this beats camera detection:
  • Traditional pin stack: camera sees the pin → OCR or position detection → ±5 kg accuracy
  • Our electric stack:     motor commanded to position → reads back exact position → 100% accuracy
  • Can do 1 kg increments vs 5 kg pin stack steps
  • Member selects weight on the tablet → motor moves → Pi reads confirmed weight

API (planned, subject to motor controller firmware):
    GET http://{STACK_IP}/api/weight   → {"weight_kg": 42.5, "locked": true}
    POST http://{STACK_IP}/api/weight  → {"weight_kg": 50.0}  (set weight)

Integration (Phase 2):
    client = StackClient(stack_ip="192.168.1.200")
    weight = client.get_weight()
    if weight is not None:
        db.log_set(member_id, machine_id, weight_kg=weight, reps=reps)
"""

import time
from typing import Optional

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


class StackClient:
    """
    API client for an electric weight stack motor controller.

    Phase 2 placeholder — API endpoints will be finalised when the
    motor controller firmware is written.

    Args:
        stack_ip:   IP address of the motor controller on the local network.
        port:       HTTP port (default 80).
        timeout_s:  Request timeout in seconds.
        enabled:    Set False to disable (returns None for all reads — fail-open).
    """

    API_WEIGHT_PATH = "/api/weight"

    def __init__(
        self,
        stack_ip:  str   = "",
        port:      int   = 80,
        timeout_s: float = 2.0,
        enabled:   bool  = False,   # disabled by default until hardware exists
    ):
        self.base_url  = f"http://{stack_ip}:{port}"
        self.timeout_s = timeout_s
        self.enabled   = enabled and bool(stack_ip) and _REQUESTS

        if enabled and not stack_ip:
            print("[e_weight] WARNING: stack_ip not configured — e_weight disabled")
        elif self.enabled:
            print(f"[e_weight] Stack client active  url={self.base_url}")
        else:
            print("[e_weight] Stack client DISABLED (Phase 2 — hardware not yet installed)")

    def get_weight(self) -> Optional[float]:
        """
        Read the current weight from the motor controller.

        Returns:
            Weight in kg (float), or None if unavailable / not configured.
        """
        if not self.enabled:
            return None

        try:
            r = requests.get(
                self.base_url + self.API_WEIGHT_PATH,
                timeout=self.timeout_s,
            )
            if r.status_code == 200:
                data = r.json()
                kg = data.get("weight_kg")
                return float(kg) if kg is not None else None
        except Exception as e:
            print(f"[e_weight] Read error: {e}")

        return None

    def set_weight(self, weight_kg: float) -> bool:
        """
        Command the motor to move to the specified weight.
        Returns True on success, False otherwise.

        Phase 2 — called from tablet kiosk when member selects their weight.
        """
        if not self.enabled:
            return False

        try:
            r = requests.post(
                self.base_url + self.API_WEIGHT_PATH,
                json={"weight_kg": weight_kg},
                timeout=self.timeout_s,
            )
            return r.status_code in (200, 201, 204)
        except Exception as e:
            print(f"[e_weight] Set error: {e}")
            return False

    def is_locked(self) -> bool:
        """
        Check whether the stack is locked at the target position.
        Motor controller signals 'locked': true when position is confirmed.
        Returns False if unavailable.
        """
        if not self.enabled:
            return False
        try:
            r = requests.get(self.base_url + self.API_WEIGHT_PATH, timeout=self.timeout_s)
            if r.status_code == 200:
                return bool(r.json().get("locked", False))
        except Exception:
            pass
        return False
