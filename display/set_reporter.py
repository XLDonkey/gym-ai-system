"""
XL Fitness AI — Set Reporter
Fires when a set completes and sends the result to Power Automate and Supabase.

The Bible spec (Section 42) defines the JSON payload format. Pi detects a
completed set (member stands up / long rest) → SetReporter.report_set() →
HTTP POST to Power Automate webhook → Power Automate writes to Dataverse →
Power Apps tablet updates.

When moving off Power Apps (3–5 gym scale), only POWER_AUTOMATE_URL in
config.py changes. The Pi-side code is unchanged.

Payload format:
{
  "machine_id":     "lat-pulldown-01",
  "machine_name":   "Nautilus Lat Pulldown",
  "member_id":      "M1089",
  "member_name":    "Matthew",
  "timestamp":      "2026-03-27T14:32:11Z",
  "exercise":       "Lat Pulldown",
  "weight_kg":      52.5,
  "reps":           10,
  "form_breakdown": {"good": 7, "partial": 2, "bad": 1},
  "form_score":     0.78,
  "model_version":  "v1.0-lstm"
}
"""

from datetime import datetime, timezone
from typing import Optional

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


class SetReporter:
    """
    Reports completed sets to Power Automate (and optionally a custom webhook).

    Args:
        power_automate_url: HTTP trigger URL from Power Automate (config.py).
                            Leave blank to log to console only.
        machine_id:         e.g. "xlf-pi-001"
        machine_name:       e.g. "Nautilus Lat Pulldown"
        model_version:      Active ONNX version (from models/registry.json)
        custom_webhook_url: Optional second endpoint (future custom web app)
        enabled:            Set False to disable all HTTP calls
    """

    def __init__(
        self,
        power_automate_url: str  = "",
        machine_id:         str  = "",
        machine_name:       str  = "",
        model_version:      str  = "v0.1-rule-based",
        custom_webhook_url: str  = "",
        enabled:            bool = True,
    ):
        self.pa_url        = power_automate_url
        self.custom_url    = custom_webhook_url
        self.machine_id    = machine_id
        self.machine_name  = machine_name
        self.model_version = model_version
        self.enabled       = enabled

        self._set_n = 0
        self._can_send = enabled and _REQUESTS and bool(power_automate_url)

        if enabled and not power_automate_url:
            print("[set_reporter] No Power Automate URL — console-only mode")
        elif self._can_send:
            print(f"[set_reporter] Active  pa_url={power_automate_url[:50]}...")

    # ── Public API ─────────────────────────────────────────────────────────────

    def report_set(
        self,
        reps:         int,
        form_good:    int,
        form_partial: int,
        form_bad:     int,
        weight_kg:    Optional[float] = None,
        member_id:    Optional[str]   = None,
        member_name:  Optional[str]   = None,
    ) -> bool:
        """
        Call when a set is detected as complete.
        Returns True if successfully delivered to Power Automate.
        """
        self._set_n += 1
        total = form_good + form_partial + form_bad
        score = round(form_good / max(total, 1), 2)

        payload = {
            "machine_id":     self.machine_id,
            "machine_name":   self.machine_name,
            "member_id":      member_id   or "unknown",
            "member_name":    member_name or "Unknown",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "exercise":       self.machine_name,
            "weight_kg":      weight_kg,
            "reps":           reps,
            "form_breakdown": {"good": form_good, "partial": form_partial, "bad": form_bad},
            "form_score":     score,
            "model_version":  self.model_version,
        }

        print(
            f"[set_reporter] Set #{self._set_n} complete — "
            f"{member_name or 'unknown'}  "
            f"{reps} reps @ {weight_kg or '?'}kg  "
            f"form={score:.0%}  ({form_good}G/{form_partial}P/{form_bad}B)"
        )

        if not self._can_send:
            return True  # console-only mode — not an error

        ok_pa     = self._post(self.pa_url, payload)
        ok_custom = self._post(self.custom_url, payload) if self.custom_url else True
        return ok_pa and ok_custom

    # ── Internal ───────────────────────────────────────────────────────────────

    def _post(self, url: str, payload: dict) -> bool:
        if not url:
            return True
        try:
            r = requests.post(url, json=payload, timeout=10,
                              headers={"Content-Type": "application/json"})
            if r.status_code in (200, 202):
                return True
            print(f"[set_reporter] HTTP {r.status_code}: {r.text[:120]}")
            return False
        except Exception as e:
            print(f"[set_reporter] Send error: {e}")
            return False
