/**
 * XL Fitness — Session Manager + Server Event Pipeline
 * 
 * Handles:
 * - Session lifecycle (start / end)
 * - Posting events to Manus's server endpoint
 * - Card tap integration (member identity)
 * - Local fallback when server is unavailable
 */

class SessionManager {
  constructor(machineConfig) {
    this.config = machineConfig;
    this.serverUrl = machineConfig.server?.endpoint || '';
    this.machineId = machineConfig.machine_id || 'unknown';
    this.session = null;
    this.queue = []; // offline queue — fires when server reconnects
    this.online = false;
    this._pingInterval = null;
  }

  // ── Server health check ───────────────────────────────────────────────────
  async checkServer() {
    if (!this.serverUrl) return false;
    try {
      const r = await fetch(this.serverUrl.replace('/event', '/health'), {
        method: 'GET', signal: AbortSignal.timeout(2000)
      });
      this.online = r.ok;
    } catch {
      this.online = false;
    }
    return this.online;
  }

  startPing(intervalMs = 30000) {
    this.checkServer();
    this._pingInterval = setInterval(() => {
      this.checkServer().then(() => {
        if (this.online && this.queue.length > 0) this._flushQueue();
      });
    }, intervalMs);
  }

  // ── Post event to server ──────────────────────────────────────────────────
  async postEvent(eventType, payload = {}) {
    const body = {
      machine_id: this.machineId,
      timestamp_utc: new Date().toISOString(),
      event_type: 'pose_estimation',
      payload: { event: eventType, ...payload }
    };

    if (!this.serverUrl) {
      console.log('[Session] No server configured — local only:', body);
      return;
    }

    if (!this.online) {
      this.queue.push(body);
      console.log(`[Session] Offline — queued event (${this.queue.length} pending)`);
      return;
    }

    try {
      const r = await fetch(this.serverUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(3000)
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      console.log(`[Session] ✓ Event posted: ${eventType}`);
    } catch (e) {
      console.warn(`[Session] Post failed (${e.message}) — queuing`);
      this.queue.push(body);
      this.online = false;
    }
  }

  async _flushQueue() {
    console.log(`[Session] Flushing ${this.queue.length} queued events…`);
    const pending = [...this.queue];
    this.queue = [];
    for (const body of pending) {
      try {
        await fetch(this.serverUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
      } catch {
        this.queue.push(body); // re-queue if still failing
      }
    }
  }

  // ── Session lifecycle ─────────────────────────────────────────────────────
  startSession(memberId = null) {
    this.session = {
      id: `${this.machineId}_${Date.now()}`,
      member_id: memberId,
      start_time: new Date().toISOString(),
      reps: 0,
      good_reps: 0,
      bad_reps: 0,
      sets: 0,
      weight_kg: null
    };

    this.postEvent('session_start', {
      session_id: this.session.id,
      member_id: memberId
    });

    console.log(`[Session] Started — member: ${memberId || 'unknown'}`);
    return this.session;
  }

  // Card tap — identifies member mid-session (data flywheel)
  onCardTap(memberId) {
    if (!this.session) this.startSession(memberId);
    else {
      const wasUnknown = !this.session.member_id;
      this.session.member_id = memberId;
      // If we didn't know who this was — this tap creates a training label
      if (wasUnknown) {
        this.postEvent('member_identified_by_tap', {
          session_id: this.session.id,
          member_id: memberId,
          note: 'AI was uncertain — card tap provided ground truth label'
        });
      }
    }
  }

  onUserSeated() {
    if (!this.session) this.startSession();
    this.postEvent('user_seated_engaged', {
      session_id: this.session?.id
    });
  }

  onRepCompleted(quality, confidence, elbowAngle) {
    if (!this.session) this.startSession();
    this.session.reps++;
    if (quality === 'good') this.session.good_reps++;
    else this.session.bad_reps++;

    this.postEvent('rep_completed', {
      session_id: this.session.id,
      rep_quality: quality,
      confidence: confidence,
      elbow_angle: elbowAngle,
      rep_number: this.session.reps
    });
  }

  onBadRep(elbowAngle) {
    if (!this.session) return;
    this.session.bad_reps++;
    this.postEvent('bad_rep', {
      session_id: this.session?.id,
      elbow_angle: elbowAngle
    });
  }

  onWeightSet(weightKg) {
    if (this.session) this.session.weight_kg = weightKg;
    this.postEvent('weight_set', {
      session_id: this.session?.id,
      weight_kg: weightKg
    });
  }

  endSession() {
    if (!this.session) return;
    const duration = (Date.now() - new Date(this.session.start_time).getTime()) / 1000;

    this.postEvent('session_end', {
      session_id: this.session.id,
      member_id: this.session.member_id,
      total_reps: this.session.reps,
      good_reps: this.session.good_reps,
      bad_reps: this.session.bad_reps,
      weight_kg: this.session.weight_kg,
      duration_s: Math.round(duration)
    });

    console.log(`[Session] Ended — ${this.session.reps} reps, ${Math.round(duration)}s`);
    const ended = { ...this.session };
    this.session = null;
    return ended;
  }

  // ── Status ────────────────────────────────────────────────────────────────
  getStatus() {
    return {
      server: this.serverUrl || 'not configured',
      online: this.online,
      queued_events: this.queue.length,
      active_session: !!this.session,
      member: this.session?.member_id || null,
      reps: this.session?.reps || 0
    };
  }
}

// Export for use in HTML pages
if (typeof module !== 'undefined') module.exports = { SessionManager };
