---
title: Display Layer Decisions
tags: [decisions, display, tablet, staff, web-app]
created: 2026-04-09
---

# Display Layer — Decisions

How gym state is shown to members and staff, and why.

---

## Member Tablet (`display/tablet.html`)

- Mounted on each machine in **Kiosk mode** (full-screen browser, no UI chrome)
- Connects to Pi's WebSocket: `ws://[pi-ip]:8788`
- Works **entirely on the local network** — no internet needed
- Shows: rep counter (huge font), member name, weight, form pills, confidence bar

**Why WebSocket over polling?** Sub-100ms updates. Rep counter feels instant.
**Why plain HTML, not React?** Runs on a cheap tablet with no build step needed. Zero maintenance.
**Why served from Pi?** Tablet and Pi are on the same machine. No external dependency.

---

## Staff Floor View (`display/staff.html`)

- Open on any browser on the local gym network
- Connects to each Pi's WebSocket directly (staff enters Pi IPs once, saved to localStorage)
- Shows all machines on one screen — no server needed

**Why not a server-rendered dashboard?** For a single gym, staff.html connecting directly to Pis is simpler and has no single point of failure. If internet goes down, staff view still works.

---

## Set Reporting — Current vs Planned

### Current: Power Automate

```
Pi → HTTP POST → Power Automate webhook → Dataverse → Power Apps
```

- Works today, zero infrastructure
- Free up to Power Automate limits (~750 runs/month on free tier)
- Latency: 3–10 seconds to appear in Power Apps
- Limitation: locked in Microsoft, expensive at scale

### Planned: Next.js Web App

```
Pi → HTTP POST → /api/set (Vercel) → Supabase → Realtime → Dashboard
```

- Sub-1 second end-to-end
- Full custom UI — your brand, your design
- Free on Vercel + Supabase free tier for first gym
- Scales to multiple gyms with `gym_id` + Row-Level Security
- Member app (PWA) possible: scan QR → see your own session live

### Migration

Only one line changes in `pi/config.py`:
```python
# Before
POWER_AUTOMATE_URL = "https://prod-xx.logic.azure.com/..."

# After — point to Next.js instead
SERVER_URL = "https://xlf-dashboard.vercel.app/api/set"
```

Pi code is **unchanged**.

---

## Future: Member Mobile App

Once Next.js backend exists:
- Members scan a QR code at the machine
- Opens a PWA (progressive web app) on their phone
- Shows live rep count, their session history, personal bests
- No app store needed — just a URL

---

## Professional Dashboard Stack (When Ready to Build)

```
dashboard/
├── app/
│   ├── api/
│   │   ├── set/route.ts        ← replaces Power Automate
│   │   ├── heartbeat/route.ts  ← Pi health pings
│   │   └── member/route.ts     ← enrol + lookup
│   ├── dashboard/page.tsx      ← staff floor view (SSR)
│   └── layout.tsx
├── lib/
│   ├── supabase.ts
│   └── types.ts
└── package.json
```

Deploy to Vercel — free tier, auto-deploys from GitHub push.

---

## Related

- [[System/WebSocket Layer]] — technical detail of ws_server.py
- [[System/Database Schema]] — Supabase tables
- [[Decisions/Stack Choices]] — why Supabase over Power Apps
