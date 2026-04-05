# gym-ai-system

**XL Fitness AI — The complete AI system for Australia's first tech-enabled gym.**

---

## Live Demo

👉 **https://xldonkey.github.io/gym-ai-system/pose/alpha.html**

Open in Chrome. Point camera side-on to the machine. Start pulling.


## Bible

👉 **https://xldonkey.github.io/gym-ai-system/pose/alpha.html**

A html file of the overall plan for the overseer 

---

## Structure

```
gym-ai-system/
├── pose/           ✅ Active  — Rep counting, form scoring
├── weight/         🔜 Next   — Weight plate detection
├── face/           🔜 Later  — Face ID, member recognition
├── members/        🔜 Later  — Member profiles, session history
└── docs/                     — Guides and documentation
```

Each folder is its own project. We build them one at a time.

---

## When We Split Into Separate Repos

We'll split a folder into its own repo when:
- It needs its own team or developer
- It needs its own deployment pipeline
- It gets too large to share a repo cleanly

Until then, everything lives here.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Always stable — live demo runs from here |
| `dev` | Active development |
| `feature/xxx` | One branch per new feature |

---

*Built by Donkey 🫏 for XL Fitness Australia*
