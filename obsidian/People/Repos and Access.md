---
title: Repos and Access
tags: [people, github, repos, access]
created: 2026-04-09
---

# GitHub Repos and Access

---

## Repositories

| Repo | Owner | Visibility | Purpose |
|------|-------|-----------|---------|
| `xldonkey/gym-ai-system` | XLDonkey | **Public** | Development branch, GitHub Pages (Bible) |
| `Matt-xlfitness/Gym-Overseer-AI` | Matt-xlfitness | **Private** | Production code |

---

## GitHub Pages

- **Live Bible:** https://xldonkey.github.io/gym-ai-system/bible.html
- Hosted from `xldonkey/gym-ai-system` (must stay public for free GitHub Pages)
- Source: `bible.html` in repo root
- Updates automatically on push to `main`

---

## Development Branch

All Claude Code development happens on:
```
claude/gym-tracking-neural-network-sHldz
```

Do not develop directly on `main`.

---

## Git Push Authentication

The Claude session proxy authenticates as `Matt-xlfitness`. Pushing to XLDonkey's repo requires an explicit PAT in the URL:

```bash
git push https://Matt-xlfitness:<PAT>@github.com/XLDonkey/gym-ai-system.git \
  claude/gym-tracking-neural-network-sHldz
```

This is handled automatically by Claude Code in each session.

---

## Access Summary

| Person | Role | Access |
|--------|------|--------|
| XLDonkey | Repo owner | Full access to xldonkey/gym-ai-system |
| Matt-xlfitness | Contributor | Write access to xldonkey/gym-ai-system, owner of Gym-Overseer-AI |

---

## Related

- [[Decisions/Stack Choices]] — why two repos
- [[Home]] — project overview
