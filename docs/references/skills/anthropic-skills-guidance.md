---
source: https://x.com/trq212/status/2033949937936085378
captured: 2026-03-18
version: Published 2026-03-17 by Thariq (Anthropic). 4.3M views, 35K bookmarks.
---

# Anthropic Internal Guidance: How to Use Skills

Primary internal reference on skill design from the team that built Claude Code. Read this before creating any new skill for this project.

---

## Core Concept: A Skill Is a Folder, Not a Markdown File

The most important thing to internalize: skills can include scripts, assets, data, config files, reference subdirectories, and hooks — not just instructions. The file system is a form of context engineering and progressive disclosure.

---

## The 9 Skill Types

Use this taxonomy to decide what kind of skill you're building before writing it.

| Type | What it does | Relevant for this project? |
|---|---|---|
| **1. Library & API Reference** | How to correctly use a library/CLI — edge cases, gotchas, reference snippets | Yes — Stan, cmdstanpy, brms, census API |
| **2. Product Verification** | How to test that code is working; often includes scripts + external tools | Yes — Stage 1 output validation, tract FIPS integrity checks |
| **3. Data Fetching & Analysis** | Connects to data/monitoring stacks; includes credentials, dashboard IDs, workflow instructions | Yes — ACS pipeline, VEST crosswalk, ARDA fetch |
| **4. Business Process & Team Automation** | Automates repetitive workflows into one command | Maybe — pipeline run orchestration |
| **5. Code Scaffolding & Templates** | Boilerplate for specific codebase patterns | Maybe — Stan model scaffolding |
| **6. Code Quality & Review** | Enforces code quality; can include deterministic scripts | Maybe — later |
| **7. CI/CD & Deployment** | Push, deploy, monitor code | No — solo project |
| **8. Runbooks** | Takes a symptom, walks through investigation, produces structured report | Yes — "NMF result looks wrong" debug guide |
| **9. Infrastructure Operations** | Routine maintenance with guardrails | No |

---

## Key Tips (Distilled)

**Build a Gotchas section first.** The highest-signal content in any skill. Start it from the first real failure. Update it every time Claude hits a new edge case. Every reference file in this project should have a `## Gotchas` section that starts empty and fills in from real use.

**Don't state the obvious.** Claude knows Python, knows APIs, knows Stan syntax at a general level. The skill should push Claude out of its defaults — project-specific table names, the specific FIPS trimming pattern, the MOE threshold we chose, why we use 2019 vintage for training. Generic advice wastes context.

**Use the file system for progressive disclosure.** Put detailed reference code in `scripts/`, point to `references/api.md` for function signatures. Tell Claude what files exist; it will read them when needed. Don't dump everything into the top-level SKILL.md.

**Avoid railroading.** Give Claude the information it needs but flexibility to adapt. A skill that specifies `K=10 components` is wrong for exploratory research. A skill that says "here's how to evaluate K, here's the reconstruction error check" is right.

**The description field is for the model, not the human.** It's what Claude scans to decide whether to invoke the skill. Write it as a trigger condition: "Use when fetching ACS data at census tract level for FL/GA/AL."

**Store scripts in the skill.** Giving Claude a working `fetch_tracts.py` is more valuable than explaining how to write one. Claude then spends its turns on composition and decisions, not reconstructing boilerplate.

**On-demand hooks.** Skills can activate hooks only for the duration of the session. Useful for guardrails during dangerous operations (e.g., a `/careful` skill that blocks destructive commands when touching production data).

**Memory.** Skills can store data in log files, JSON, or SQLite. For a data pipeline skill, this means storing state of what's been downloaded so re-runs don't re-fetch. Use `$CLAUDE_PLUGIN_DATA` as the stable storage path (survives skill upgrades).

---

## The Reference Web → Skill Pipeline

Reference files in `docs/references/` are the raw material for skills. The relationship:

- **Reference file** = distilled knowledge (what to know)
- **Skill** = procedure + knowledge + scripts (what to do, step by step, with tooling)

When a reference file has been consulted for the same type of operation 3+ times, that operation is ready to become a skill. The reference file's content (especially its Gotchas section) becomes the core of the skill's SKILL.md.

---

## Skill Structure Template for This Project

```
.claude/skills/<skill-name>/
├── SKILL.md           ← description (for model trigger), what it does, when to use
├── gotchas.md         ← failure modes encountered in practice (starts empty)
├── scripts/           ← Python/R/bash scripts Claude can run directly
└── references/        ← pointer files to docs/references/ entries
```
