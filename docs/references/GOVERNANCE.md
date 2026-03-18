# Reference Web — Governance Rules

This directory is a local, stable knowledge base for the US Political Covariation Model. Its job is to give agents pre-distilled context without requiring web searches or re-reading raw documentation.

These rules exist to prevent the web from becoming a logjam.

---

## The 7 Rules

### 1. Index-first
`RESOURCE_INDEX.md` is the authoritative catalog. Create the index entry at the same time as the file. **A file with no index entry doesn't exist.**

### 2. Entry threshold
A reference earns a file when you have needed the same information **twice**, OR when you are about to start a new pipeline stage and need to capture what you're about to use. Do not capture speculatively.

### 3. Distill, don't dump
Every file is a summary — not a paste, not a raw doc export. Max **150 lines**. If the source has 20 relevant concepts, capture the 5 you'll actually use and link to the source for the rest. A file that could be replaced by just reading the source URL has failed its purpose.

### 4. Provenance header required
Every file must open with a frontmatter block:
```markdown
---
source: <URL or citation>
captured: YYYY-MM-DD
version: <version or edition of source material>
---
```

### 5. One concept per file
If a file covers two distinct topics, split it. The index maps concepts to files, not the other way around.

### 6. Stage-tagged
Every index entry is tagged to the pipeline stage(s) where it's relevant (`S1`–`S6`, or `ALL` for cross-cutting). This is how agents know what to load for a given task.

### 7. Deprecation over deletion
When an approach is rejected or superseded, move the file to `_deprecated/` with a one-line note explaining why. Rejected approaches have **negative knowledge value** — they prevent re-exploring dead ends.

---

## Logjam Warning Signs

If you notice any of these, stop and prune before adding:

- A file has grown past 150 lines without being split
- The index has 4+ entries for one stage with no sub-directory
- You're capturing something you've only needed once
- A file hasn't been opened across the last two pipeline stages
- The information is already in `CLAUDE.md`, `ARCHITECTURE.md`, or inline code comments

---

## The Skill Trigger

When you've consulted the same reference file for the same **type of operation** three or more times, that operation needs a **skill**, not just a reference. Flag it to the user.

A skill encodes the *procedure*. A reference encodes the *knowledge the procedure draws on*. Both are needed; they're not substitutes.

---

## Directory Map

```
docs/references/
├── GOVERNANCE.md          ← this file
├── RESOURCE_INDEX.md      ← stage-to-resource lookup table
├── stan/                  ← Stan language, model patterns, cmdstanpy interface
├── data-sources/          ← Census, ACS, religious surveys, election returns
├── methods/               ← Statistical methods: MRP, NMF, Bayesian workflow
├── r-ecosystem/           ← R packages: brms, rstanarm, survey
├── skills/                ← Skill design guidance and skill taxonomy
└── _deprecated/           ← Rejected approaches, with notes
```

New directories may be added when an existing directory reaches 5+ files on a coherent sub-topic. Add the directory to this map and to `RESOURCE_INDEX.md` at the same time.
