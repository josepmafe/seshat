---
name: genai-review
description: Use when you want to review a genAI spec, implementation plan, or codebase with a panel of expert agents covering software engineering, data engineering, AI/ML, architecture, security, UX, cost, legal, and adversarial perspectives. Invoke after brainstorming or after a major implementation session.
---

# genai-review

Convene an expert review panel for a genAI spec, plan, or codebase. Runs in three phases: parallel domain reviewers → devil's advocate → tech lead synthesis.

## Step 1: Detect the Review Target

Check the following in order:

1. Does `docs/superpowers/specs/` contain a spec file modified in the last session? If yes, note it as a candidate for spec review.
2. Does `docs/superpowers/plans/` contain a plan file modified in the last session? If yes, note it as a candidate for plan review.
3. Does `git diff HEAD~1` or `git diff --staged` return meaningful output? If yes, note it as a candidate for code review.
4. If multiple candidates exist, ask the user: "I found [list candidates]. Which should I review? (spec / plan / code)"
5. If none exists, ask: "What should I review? Please provide a file path or describe what you want reviewed."

For **plan review**: auto-locate the matching spec by looking in `docs/superpowers/specs/` for a file with the same date prefix or matching topic name (e.g. `2026-04-22-genai-review.md` pairs with `2026-04-22-genai-review-design.md`). If no match is found, ask the user.

Once the target is identified, ask: **"Include Legal/Compliance review? (y/n, default: n)"**

## Step 1b: Determine Model Tier

**For spec review and plan review:** always use Haiku for domain reviewers, Sonnet for DA and Tech Lead.

**For code review:**
- Run `git diff HEAD~1 | wc -l` to count diff lines.
- Show the user: "Diff is N lines. Use Opus for DA and Tech Lead? (y/n, default: y if N > 500)"
- If N > 500, default to Opus for DA and Tech Lead; user can override to n.
- If N ≤ 500, default to Sonnet for DA and Tech Lead; user can override to y.
- Domain reviewers always use Sonnet for code review regardless.

Record the chosen tier: `DOMAIN_MODEL`, `DA_MODEL`, `TECHLEAD_MODEL`.

## Step 2: Build the Agent Briefs and Mode Instructions

Determine `{REVIEW_MODE_INSTRUCTIONS}` based on the target type. Three separate values are needed: one for domain reviewers, one for the DA, one for the Tech Lead.

**For spec review:**
- Read the spec file in full.
- Ask: "Any key decisions or constraints I should include in the reviewer briefs?"
- Agent brief = spec content + constraints summary.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"Review this spec for correctness, completeness, and design quality within your domain."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty — no additional mandates)
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty — no additional output sections)

**For plan review:**
- Read the plan file in full.
- Read the matching spec file in full (auto-located or provided by user).
- Agent brief = plan content + spec content.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"You are reviewing an implementation plan against the spec it claims to implement. Your job is to find gaps — requirements in the spec that are missing, underspecified, or incorrectly handled in the plan — and flag any steps in your domain that are vague, risky, or sequenced in a way that could cause problems. The spec is the source of truth; the plan must fully satisfy it."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `"**Additional mandate:** Check whether the plan's task ordering is safe — are there steps that assume earlier steps completed successfully without verifying? Are there missing rollback steps for destructive operations?"`
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = the Spec Coverage Summary template:

```
### Spec Coverage Summary
*Requirements from the spec that are fully covered, partially covered, or missing from the plan.*

| Spec Requirement | Coverage | Notes |
|---|---|---|
| [requirement] | Full / Partial / Missing | [which task addresses it, or why it's missing] |

---
```

**For code review:**
- Use the diff already fetched in Step 1b.
- Identify the spec the code implements (look in `docs/superpowers/specs/` for the matching spec, or ask the user).
- Agent brief = diff + spec content.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"Review this diff for correctness and quality within your domain. The spec is the intended behaviour; flag any divergence."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty)
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty)

## Step 3: Phase 1 — Dispatch Domain Reviewers in Parallel

Read each reviewer prompt from `reviewer-prompts/` in this skill directory. Substitute:
- `{REVIEW_TARGET_DESCRIPTION}` with a one-line description of what is being reviewed (e.g. "The genai-review skill implementation plan, compared against its design spec.")
- `{ARTIFACT_CONTENT}` with the agent brief built in Step 2
- `{REVIEW_MODE_INSTRUCTIONS}` with the domain value from Step 2

Dispatch the following agents **simultaneously** using the Agent tool, each with `model: DOMAIN_MODEL`:

1. Software Engineer — `reviewer-prompts/software-engineer.md`
2. Data Engineer — `reviewer-prompts/data-engineer.md`
3. AI/ML Engineer — `reviewer-prompts/ai-ml-engineer.md`
4. Solutions Architect — `reviewer-prompts/solutions-architect.md`
5. AI System Designer — `reviewer-prompts/ai-system-designer.md`
6. Security Expert — `reviewer-prompts/security-expert.md`
7. Domain SME / Evaluator — `reviewer-prompts/domain-sme.md`
8. UI/UX Designer — `reviewer-prompts/ux-designer.md`
9. FinOps / Cost Analyst — `reviewer-prompts/finops.md`
10. Plan Quality Analyst — `reviewer-prompts/plan-quality-analyst.md` *(only if target type is plan review)*
11. Legal / Compliance — `reviewer-prompts/legal-compliance.md` *(only if user said yes in Step 1)*

Wait for all agents to return before proceeding.

**Evidence filter:** Before passing reports to Phase 2, scan each report. Remove any finding that does not contain an `Evidence:` line. Note how many findings were dropped per agent.

## Step 4: Phase 2 — Devil's Advocate

Read `reviewer-prompts/devils-advocate.md`. Substitute:
- `{PHASE1_REPORTS}` with the filtered Phase 1 reports (one per agent, labelled by role)
- `{ARTIFACT_CONTENT}` with the same artifact brief used in Phase 1
- `{REVIEW_MODE_INSTRUCTIONS}` with the DA value from Step 2

Dispatch the Devil's Advocate agent with `model: DA_MODEL`. Wait for it to return.

Apply the same evidence filter to the DA report.

## Step 5: Phase 3 — Tech Lead Synthesis

Read `tech-lead-prompt.md`. Substitute:
- `{PHASE1_REPORTS}` with the filtered Phase 1 reports
- `{DA_REPORT}` with the filtered DA report
- `{REVIEW_MODE_INSTRUCTIONS}` with the Tech Lead value from Step 2

Dispatch the Tech Lead agent with `model: TECHLEAD_MODEL`. Wait for it to return.

## Step 6: Present the Final Report

Output the Tech Lead report in full. Then add:

```
---
## Review Metadata
- Reviewers: [list of agents dispatched]
- Models: domain=[DOMAIN_MODEL], DA=[DA_MODEL], Tech Lead=[TECHLEAD_MODEL]
- Diff size: [N lines] (code review only)
- Spec file: [path] (plan review only)
- Findings dropped (no evidence): [count per agent]
- DA conflicts flagged: [count]
- Review target: [spec/plan file path or git range]
```
