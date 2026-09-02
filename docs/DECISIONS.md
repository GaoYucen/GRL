# GRL Decision Log

## 2026-09-02 — Use marginal gain as the primary learned target
**Decision:** Center the learning component on conditional marginal gain Δ(v|S), rather than direct final-set/solution prediction.

**Rationale:** Existing validation indicates marginal gain is more learnable and maps naturally to sequential influence-maximization decisions.

**Implication:** Future model evaluation must include conditional ranking quality and downstream decision quality.

## 2026-09-02 — Develop toward a learning-augmented certified oracle
**Decision:** Treat the learned predictor as part of an oracle/decision system with uncertainty, trust, certification, or fallback behavior rather than as a standalone regressor.

**Rationale:** This gives a clearer algorithmic contribution and connects predictive accuracy to reliable influence-maximization decisions and computational savings.

**Alternatives not used as the main route:**
- Direct solution/set prediction as the primary model target.
- Purely adding a generic RL module without first establishing reliable marginal-gain estimation and decision benefit.

## 2026-09-02 — Make project state independent of a single Codex session
**Decision:** Persist research state in `AGENTS.md` and `docs/{RESEARCH_STATE,DECISIONS,EXPERIMENT_LOG,NEXT_STEPS}.md`.

**Rationale:** Multiple computers and ChatGPT/Codex entry points should be able to resume from repository state without relying on local chat history.
