# Findings

## Current Understanding

The current failure mode is not simple long-context forgetting. On `dialogue_persistence_track_a_runs_assistreg_cmp_anxious_skel4` with `assistant_register_interference` at an 8k token budget, `canonical_memory` keeps the anxious style active but degrades response quality badly. `prompt_once_head` currently dominates on the relevant combined objective: it keeps strong anxious persistence while preserving coherence, non-repetitiveness, and overall helpfulness.

This failure reproduces on a cheaper H200 debug proxy built from two representative 24-turn cases truncated to the first 12 turns. That proxy is now the preferred inner-loop evaluator for tuning because it preserves the long-context prefill behavior while finishing fast enough to iterate.

## Patterns And Insights

- `canonical_memory` and `hybrid_post_attn_caa_canonical` both look like oversteering failures rather than understeering failures.
- Conversation-level anxious persistence is moderate-to-high for canonical memory, but turn-level target score is still below `prompt_once_head`.
- The qualitative transcripts show mode collapse into repeated reassurance boilerplate. This explains the large drop in coherence and non-repetitiveness.
- In the current implementation, `contrastive_neutral` does not activate the reference-bank path. That means:
  - `negative_gain` is effectively irrelevant;
  - query-adaptive gates are not doing meaningful control work;
  - steering is closer to an always-on trait-bank bias during decode.
- A pilot comparison suggests the main win comes from gentler steering strength, not necessarily from switching mix mode:
  - `neutral rho8 pg4`: overall `3.5`
  - `neutral rho6 pg2`: overall `6.0`
  - `opposite rho6 pg2`: overall `6.0`
- `contrastive_opposite` did slightly improve turn-level target score over the gentler neutral control, but not enough on the 2-case proxy to call it the clear winner.

## Lessons And Constraints

- Do not treat the current `rho/positive_gain/negative_gain` tuple as a fully expressive tuning surface under `contrastive_neutral`; some controls are structurally inactive.
- Long-context evaluation must optimize a joint objective, not persistence alone. A method that raises style persistence while collapsing content is not a win.
- Use H200 for all fresh generation comparisons because hardware choice can affect generation behavior and throughput.
- Prefer `gpu_h200` / `pi_amk266` for batch work; keep `priority_gpu` fallback minimal.
- For interactive debugging, use the new `--max-turns` option instead of full 24-turn runs.

## Open Questions

- Does `contrastive_opposite` recover quality by activating reference competition and adaptive gating?
- Is the problem primarily decode-time steering, excessive layer budget, excessive gain, or all three?
- Does the gentler neutral setting stay improved on the full anxious benchmark, or is the small proxy flattering it?
- Can a gentler canonical method beat `prompt_once_head` specifically at longer budgets where prompt-only methods may drift?
