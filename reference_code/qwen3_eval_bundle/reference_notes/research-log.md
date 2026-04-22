# Research Log

## 2026-04-11

- Initialized autoresearch state for long-context troubleshooting on `Qwen/Qwen3-30B-A3B`.
- Verified live device is `NVIDIA H200`.
- Confirmed this workspace is not a git repository, so protocol locking is being recorded in files rather than commits.
- Reviewed `long_context/dialogue_persistence_track_a_runs_assistreg_cmp_anxious_skel4/anxious/workflow_summary.csv`.
- Observed baseline ranking at 8k `assistant_register_interference`:
  - `prompt_once_head`: best overall balance.
  - `canonical_memory`: slightly below prompt on persistence and far worse on coherence/non-repetitiveness.
  - `hybrid_post_attn_caa_canonical`: no clear rescue over canonical memory.
  - `plain`: high quality but weak anxious persistence.
- Read `canonical_attention_patch.py` and found a likely mechanism:
  - with `mix_mode=contrastive_neutral`, reference competition is absent;
  - `negative_gain` is inactive;
  - query-adaptive gates are only active when reference memory is used;
  - the canonical trait bank still participates during decode, which can keep injecting style even when content quality degrades.
- Read worst-case transcripts for `career_transition_coach_24_a`; canonical memory collapses into stale template reuse while `prompt_once_head` remains helpful and specific.
- Opened experiment `experiments/long_context_canonical_memory_debug_2026_04_11/`.
- Verified that direct Python in the interactive shell did not expose CUDA to PyTorch, but `srun --jobid=${SLURM_JOB_ID} --overlap --gres=gpu:1 ...` did expose the H200 correctly.
- Added `--max-turns` to [`long_context/run_dialogue_persistence_track_a_workflow.py`](/nfs/roberts/project/pi_amk266/zl664/Introspection/qwen3_moe/long_context/run_dialogue_persistence_track_a_workflow.py) to support cheaper debugging without changing the long-context prefill setup.
- Ran a targeted H200 proxy on first 12 turns of:
  - `career_transition_coach_24_a`
  - `remote_team_conflict_24_a`
  - with `assistant_register_interference`, `budget=8000`, `max_new_tokens=80`
- Proxy results:
  - `prompt_once_head`: persistence `8.0`, coherence `8.5`, non-repetitiveness `7.5`, overall `8.0`
  - `canonical_memory`, `contrastive_neutral`, `rho=8`, `pg=4`, `ng=0.1`: persistence `8.0`, coherence `4.5`, non-repetitiveness `2.5`, overall `3.5`
  - `canonical_memory`, `contrastive_neutral`, `rho=6`, `pg=2`, `ng=0.5`: persistence `7.0`, coherence `7.0`, non-repetitiveness `5.0`, overall `6.0`
  - `canonical_memory`, `contrastive_opposite`, `rho=6`, `pg=2`, `ng=0.5`: persistence `7.0`, coherence `7.0`, non-repetitiveness `4.5`, overall `6.0`
- Immediate conclusion:
  - the collapse is reproducible on a cheap proxy;
  - gentler steering clearly helps;
  - on this proxy, `contrastive_opposite` was not clearly better than the gentler neutral control.
- Submitted full regular-partition H200 follow-up jobs:
  - `7973832`: gentler neutral full run
  - `7973831`: gentler opposite full run
