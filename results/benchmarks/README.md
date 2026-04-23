# Benchmarks

This directory is the clean pointer target for collaborator-facing results.
Large support artifacts and logs stay ignored; the run log for the tmux job is
present locally as `run_benchmarks_tmux.log` but ignored by Git.

## GSM8K

| Method set | Path | Scope |
| --- | --- | --- |
| `plain`, `prompt` | `gsm8k/plain_prompt_768/` | vLLM, 100 items per trait, six style traits, `max_new_tokens=768`. |
| `post_attn_caa` | `gsm8k/post_attn_caa_768/` | vLLM, 100 items per trait, six style traits, `max_new_tokens=768`. |
| `canonical_memory` | `gsm8k/canonical_memory_rho2p5_pg2p5_768/` | vLLM, 100 items per trait, six style traits, `rho=2.5`, `positive_gain=2.5`, peak boost about `+1.56` logits. |

Each GSM8K run keeps the usual layout:

```text
{trait}/high/{method}/generations.jsonl
{trait}/high/{method}/generations.csv
{trait}/high/summary.json
all_summaries.json
all_summaries.csv
```

## MMLU

| Method set | Path | Scope |
| --- | --- | --- |
| `plain`, `prompt`, `post_attn_caa`, `canonical_memory` | `mmlu/smoke_plain_prompt_caa_canonical_rho2p5_pg2p5/` | vLLM smoke, `warm`, subjects `abstract_algebra` and `anatomy`, `max_items=10`, `few_shot_k=5`; canonical uses `rho=2.5`, `positive_gain=2.5`. |

## Notes

- `post_attn_caa` uses the existing vLLM CAA artifacts.
- `canonical_memory` uses the combined 2.5/2.5 artifact root staged on the GPU node under `results/raw/`.
- The MMLU vLLM runner was added as `vllm_port/run_mmlu_vllm_intervention_eval.py` so the MMLU smoke does not fall back to Transformers.
