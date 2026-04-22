# Results Status (2026-04-07, updated 2026-04-10)

## Main takeaways

- The strongest cross-trait canonical-memory base sweep is still in `0504_track_a_workflow_results`, but neuroticism improved materially in the `0408_neuroticism_*` follow-up sweeps.
- Canonical memory is strong on agreeableness and openness, moderate on neuroticism, weak on extraversion, and still not convincingly solving conscientiousness.
- Activation remains the cleanest standalone neuroticism baseline, and hybrid canonical-memory plus post-attention CAA now matches it on neuroticism.
- Prompting is the strongest raw questionnaire baseline on agreeableness, conscientiousness, extraversion, and neuroticism, but it pays for that with consistently high non-target drift (`~0.50-0.63`).
- The prompt questionnaire runs also show a repeat pattern: target gain usually comes with lower conscientiousness and openness, so the method looks more like broad persona overwriting than clean trait steering.
- There is no completed openness questionnaire prompt run in the current repo state; the prompt evidence for openness is generation-only.
- The recent `0704_conscientiousness_item_qattn_sweep` is useful mostly as a failure case: every completed canonical-memory run is flat or negative on conscientiousness.
- `0704_neuroticism_item_cues_short_anchor` remains a useful failure-case precursor, but it has now been superseded by the `0408_neuroticism_bank_legacy_sweep` and `0408_neuroticism_hybrid_*` follow-ups.
- The 2026-04-09 conscientiousness follow-ups improve only slightly: the edited-bank canonical sweep reaches `+0.10 / 0.15`, while the returned hybrid runs are still flat at best.
- The latest extraversion trait-specific sweep still tops out at `+0.10 / 0.10`, below the older `0504` raw best canonical result (`+0.20 / 0.15`).

## Best Parameters By Method

### Activation

| Trait | Best config | Target shift | Drift | Interpretation |
| --- | --- | ---: | ---: | --- |
| agreeableness | `activation_l23_s0.50_s11` | `+0.30` | `0.05` | clean low-drift result |
| conscientiousness | `activation_l32_s1.50_s11` | `+0.20` | `0.075` | better than current canonical-memory runs |
| extraversion | `activation_l17_s1.50_s11` | `+0.10` | `0.05` | small but clean |
| neuroticism | `activation_l17_s3.50_s11` | `+0.50` | `0.10` | best activation result overall |
| openness | `activation_l20_s1.50_s11` | `+0.20` | `0.075` | solid but not SOTA here |

### Post-Attention CAA

| Trait | Best config | Target shift | Drift | Interpretation |
| --- | --- | ---: | ---: | --- |
| agreeableness | `post_attn_caa_a1.50_s11_tslast` | `+0.30` | `0.075` | competitive but below best canonical-memory result |
| conscientiousness | `post_attn_caa_a3.50_s11_tslast` | `+0.10` | `0.075` | still weak, but better than current canonical-memory runs |
| extraversion | `post_attn_caa_a0.50_s11_tslast` | `+0.10` | `0.10` | modest |
| neuroticism | `post_attn_caa_a1.50_s11_tslast` | `+0.40` | `0.10` | strongest post-attn result |
| openness | `post_attn_caa_a2.50_s11_tslast` | `+0.10` | `0.075` | underperforms activation and canonical memory |

### Hybrid Canonical Memory + Post-Attention CAA

| Trait | Best config | Target shift | Drift | Interpretation |
| --- | --- | ---: | ---: | --- |
| conscientiousness | `track_a_workflow_sweeps/conscientiousness_hybrid_facet_ablation_v1/rho3_pg2_caal20_lwsmiddle_heavy` | `+0.00` | `0.025` | partial follow-up; current best returned C hybrid run is a low-drift null |
| neuroticism | `0408_neuroticism_hybrid_rho8pg4_layer_scale/caal17_caa1p5` | `+0.50` | `0.10` | current best hybrid result; matches the best activation benchmark and beats pure canonical memory |

Large-`rho` neuroticism hybrid follow-up:

- `0408_neuroticism_hybrid_rho_large/rho14`: `+0.50` shift, `0.225` drift
- `0408_neuroticism_hybrid_rho_large/rho16`: `+0.50` shift, `0.175` drift
- `0408_neuroticism_hybrid_rho_large/rho18`: `+0.50` shift, `0.15` drift

Interpretation:

- Higher `rho` can preserve the stronger hybrid target gain, but it did not
  beat the `rho8_pg4` hybrid anchor on drift.
- `track_a_workflow_sweeps/conscientiousness_hybrid_facet_ablation_v1` is
  partially back (`23 / 39` complete), and no returned run is positive yet.
- There is no returned extraversion hybrid folder under
  `track_a_workflow_sweeps` yet.

### Prompting

Questionnaire prompt baseline (`method=prompt`, `prompt_instruction_style=bank_wrapper`, `prompt_role=system`, `prompt_bank_normalization=log_token_count`):

| Trait | Best config | Target shift | Drift | Interpretation |
| --- | --- | ---: | ---: | --- |
| agreeableness | `prompt_questionnaire_agreeableness_20260404` | `+0.80` | `0.50` | strongest raw questionnaire shift for A, but it lowers C and O materially |
| conscientiousness | `prompt_questionnaire_conscientiousness_20260404` | `+0.40` | `0.525` | beats current activation/post-attn/canonical on raw shift, but not on steering cleanliness |
| extraversion | `prompt_questionnaire_extraversion_20260405` | `+1.40` | `0.50` | strongest raw questionnaire result in the repo |
| neuroticism | `prompt_questionnaire_neuroticism_20260404` | `+1.10` | `0.625` | strongest raw N shift, but also the sloppiest prompt baseline |
| openness | none completed | — | — | no questionnaire prompt run currently available |

Prompt generation baseline (`bigfive_generation_results/prompt_generation_local_judge_all5_internal_20260405`, internal contrastive local judge):

| Trait | Mean margin | Off-target drift | Coherence | Interpretation |
| --- | ---: | ---: | ---: | --- |
| agreeableness | `+3.33` | `0.708` | `5.00` | strongest prompt-generation result |
| conscientiousness | `+2.83` | `0.625` | `5.00` | strong margin, but still broad trait spillover |
| extraversion | `+2.33` | `0.75` | `4.67` | positive signal, noisier than A/C |
| openness | `+1.17` | `0.833` | `4.33` | only prompt evidence for openness; positive but high-drift |
| neuroticism | `-0.67` | `0.625` | `4.67` | failure case: opposite score beats target score |

### Canonical Memory

| Trait | Best config | Target shift | Drift | Interpretation |
| --- | --- | ---: | ---: | --- |
| agreeableness | `rho4_pg4_lwsrank_linear` | `+0.70` | `0.15` | strongest canonical-memory win |
| conscientiousness | `track_a_workflow_sweeps/conscientiousness_canonical_memory_facet_ablation_v2/rho6_pg1p5_lwsmiddle_heavy` | `+0.10` | `0.15` | latest edited-bank follow-up finds only a slight, noisy win |
| extraversion | `rho8_pg4_lwsmiddle_heavy` | `+0.20` | `0.15` | weak/fragile compared with A and N |
| neuroticism | `rho12_pg4_lwsmiddle_heavy` | `+0.40` | `0.15` | current best pure canonical-memory neuroticism result; improved over the old `+0.2` control |
| openness | `rho4_pg4_lwsmiddle_heavy` | `+0.50` | `0.10` | best canonical-memory openness result |

Low-drift neuroticism-specific follow-up worth keeping:

- `0408_neuroticism_bank_legacy_sweep/rho10_pg4_lwsmiddle_heavy`: `+0.30` shift, `0.075` drift

Low-drift openness-specific follow-up worth keeping:

- `track_a_workflow_sweep_openness_v4/rho4_pg0p5_lwsrank_linear`: `+0.30` shift, `0.025` drift
- `track_a_workflow_sweep_openness_v4/rho3_pg1p5_lwsrank_linear`: `+0.30` shift, `0.025` drift
- `track_a_workflow_sweep_openness_v4/rho5_pg1p5_lwsmiddle_heavy`: `+0.40` shift, `0.10` drift

## Latest 2026-04-09 Conscientiousness / Extraversion Follow-Ups

| Trait | Folder | Completion | Best raw returned | Cleaner returned frontier | Observation |
| --- | --- | --- | --- | --- | --- |
| conscientiousness | `track_a_workflow_sweeps/conscientiousness_canonical_memory_facet_ablation_v2` | `32 / 32` | `rho6_pg1p5_lwsmiddle_heavy` at `+0.10 / 0.15` | `rho6_pg4_lwsrank_linear` at `+0.00 / 0.05` | the edited shared bank finds one slight positive run, but only in a noisy regime |
| conscientiousness | `track_a_workflow_sweeps/conscientiousness_hybrid_facet_ablation_v1` | `23 / 39` | none positive yet; current best returned is `rho3_pg2_caal20_lwsmiddle_heavy` at `+0.00 / 0.025` | same | hybrid currently improves cleanliness more than target movement; `caa_layer=20` looks least harmful so far |
| extraversion | `track_a_workflow_sweeps/extraversion_canonical_memory` | `24 / 24` | `rho4_pg5_lwsrank_linear` at `+0.10 / 0.10` | `rho4_pg5_lwsmiddle_heavy` at `+0.00 / 0.05` | the latest trait-specific sweep is still capped at a single modest positive run |

## Latest 2026-04-10 Long-Context Dialogue Persistence Benchmark

Current reference folder:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v4`

Shared run settings:

- model: `Qwen/Qwen3-30B-A3B`
- target trait / pole: `anxious` / `high`
- methods compared: `plain`, `prompt`, `canonical_memory`
- seed: `11`
- decoding: `max_new_tokens=180`, `do_sample=false`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.0`
- judge: `gpt-4o-mini`
- workload: `case_limit=4`, `skeleton_limit=2`, `48` judged turns per config
- long-context budgets: `0`, `8000`, `16000`, `24000`
- selector-derived steering layers: `[22, 24, 26, 30, 42]`
- prompt settings: `prompt_role=system`, `prompt_bank_normalization=log_token_count`
- canonical-memory settings: `rho=8.0`, `positive_gain=4.0`, `negative_gain=0.1`, `layer_weight_schedule=middle_heavy`, `mix_mode=contrastive_opposite`, `prefill_steering=last_token_only`, `steering_operator=bank_softmax_mixture`, `query_gate_scale=1.0`, `memory_role=system`

All `v4` configs have `conversation_judge_ok_rate=1.0`, `turn_judge_ok_rate=1.0`, and `mean_coherence=9.0`.

| Context budget | Actual context tokens | `plain` turn / persistence | `prompt` turn / persistence | `canonical_memory` turn / persistence | Best turn-score method |
| --- | ---: | ---: | ---: | ---: | --- |
| `0` | `0` | `2.042 / 6.00` | `2.396 / 7.25` | `2.354 / 6.50` | `prompt` |
| `8000` | `8001` | `2.042 / 6.50` | `2.250 / 6.75` | `2.417 / 6.75` | `canonical_memory` |
| `16000` | `14174` | `2.000 / 5.75` | `2.229 / 7.00` | `2.208 / 6.75` | `prompt` |
| `24000` | `14174` | `2.021 / 5.75` | `2.271 / 6.75` | `2.208 / 6.75` | `prompt` |
| Average across budgets | — | `2.026 / 6.00` | `2.287 / 6.94` | `2.297 / 6.69` | `prompt` on persistence, `canonical_memory` on turn score |

Interpretation:

- `plain` is weakest at every budget on both turn-level anxious score and conversation-level persistence.
- `prompt` wins at `0`, `16000`, and `24000` on mean turn target score, and has the best average persistence across budgets.
- `canonical_memory` wins at `8000` and has the best average mean turn target score across budgets, with flatter early-vs-late behavior than `prompt`.
- `16000` and `24000` currently use the same actual context length (`14174` tokens), so the `24000` row is not yet a harder stress test than `16000`; the neutral-context corpus saturates before the nominal budget.
- Current long-context conclusion: both steering methods survive distraction materially better than the `plain` baseline, with `prompt` and `canonical_memory` still close enough that the current smoke scale is not decisive.

## Folder Status And Recommendation

| Folder | Current status | Interpretation | Recommendation |
| --- | --- | --- | --- |
| `0504_track_a_workflow_results` | Full 5-trait canonical-memory sweep; strongest overall signal | Keep as the main canonical-memory result set | Keep |
| `0408_neuroticism_bank_legacy_sweep` | New-bank neuroticism follow-up; best pure canonical neuroticism result is now `rho12_pg4_lwsmiddle_heavy` at `+0.40 / 0.15` | High-value neuroticism follow-up and current canonical reference for N | Keep |
| `0408_neuroticism_hybrid_oldheads_layer_scale` | Frozen-old-heads hybrid sweep; established that adding post-attn CAA can beat the old `+0.2` canonical control | Useful bridge from old canonical control to hybrid | Keep |
| `0408_neuroticism_hybrid_rho8pg4_layer_scale` | Current best hybrid neuroticism folder; `caal17_caa1p5` reaches `+0.50 / 0.10` | Best current hybrid result and best neuroticism steering result outside prompting | Keep |
| `0408_neuroticism_hybrid_rho10pg4_layer_scale` | Useful comparison hybrid sweep; same `+0.50` shift is possible with worse drift than the `rho8_pg4` anchor | Valuable comparison set, but not the winner | Keep |
| `0408_neuroticism_hybrid_rho_large` | Large-`rho` hybrid follow-up; `rho14/16/18` all preserve `+0.50` target shift but drift worsens to `0.15-0.225` | Useful confirmation that stronger canonical gain is not the missing ingredient | Keep |
| `track_a_workflow_sweeps/conscientiousness_canonical_memory_facet_ablation_v2` | Edited-bank conscientiousness follow-up; complete at `32 / 32`; best raw run is `+0.10 / 0.15`, while the cleaner frontier remains flat | Small signal increase, but still not a convincing conscientiousness steering solution | Keep |
| `track_a_workflow_sweeps/conscientiousness_hybrid_facet_ablation_v1` | Hybrid conscientiousness follow-up; partial at `23 / 39`; current best returned run is `+0.00 / 0.025` | Useful evidence that hybrid has not yet rescued conscientiousness | Keep until completion |
| `track_a_workflow_sweeps/extraversion_canonical_memory` | Latest trait-specific extraversion sweep; complete at `24 / 24`; best current run is `+0.10 / 0.10` | Useful current reference, but it does not beat the older `0504` raw best | Keep |
| `0504_sweep_8layer` | Extraversion-only 8-layer follow-up; H200 is best, B200/RTX6000 are weaker | Mostly useful as the H200 confirmation run | Keep only `extraversion_h200` |
| `0504_layer_budget_followup` | Mostly flat/negative; only `extraversion_h200_ml6_h4` is clearly positive | Good negative evidence, but only one setting really matters | Keep only `extraversion_h200_ml6_h4` |
| `0504_gsm8k_results` | One downstream eval; canonical memory hurts GSM8K vs plain (`0.49` vs `0.705`) | Useful warning about capability tradeoff | Keep |
| `0704_conscientiousness_item_qattn_sweep` | All completed canonical-memory runs are `<= 0` shift; two runs incomplete | Useful as a failure-analysis folder, not as a winning sweep | Keep completed runs, delete incomplete `ng` runs |
| `0704_neuroticism_item_cues_short_anchor` | Early neuroticism selector experiment: `+0.10` neuroticism shift, `0.65` drift | Experimental precursor only; superseded by the `0408_neuroticism_*` follow-ups | Keep for reference |
| `bigfive_activation_sweep_rtx6000` | Best activation benchmark set | High-value method benchmark | Keep |
| `bigfive_post_attn_caa_sweep` | Best post-attention CAA benchmark set | High-value method benchmark | Keep |
| `bigfive_generation_results` | `all5_internal` shows prompt-generation wins on A/C/E/O and a neuroticism failure; smoke folder is empty/redundant | Useful prompt-generation reference, but not a complete benchmark suite | Keep `all5_internal`, delete smoke |
| `bigfive_questionnaire_openness_prompt_activation_benchmark` | Small benchmark summary | Useful comparison point | Keep |
| `bigfive_questionnaire_openness_selector` | Single weak selector check | Small, but largely superseded | Keep if space does not matter |
| `bigfive_questionnaire_openness_tracka_v2_selector` | Small selector sanity check | Small, but superseded by later sweeps | Keep if space does not matter |
| `prompt_questionnaire_agreeableness_20260404` | Prompt baseline reaches `+0.80` target shift with `0.50` drift | Best raw A questionnaire baseline, but much less selective than canonical memory | Keep |
| `prompt_questionnaire_conscientiousness_20260404` | Prompt baseline reaches `+0.40` target shift with `0.525` drift | Best raw C questionnaire baseline currently present, but highly entangled | Keep |
| `prompt_questionnaire_extraversion_20260405` | Prompt baseline reaches `+1.40` target shift with `0.50` drift | Strongest raw questionnaire result in the repo | Keep |
| `prompt_questionnaire_neuroticism_20260404` | Prompt baseline reaches `+1.10` target shift with `0.625` drift | Strong raw N baseline, but too much collateral movement to call clean steering | Keep |
| `track_a_workflow_sweep_openness` | Single early openness run | Redundant after `0504_*` and `openness_v4` | Delete |
| `track_a_workflow_sweep_openness_v2` | Older openness sweep with broader coverage | Useful historically, but dominated by `0504_*` and `openness_v4` | Delete |
| `track_a_workflow_sweep_openness_v3` | Baseline-only / no canonical-memory summaries | Low value | Delete |
| `track_a_workflow_sweep_openness_v4` | Best low-drift openness-specific follow-up | Still useful despite being older | Keep |
| `track_a_workflow_sweep_neuroticism_v1` | Early neuroticism sweep with small positive shifts | Not the best set, but still a usable reference | Keep |
| `track_a_workflow_sweep_neuroticism_v5_qgs_ng` | Large sweep, flat `0.0` target shifts, mislabeled as conscientiousness | Looks misconfigured and low-signal | Delete |
| `track_a_workflow_results` | Mixed old/debug folder; root report empty, smoke empty, retry negative | Low-value clutter | Delete |
| `prompt_questionnaire__20260404` | Empty | No value | Delete |

## Cleanup Applied

Applied on 2026-04-07. Approximate space recovered: `~558 MB`.

Deleted top-level folders:

- `prompt_questionnaire__20260404`
- `track_a_workflow_results`
- `track_a_workflow_sweep_openness`
- `track_a_workflow_sweep_openness_v2`
- `track_a_workflow_sweep_openness_v3`
- `track_a_workflow_sweep_neuroticism_v5_qgs_ng`

Deleted nested low-value folders:

- `bigfive_generation_results/prompt_generation_local_judge_smoke_openness_20260405_v2`
- `0704_conscientiousness_item_qattn_sweep/rho6_pg4_ng0p1_qg0p5`
- `0704_conscientiousness_item_qattn_sweep/rho6_pg4_ng0p1_qg1`
- `0504_sweep_8layer/extraversion_b200`
- `0504_sweep_8layer/extraversion_rtx6000`
- `0504_layer_budget_followup/extraversion_h200_max10`
- `0504_layer_budget_followup/extraversion_h200_max12`
- `0504_layer_budget_followup/extraversion_h200_ml10_h4`
- `0504_layer_budget_followup/extraversion_h200_ml12_h4`
- `0504_layer_budget_followup/extraversion_h200_ml12_h6`
- `0504_layer_budget_followup/extraversion_h200_ml4_h2`
- `0504_layer_budget_followup/extraversion_h200_ml4_h4`
- `0504_layer_budget_followup/extraversion_h200_ml8_h4`
