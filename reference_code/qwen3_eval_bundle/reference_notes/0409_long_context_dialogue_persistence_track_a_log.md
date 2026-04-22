04/09 Long-Context Dialogue Persistence Track-A Log

## Context

This log records the long-context dialogue-persistence work added to
`qwen3_moe` on 2026-04-09.

The goal was to move beyond short single-prompt trait steering and add a
benchmark where we can test whether a steered trait remains visible across:

- multi-turn dialogue,
- large neutral prefix context,
- one-time long-context prefill followed by continued dialogue from cache,
- and a separate external judge that scores both trait persistence and
  coherence.

The target benchmark is Michael's multi-turn conversation set:

- `dialogue_persistence_benchmark_michael.py`

The target traits for this benchmark are not the Big Five. They are the six
style traits already used by the long-context judge:

- `warm`
- `formal`
- `assertive`
- `cautious`
- `dismissive`
- `anxious`

The existing `long_context` runner only supported:

- `plain`
- prompt-style steering

and it did not support:

- `canonical_memory`
- `hybrid_post_attn_caa_canonical`
- fresh memory-bank construction for the style traits
- neutral-document bundles with token budgets
- a Track-A-style "select heads first, then steer" workflow

There were also two repo hygiene issues:

- `long_context/run_dialogue_persistence_benchmark.py` imported the missing
  module `dialogue_persistence_benchmark_full`
- `long_context/judge.py` imported a non-existent `long_context.traits`

Those were fixed as part of this work.

## High-Level Design Choice

The main evaluation claim here should now be:

- read-once long-context persistence with cache-preserving continuation

not:

- repeated full-history long-prefix robustness

The workflow was updated after the first implementation so that the Qwen path
now does what we actually want for this benchmark:

- read the steering prompt and neutral long-context prefill once
- build `past_key_values`
- then continue the benchmark conversation turn by turn from cache

This means the long neutral text and the prompt steering message are **not**
re-encoded at every turn in the Qwen benchmark path.

That is the correct setup for the hypothesis we care about here:

- prompt steering may fade because it was only read once
- hidden attention steering may remain active because the patch is applied at
  each generation step while the original long context stays in cache

There is still a fallback full-history code path for non-Qwen models, but the
intended benchmark configuration here is `Qwen/Qwen3-30B-A3B`, and for that
configuration the benchmark is now genuinely a read-once / then-chat test.

## Core Design Decisions

### 1. Separate Neutral Prefill From The First Real User Turn

The long neutral text is **not** appended to the first actual user question in
the new Track-A workflow.

Instead, before dialogue starts, the model receives:

- a synthetic user message containing the neutral document
- a synthetic assistant acknowledgement

Then the real benchmark dialogue begins.

Reason:

- this isolates "long context already in memory" from the semantic content of
  the first benchmark question
- it prevents the first user turn from becoming a weird mixed instruction-plus-
  corpus prompt
- it makes the benchmark better aligned with the intended claim: "the model
  read a lot of neutral material before the conversation started"

This is implemented in:

- `long_context/run_dialogue_persistence_track_a_workflow.py`

with:

- `LONG_CONTEXT_PREFILL_TEMPLATE`
- `LONG_CONTEXT_ACKNOWLEDGEMENT`

### 2. Use Neutral Expository Documents, Not Literary Prose

Michael proposed `the_great_gatsby.txt`.

I did **not** make that the primary long-context source for the new workflow.
Instead I added a manifest of neutral, mostly U.S. government educational or
technical pages.

Reason:

- literary prose can itself steer style
- the benchmark should attribute style drift to the steering method or long-
  context load, not to a strongly voiceful source document
- public-domain or government sources are easier to cache locally in the repo
  and safer to use repeatedly

Current manifest sources:

- NASA atmosphere overview
- USGS water-cycle and groundwater explainers
- NIST AI RMF FAQ / roadmap pages
- two NOAA PFEL pages were attempted, but currently fail TLS hostname checks

Files:

- `long_context/data/neutral_contexts/manifest.json`
- `long_context/download_neutral_contexts.py`
- downloaded text files under
  `long_context/data/neutral_contexts/text/`

Procedure update after inspecting the downloaded corpus:

- treat the cached raw HTML under `raw/` as the source of truth
- extract article-like content instead of taking the largest visible page region
- strip website boilerplate such as gov banners, nav/search/share blocks, and
  long runs of short menu items before budgeting tokens
- refresh the normalized `.txt` artifacts from cached raw HTML whenever the
  cleaner changes

### 3. Budget The Prefix Explicitly

The workflow uses token budgets instead of "one giant document":

- `0`
- `8000`
- `16000`
- `24000`

Reason:

- `Qwen/Qwen3-30B-A3B` has a finite context window
- the benchmark should measure degradation as a function of context budget
- this gives a clean persistence curve instead of a single hard-to-interpret
  point
- the run should fail early if the chosen budget plus the whole dialogue would
  overflow the model window

The bundle builder no longer fills budgets greedily from the top of the
manifest. It now enforces explicit multi-source mixing:

- `8000`: target `4` sources at about `2000` tokens each
- `16000`: target `4` sources at about `4000` tokens each
- `24000`: target `6` sources at about `4000` tokens each

Implementation details:

- sources are chosen in a stratified / manifest-spread way rather than by
  sequentially taking the first documents until the budget is exhausted
- if one chosen source is shorter than its share, the leftover budget is
  water-filled across the remaining selected sources
- this reduces source-specific style/topic confounds and prevents small budgets
  from collapsing onto a single long source

Local smoke validation with a mock tokenizer confirmed the intended exact
allocation pattern:

- `8000 -> 4 x 2000`
- `16000 -> 4 x 4000`
- `24000 -> 6 x 4000`

The workflow now also performs an explicit context-window preflight before
generation. For each selected case it estimates:

- initial tokens for the steering prompt, system prompt, and long-context
  prefill
- plus the tokenized wrapper for each user turn
- plus `max_new_tokens`
- plus a small reserve for the assistant end token

That estimate is compared against
`model.config.max_position_embeddings - context_window_safety_margin`.

New knobs:

- `--context-window-safety-margin`
- `--disable-context-window-preflight`

### 4. Select Heads Once Per Trait, Then Reuse Them Across Budgets/Methods

The new workflow performs head selection once per style trait, then reuses the
same selector artifact across:

- `plain`
- `prompt`
- `canonical_memory`
- `hybrid_post_attn_caa_canonical`
- all requested token budgets

Reason:

- this isolates the effect of long context and method from selector noise
- it makes comparisons across budgets much cleaner
- it matches the intended "Track A" structure: select first, then steer

### 5. Build Fresh Style-Trait Banks Instead Of Reusing Big-Five Banks

The style traits do not map cleanly onto the existing Big-Five bank profiles.
So I added a dedicated style-bank profile file:

- `profiles/dialogue_persistence/style_traits_v1.json`

and a matching helper module:

- `long_context/style_traits.py`

Each trait has:

- high-pole generation card
- low-pole generation card
- high-pole questionnaire card
- low-pole questionnaire card
- phrase sets for selector target/control phrases

The generation cards follow the existing repo pattern:

- `Core:`
- `Behavior:`
- `Guardrail:`

Reason:

- this keeps bank construction aligned with the existing canonical-memory
  workflow
- it avoids vague one-sentence descriptors
- it makes the banks more discriminative and less entangled

### 6. Use A Trait-Agnostic Phrase-Contrast Selector For Style Traits

The current Qwen Track-A selector wrapper is wired to Big-Five trait
resolution. Rather than forcing the Big-Five selector stack to accept the
style traits indirectly, I implemented a new Qwen-safe selector for the style
traits.

File:

- `long_context/style_track_a.py`

The selector:

- builds canonical memory over target phrases and control phrases
- probes selected layers and heads on neutral style prompts
- scores heads by target-vs-control phrase mass
- selects top heads per layer
- ranks layers by summed or mean top-head contrast score
- writes a standard selector artifact with:
  - `selected_layer_ids`
  - `selected_layer_ids_by_score`
  - `selected_layer_head_map`
  - `selected_layer_rho_map`

Current selector mode string:

- `style_phrase_contrast_v1`

The scoring rule is:

- `target_agg(target_phrase_mass_per_slot) - control_weight * control_agg(control_phrase_mass_per_slot)`

with defaults:

- target aggregation: `mean`
- control aggregation: `mean`
- control weight: `1.0`

For low-pole steering, the selector now also flips the target/control phrase
sets instead of always behaving like a high-pole selector. The selector
artifact records the actual requested `target_pole`.

### 7. Match The Qwen Attention Path Used Elsewhere In This Repo

The selector uses the same Qwen-aware canonical-memory coordinate system as the
main repo:

- query states are taken after `q_proj`
- `q_norm` is applied when present
- memory keys are the same pre-RoPE normalized key memory used by canonical
  steering

Reason:

- this keeps head selection faithful to the actual canonical-memory attention
  path rather than selecting on some mismatched proxy
- it is architecturally more consistent with the Qwen3 path already used in
  `qwen3_moe`

The selector metadata records this explicitly:

- `query_norm_expected = true`
- `"Qwen selector note": "Query states are q_proj outputs after q_norm, matched to pre-RoPE normalized memory keys."`

### 8. Prompt Baseline Uses The Same Style Bank Family

The prompt baseline now uses the same style bank family as the hidden steering
methods.

Reason:

- the comparison should be about *delivery mechanism* and retention, not
  about one method getting a richer trait specification than the others

This is implemented via:

- `build_style_prompt_message(...)`

in:

- `long_context/style_track_a.py`

### 9. Hybrid CAA Vectors Are Built From Synthetic Contrastive Style Examples

For `hybrid_post_attn_caa_canonical`, I did not reuse Big-Five contrastive
examples. Instead I generate a style-specific synthetic example set from the
same neutral prompt pool.

Helper:

- `build_style_contrastive_examples(...)`

These examples provide:

- `prompt`
- `positive_response`
- `negative_response`

which are then passed into the existing site-activation vector builder.

Reason:

- the hybrid CAA component needs contrastive positive/negative completions
- the style traits do not already have a stored training corpus in this repo
- synthetic examples are the fastest controlled first version

For low-pole steering, the positive/negative responses are swapped before
building the vector, so the CAA direction is reversed consistently with the
canonical memory direction.

## What Was Implemented

### 1. New Style-Trait Helper Layer

Added:

- `long_context/style_traits.py`

This file defines:

- the six style traits
- discriminative generation and questionnaire bank cards
- phrase sets for selector target/control phrases
- neutral selector prompts
- synthetic contrastive examples for hybrid CAA
- profile loading and resolution helpers

### 2. New Neutral-Context Helper Layer

Added:

- `long_context/neutral_contexts.py`

This file:

- loads the neutral-source manifest
- extracts article-like visible text from HTML
- strips website boilerplate and short link/menu runs
- reads cleaned downloaded `.txt` sources
- builds explicit stratified token-budget context bundles

### 3. Neutral-Document Downloader

Added:

- `long_context/download_neutral_contexts.py`

This script:

- downloads each manifest URL when raw content is missing
- stores raw HTML under `raw/`
- extracts cleaned visible text
- can regenerate `.txt` files from cached raw HTML with `--reuse-raw`
- stores normalized `.txt` files under `text/`
- writes metadata JSON and a download report

Current successful downloads:

- `nasa_earth_atmosphere`
- `usgs_atmosphere_water_cycle`
- `usgs_evaporation_water_cycle`
- `usgs_groundwater_storage`
- `usgs_groundwater_flow`
- `nist_ai_rmf_faq`
- `nist_ai_rmf_roadmap`

Current failures:

- `noaa_atmosphere_ocean_1`
- `noaa_atmosphere_ocean_2`

Failure reason:

- TLS hostname mismatch on `pfeg.noaa.gov`

These failures do not block the benchmark because the existing downloaded set
already supports the intended token budgets.

### 4. New Qwen-Safe Style Track-A Helper Module

Added:

- `long_context/style_track_a.py`

This module provides:

- message formatting and generation helpers
- cache-preserving Qwen incremental dialogue generation helpers
- context-window token-estimation helpers
- a Qwen-safe style phrase selector
- prompt steering message construction
- CSV writing helpers
- canonical-memory / selector support utilities shared by the workflow

### 5. New Main Workflow

Added:

- `long_context/run_dialogue_persistence_track_a_workflow.py`

This is the main new entrypoint.

It does:

1. resolve target style traits
2. load the Qwen model/tokenizer
3. load neutral text records
4. build token-budgeted context bundles
5. select heads once per trait
6. run each requested method at each requested budget
7. preflight the full read-once dialogue length against the model context
   window
8. prefill the steering prompt and long neutral context once, then continue
   the benchmark conversation from cache
9. generate dialogue case responses
10. call the long-context judge for:
   - turn-level trait scores
   - conversation-level persistence and coherence
11. write:
   - selector traces
   - per-case dialogue rows
   - per-turn rows
   - per-run summaries
   - an overall workflow summary

### 6. New Slurm Launcher

Added:

- `long_context/submit_dialogue_persistence_track_a.sh`

This launcher:

- optionally downloads neutral docs first if missing
- accepts environment overrides for:
  - traits
  - methods
  - budgets
  - selector layers
  - gains
  - output directory
- launches the main workflow script

### 7. New Explicit Profile JSON

Materialized:

- `profiles/dialogue_persistence/style_traits_v1.json`

This is generated from the in-repo style-trait definitions and gives an
explicit profile file that can be reused by other scripts.

### 8. Repo Hygiene Fixes

Fixed:

- `long_context/judge.py`
- `long_context/run_dialogue_persistence_benchmark.py`

Specific fixes:

- `judge.py` now falls back to the new style-trait spec instead of importing a
  missing `traits` module
- `run_dialogue_persistence_benchmark.py` now imports Michael's benchmark file
  rather than the missing `dialogue_persistence_benchmark_full`
- the prompt/plain runner now uses the new style prompt helper, imports cleanly
  as a direct script, and follows the same read-once long-context protocol
  when `--run-long-context` is enabled

## Validation Performed

### Static Validation

Passed:

```bash
python -m py_compile \
  long_context/style_traits.py \
  long_context/neutral_contexts.py \
  long_context/style_track_a.py \
  long_context/download_neutral_contexts.py \
  long_context/judge.py \
  long_context/run_dialogue_persistence_benchmark.py \
  long_context/run_dialogue_persistence_track_a_workflow.py
```

### CLI Validation

Confirmed `--help` works for:

```bash
python long_context/download_neutral_contexts.py --help
python long_context/run_dialogue_persistence_benchmark.py --help
python long_context/run_dialogue_persistence_track_a_workflow.py --help
```

### Neutral-Document Download Validation

Ran the downloader with network permission for the successful sources and later
re-ran the text-generation path against cached raw HTML to refresh the cleaned
`.txt` files in place. The current procedure is:

- keep raw HTML under `long_context/data/neutral_contexts/raw/`
- regenerate cleaned `.txt` files with `--overwrite --reuse-raw` after cleaner
  changes
- use the refreshed `.txt` files as the benchmark inputs

### Token-Budget Validation

Confirmed that the new bundle builder still hits exact requested budgets, but
now does so through explicit multi-source allocation rather than greedy
front-filling. A local mock-tokenizer smoke test confirmed:

- `0 -> 0`
- `8000 -> 4 x 2000`
- `16000 -> 4 x 4000`
- `24000 -> 6 x 4000`

### Incremental-Qwen Validation

Validated the cache-preserving Qwen chat formatting and token accounting
against the locally cached tokenizer:

- confirmed the incremental formatter uses Qwen chat-control tokens correctly
- confirmed the assistant turn is explicitly closed with the model EOS /
  `<|im_end|>` token
- confirmed the token-estimation helper and incremental wrappers return
  plausible counts for multi-turn dialogue

### Import / Runner Repair Validation

Confirmed that both:

- `run_dialogue_persistence_benchmark.py`
- `run_dialogue_persistence_track_a_workflow.py`

now import and show help correctly when run as direct scripts from the repo.

## Commands To Run

### A. Download Neutral Contexts

Recommended refresh/download command:

- reuses cached raw HTML when present
- fetches missing raw files when needed
- rewrites the cleaned `.txt` artifacts with the current extraction logic

```bash
cd /nfs/roberts/project/pi_amk266/zl664/Introspection/qwen3_moe

/home/zl664/.conda/envs/transform/bin/python \
  long_context/download_neutral_contexts.py \
  --overwrite \
  --reuse-raw
```

### B. Run The New Track-A Workflow

Example:

```bash
cd /nfs/roberts/project/pi_amk266/zl664/Introspection/qwen3_moe

/home/zl664/.conda/envs/transform/bin/python \
  long_context/run_dialogue_persistence_track_a_workflow.py \
  --model-name Qwen/Qwen3-30B-A3B \
  --target-traits warm \
  --methods plain prompt canonical_memory hybrid_post_attn_caa_canonical \
  --target-pole high \
  --selector-layer-ids 18 22 26 30 34 38 42 \
  --selector-prompt-limit 10 \
  --max-heads-per-layer 5 \
  --max-layers 6 \
  --rho 8 \
  --hybrid-alpha 1.5 \
  --positive-gain 4 \
  --negative-gain 0.1 \
  --layer-weight-schedule middle_heavy \
  --long-context-token-budgets 0 8000 16000 24000 \
  --track-memory-bank-profile-file profiles/dialogue_persistence/style_traits_v1.json \
  --neutral-context-text-dir long_context/data/neutral_contexts/text \
  --context-window-safety-margin 512 \
  --judge-model gpt-4o-mini \
  --output-dir long_context/dialogue_persistence_track_a_runs
```

### C. Submit The Slurm Launcher

```bash
cd /nfs/roberts/project/pi_amk266/zl664/Introspection/qwen3_moe

sbatch long_context/submit_dialogue_persistence_track_a.sh
```

### D. Override The Slurm Launcher By Environment

Example:

```bash
TRAITS="warm anxious" \
METHODS="plain prompt canonical_memory hybrid_post_attn_caa_canonical" \
CONTEXT_BUDGETS="0 8000 16000 24000" \
OUTPUT_DIR="long_context/dialogue_persistence_track_a_runs_v2" \
sbatch long_context/submit_dialogue_persistence_track_a.sh
```

## Outputs

For each trait and run, the workflow writes:

- selector traces and selector artifact
- `dialogue_persistence_rows.csv`
- `dialogue_persistence_turn_rows.csv`
- `dialogue_persistence_summary.csv`
- `run_config.json`

At the trait root:

- `workflow_summary.csv`

At the global output root:

- `workflow_summary.csv`

The workflow summary is intended to support quick comparisons across:

- method
- context budget
- target trait

and includes aggregate metrics such as:

- mean conversation persistence for the target trait
- mean coherence
- mean per-turn target score
- first-half vs second-half target score
- first-quarter vs last-quarter target score
- `last_quarter_minus_first_quarter`

That last metric is a simple persistence-decay proxy.

## Future Note: Rolling Summary / Update Memory Banks

This mechanism already existed in the sibling repo
`/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor`. It was used
in the dialogue-persistence evaluator, not as a generic trait-bank feature
everywhere in that repo.

The design there was:

- keep the main trait bank static
- build a rolling `summary` auxiliary bank from prior dialogue turns
- optionally build a separate `update` auxiliary bank for explicit user
  corrections or preference changes
- refresh those auxiliary banks during the conversation
- mix them with the main trait bank through
  `steering_operator=bank_softmax_mixture`

This is a useful pattern to remember for `qwen3_moe` after we first finish the
cleaner generic attention-steering baselines.

### What The `descriptor` Mechanism Actually Did

The main trait bank was still the normal canonical descriptor memory. The
summary bank did **not** replace it. Instead, the evaluator built an extra
canonical memory object from compact text snippets extracted from the dialogue
history so far, then passed that object into the attention patch as an
auxiliary bank named `summary`.

There was also a separate `update` bank whose job was narrower: capture
explicit user reversals, corrections, or preference updates. So the actual
pattern was:

- static trait bank for the target persona / style direction
- rolling summary bank for compressed conversation state
- optional update bank for direct user corrections

This only activated for the canonical / hybrid methods and only after there
was already some conversation history to summarize.

### Reconstruction Map In `descriptor`

1. CLI surface and defaults

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:155-175`

Those lines define:

- `--summary-bank-mode`
- `--summary-bank-gain`
- `--summary-refresh-every`
- `--summary-max-turn-pairs`
- `--summary-max-chars`
- `--summary-include-assistant`
- `--update-bank-mode`
- `--update-bank-gain`
- `--update-max-chars`

Important detail:

- `summary_bank_mode` defaulted to `off`
- the summary modes were `recent_user_turns` and `user_state`
- the update mode defaulted to `explicit_user_updates`

2. Summary segment construction

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:359-418`

That function is `_summary_bank_segments(...)`.

It first extracts cleaned prior user turns, and optionally assistant turns,
from `conversation_turns`. Then it emits short textual segments:

- `recent_user_turns` mode:
  - segments like `Recent user concern 1: ...`
  - optionally `Recent working guidance 1: ...`
- `user_state` mode:
  - `Conversation topic: ... Initial user goal or concern: ...`
  - `Current user obstacle or request: ...`
  - `Recent user trajectory: ... | ...`
  - optionally `Most recent working guidance: ...`

Two implementation details matter if we copy this later:

- each segment is truncated inline to stay short
- the bank is built from **multiple segments**, not one monolithic summary

3. Auxiliary memory construction

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:610-649`

That helper is `_get_auxiliary_memory(...)`.

This is the key memory-construction call pattern to preserve:

- `descriptor=str(segments[0])`
- `descriptor_variants=tuple(segments)`
- `keep_descriptor_only=False`
- `slot_pooling="concat"`

That means the summary bank was treated as a full wrapped-text memory, not as
just the descriptor span inside a wrapper template.

4. Turn-time refresh and attachment

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:945-1008`

Those lines show the actual runtime pattern:

- only do this for
  `canonical_memory`, `canonical_memory_mlp_input`,
  `canonical_memory_mlp_gate`, `canonical_memory_mlp_dual`, and `hybrid_h2`
- only do it when `conversation_turns` is non-empty
- refresh the active summary segments on turn `0`, every
  `summary_refresh_every` turns, or when there is no active summary yet
- build `CanonicalAuxiliaryMemoryBank(bank_name="summary", ...)`
- build `CanonicalAuxiliaryMemoryBank(bank_name="update", ...)`
- pass both into `_build_canonical_memory_patch(...)`

So the summary bank in `descriptor` was truly a rolling per-conversation
attachment, not a static precomputed bank.

5. Attention-patch mixing requirement

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/canonical_attention_patch.py:143-192`
- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/canonical_attention_patch.py:647-703`

These lines show two important facts:

- auxiliary banks are only supported when
  `steering_operator='bank_softmax_mixture'`
- the summary bank gets its own bank-level logit and then competes with the
  `prompt`, `trait`, and optional `reference` banks inside the bank softmax

So the summary bank is not concatenated into the trait bank. It is a separate
bank in the bank-level mixture.

6. Canonical-memory semantics behind the summary bank

Look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/canonical_memory.py:118-210`

Those lines explain why the construction above works:

- `descriptor_variants` creates multiple phrase / template variants
- `keep_descriptor_only=False` keeps all wrapper tokens
- `slot_pooling="concat"` preserves all selected slots rather than averaging

That is the core reason the summary bank could encode several short summary
clauses as a compact auxiliary memory rather than collapsing them into one
vector.

7. MLP wrapper forwarding

If the remembered runs used the MLP wrapper script instead of the base eval
script, look at:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_mlp_conditioning_eval.py:54-60`
- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_mlp_conditioning_eval.py:131-140`

Those lines simply forward the summary-bank flags into the main dialogue
persistence evaluator.

### What Already Exists In `qwen3_moe`

We already have part of the machinery locally, so this is not a from-scratch
port.

1. The local attention patch already supports auxiliary banks

Look at:

- `canonical_attention_patch.py:132-209`
- `canonical_attention_patch.py:717-770`

Those local lines already define `CanonicalAuxiliaryMemoryBank`, validate
auxiliary-bank names, require `bank_softmax_mixture`, and mix auxiliary banks
into the bank softmax.

So the core patch-side algorithm is already here.

2. The local canonical memory builder already supports the needed construction

Look at:

- `canonical_memory.py:170-198`

Those lines already expose:

- `descriptor_variants`
- `keep_descriptor_only`
- `slot_pooling`
- `memory_variant_mode`

So we can reproduce the same summary-bank build pattern locally without
changing the canonical-memory builder.

3. The local Track-A workflow currently builds one static patch per run

Look at:

- `long_context/run_dialogue_persistence_track_a_workflow.py:277-459`
- `long_context/run_dialogue_persistence_track_a_workflow.py:933-941`

That is `_build_style_patch(...)` plus the run-loop call site. Right now the
workflow builds a single patch object for the selected trait / method / budget
and then uses that same patch for the entire dialogue case.

4. The local Qwen incremental helper keeps one patch active for the whole
dialogue

Look at:

- `long_context/style_track_a.py:23-32`
- `long_context/style_track_a.py:276-376`
- `long_context/run_dialogue_persistence_track_a_workflow.py:462-518`

The important constraint is:

- `generate_incremental_qwen_dialogue(...)` enters
  `with patch.patch_model(model)` once
- then it runs the whole multi-turn dialogue under that single patch
- `_generate_dialogue_case(...)` currently hands it one fixed patch for the
  entire case

So unlike the old `descriptor` evaluator, the current `qwen3_moe` Qwen path
does **not** yet have a natural per-turn hook where the summary bank can be
rebuilt after every completed turn.

### Recommended Port Plan For `qwen3_moe`

If we decide to reproduce the `descriptor` pattern here after generic
attention-steering testing, the clean port plan is:

1. Add summary / update CLI flags to
   `long_context/run_dialogue_persistence_track_a_workflow.py`

Mirror the old `descriptor` flags near the current canonical-memory argument
block around:

- `long_context/run_dialogue_persistence_track_a_workflow.py:134-147`

2. Add segment builders for rolling summary state

Either in the workflow file or in a small helper module, add:

- `_summary_bank_segments(...)`
- optionally `_update_bank_segments(...)`

The `descriptor` implementation to copy conceptually is:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:359-440`

3. Add a cached auxiliary-memory builder

Add a helper analogous to `_get_auxiliary_memory(...)` that calls
`build_canonical_descriptor_memory(...)` with:

- `descriptor_variants=tuple(segments)`
- `keep_descriptor_only=False`
- `slot_pooling="concat"`

The reference implementation is:

- `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py:610-649`

4. Extend `_build_style_patch(...)` to accept auxiliary banks

This should happen in:

- `long_context/run_dialogue_persistence_track_a_workflow.py:277-459`

The simplest version is:

- import `CanonicalAuxiliaryMemoryBank`
- thread an `auxiliary_banks` argument into each
  `CanonicalDescriptorMemoryAttentionPatch(...)`
- leave the `plain` and `prompt` paths unchanged

5. Refactor the Qwen incremental generation path so the patch can refresh per
turn while preserving cache

This is the main local implementation difference from `descriptor`.

The likely touch points are:

- `long_context/style_track_a.py:276-376`
- `long_context/run_dialogue_persistence_track_a_workflow.py:462-518`

The cleanest future refactor is probably one of:

- move the incremental turn loop into the workflow so it can rebuild the patch
  after each completed assistant turn
- or extend `generate_incremental_qwen_dialogue(...)` so it accepts a per-turn
  patch factory / callback rather than one fixed patch object

The important requirement is:

- keep `past_key_values` from the read-once long-context prefill
- but allow the steering patch to change between turns as the rolling summary
  bank changes

6. Log the new bank state into the output rows

When this is implemented locally, also add fields analogous to the old
`descriptor` outputs:

- `summary_bank_mode`
- `summary_segment_labels`
- `summary_segments`
- `update_bank_mode`
- `update_segment_labels`
- `update_segments`

That will make later debugging much easier when checking whether persistence
improved because of the auxiliary bank or for some other reason.

### Practical Bottom Line

The `descriptor` pattern is worth reusing, but the key local difference is
that `qwen3_moe` already has the auxiliary-bank math in the attention patch and
the needed canonical-memory constructor. The missing piece is mainly runtime
assembly:

- build rolling summary text from completed turns
- turn that text into auxiliary canonical memory
- refresh the patch between turns without discarding the cached long-context
  prefill

So later, when revisiting this after the generic steering tests, the first
files to open are:

- `long_context/run_dialogue_persistence_track_a_workflow.py`
- `long_context/style_track_a.py`
- `canonical_attention_patch.py`
- `canonical_memory.py`
- and, for the original reference logic,
  `/nfs/roberts/project/pi_amk266/zl664/Introspection/descriptor/run_dialogue_persistence_benchmark_eval.py`

## Caveats

- I did **not** run a full end-to-end GPU benchmark job in this session.
  Validation covered imports, CLI, download, token budgeting, and workflow
  construction.
- The intended read-once benchmark path is now Qwen-specific. For non-Qwen
  models, the scripts still fall back to full-history re-encoding.
- The benchmark still does not include an `activation` condition in the new
  Track-A workflow, so it is not yet the full `prompt` vs `activation` vs
  `canonical_memory` vs `hybrid` comparison.
- The hybrid CAA vector is built from synthetic style examples rather than a
  curated empirical style-completion corpus. That is acceptable for a first
  version, but it may not be the final best hybrid setup.
- Two NOAA sources remain in the manifest but currently fail download due to
  TLS hostname mismatch. The benchmark does not depend on them.
- Raw HTML re-extraction can still leave some inline-link fragmentation on a
  few pages, but the current `.txt` artifacts under `text/` were refreshed
  through the improved cleaner and are the intended benchmark inputs.

## Summary

The main result of this work is a complete first-pass long-context Track-A
workflow for the style traits used by Michael's dialogue persistence judge.

The benchmark now supports:

- hidden steering with `canonical_memory`
- hidden steering with `hybrid_post_attn_caa_canonical`
- a prompt baseline built from the same style-bank family
- explicit head selection before steering
- cleaned neutral long-context bundles at multiple token budgets
- explicit multi-source stratification for `8k` / `16k` / `24k` budgets
- read-once long-context prefill with cache-preserving dialogue continuation
- explicit context-window preflight before generation
- trait persistence and coherence judging
- reusable Slurm submission

The key design choices were:

- make the benchmark genuinely read-once rather than re-encode-every-turn
- separate neutral prefill from the real first turn
- avoid literary-context style contamination
- strip website boilerplate before counting long-context tokens
- force multi-source mixing instead of front-loading one source
- select heads once and reuse them across budgets and methods
- use dedicated style-trait banks rather than forcing Big-Five banks into the
  wrong benchmark

That gives a coherent baseline for the next step, which is to actually run the
grid and compare persistence curves between:

- `prompt`
- `canonical_memory`
- `hybrid_post_attn_caa_canonical`

under increasing neutral-context load.

## Later 04/09 Update: Smoke Runs And Selector Repair

After the initial implementation, I ran a small anxious-trait smoke sweep with:

- methods: `plain`, `canonical_memory`
- budgets: `0`, `8000`
- small case / skeleton limits

The first completed smoke run was:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious`

That run used the original style selector and surfaced the main selector
problem clearly.

### Old Smoke Run: What Went Wrong

The old selector artifact was:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious/anxious/selectors/head/anxious_style_selector_seed11/selector_artifact.json`

It used:

- `selector_mode = style_phrase_contrast_v1`

and the rule:

- `mean(target_phrase_mass_per_slot) - mean(control_phrase_mass_per_slot)`

The critical failure was that the selector still returned heads even when the
entire candidate set was anti-target. In the anxious smoke run, the selected
heads all had negative contrast scores in:

- `.../head_scores.csv`

So the workflow was effectively steering with "least bad" anti-anxious heads.

This explains the weak and noisy behavior in the first smoke run:

- budget `0`:
  - `plain` mean anxious persistence = `6.0`
  - `canonical_memory` mean anxious persistence = `6.5`
- budget `8000`:
  - `plain` mean anxious persistence = `6.5`
  - `canonical_memory` mean anxious persistence = `5.75`

The turn-level anxious scores stayed near `2.0` almost everywhere, so the
model was still mostly speaking in a warm / supportive / calm coaching style
rather than an obviously anxious one. The slight conversation-level gains in a
few cases should therefore be treated as weak judge noise or a very small
style shift, not as evidence that anxious steering was working well.

### Core Diagnosis

Generation-time steering was already using the correct style-trait
`generation` bank family from:

- `profiles/dialogue_persistence/style_traits_v1.json`

The mismatch was selector-time only:

- generation used the full anxious / calm style cards
- selector used phrase-only target / control proxies

So the real problem was not "wrong evaluation bank" but "selection on a
different proxy than the one used during steering."

### Selector Fix Applied

I replaced the old phrase selector in:

- `long_context/style_track_a.py`

with a new selector that:

- resolves the same `generation` target bank and opposite-pole reference bank
  used later at generation time
- runs short steered generations during selector discovery
- records decode-time diagnostics from the canonical attention patch:
  - `prompt_bank_mass`
  - `trait_bank_mass`
  - `reference_bank_mass`
  - `alignment_margin`
- scores heads by route-margin-plus-alignment instead of phrase contrast
- keeps only positive, route-consistent heads
- fails closed if no positive route-consistent heads exist

This new selector mode is recorded as:

- `style_generation_route_margin_v1`

I also added selector-specific decode knobs to the workflow and Slurm launcher:

- `--selector-diagnostic-max-decode-steps`
- `--selector-max-new-tokens`

and changed selector prompt subsampling so small smoke tests do not just take
the first `N` prompts from the list. Instead they now sample across the whole
prompt pool, which reduces the old bias toward upbeat / welcoming prompts.

Operational updates made during this pass:

- `long_context/submit_dialogue_persistence_track_a.sh` now defaults to:
  - `#SBATCH --partition=priority_gpu`
  - `#SBATCH --account=prio_jks79_zl664`
  - `#SBATCH --gpus=h200:1`
- `long_context/judge.py` and `judge.py` now resolve an OpenAI API key from
  local fallbacks if `OPENAI_API_KEY` is unset, instead of requiring explicit
  environment export every time

### New Selector Validation (`v2`)

The repaired selector outputs are under:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v2`

The new selector artifact is:

- `.../anxious/selectors/head/anxious_style_selector_seed11/selector_artifact.json`

Key validation results from `v2`:

- selector mode is now `style_generation_route_margin_v1`
- selected layers by score are:
  - `26`, `30`, `24`, `22`, `34`, `18`
- the selected layer/head map contains only positive selected heads
- the artifact records the actual anxious generation descriptor as the target
  bank and the calm generation descriptor as the reference bank
- the score tables now show many strongly positive heads instead of all-
  negative contrasts

Concrete selector-health signals from `v2`:

- `288` total candidate heads
- `258` heads with positive score
- `113` heads passing the route-consistency gate
- strongest selected heads are concentrated in mid layers, especially
  `24`, `26`, and `30`
- top selected heads have:
  - route-positive fraction near `1.0`
  - positive `q25_route_margin`
  - large alignment margins
  - much higher trait-bank mass than prompt-bank mass

This is a large qualitative improvement over the original selector. In plain
language:

- old selector: "pick the least calm heads among a bad candidate set"
- new selector: "pick heads that actually route toward the anxious generation
  bank during decode"

### Important Remaining Caveat

The repaired selector is much better aligned with the intended workflow, but
it is not yet the strongest possible "anxious vs calm" selector in the strict
sense.

Why:

- the workflow is still running with `mix_mode = contrastive_neutral`
- in the canonical attention patch, the reference bank is only active when
  `mix_mode = contrastive_opposite`

So in the current `v2` selector run:

- the anxious target bank is active
- the calm reference bank is resolved and recorded in metadata
- but the active routing score is still effectively "trait bank vs prompt bank"
  rather than "trait bank vs calm bank"

This means the new selector is:

- correctly aligned with the current generation setup

but not yet:

- the fully explicit "select heads that prefer anxious over calm" setup

To get that exact behavior, the next upgrade would be:

- switch selector and generation to `mix_mode = contrastive_opposite`

so the calm reference bank participates directly in both selection and
steering.

### Practical Interpretation

At the end of 2026-04-09, the correct interpretation is:

- the first smoke run mostly confirmed the selector mismatch and should not be
  used as evidence that anxious canonical-memory steering works
- the `v2` selector artifact looks substantially healthier and much closer to
  the intended Track-A logic
- the next thing to trust is not the old smoke metrics but a new canonical run
  produced using the repaired selector

### Later 04/10 Update: `v2` Budget-8000 Outcome

The dedicated `v2` budget-`8000` job completed successfully under:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v2/anxious/budget_08000`

The high-level result is:

- anxious persistence improves versus `plain`
- coherence drops substantially

Conversation-level summary:

- `plain` at budget `8000`:
  - mean `persistence_anxious = 6.5`
  - mean `coherence = 9.0`
- `canonical_memory` at budget `8000`:
  - mean `persistence_anxious = 7.75`
  - mean `coherence = 6.25`

So the trait gain at `8000` is real, but it comes with a large quality cost.

Turn-level evidence also shows that the trait shift is not fake:

- `plain` mean turn anxious score is about `2.02`
- `canonical_memory` mean turn anxious score is about `2.63`

This means the canonical run really is sounding more anxious. The problem is
how it achieves that.

#### Failure Mode At `8000`

The main coherence failure is repetitive template collapse, not obvious factual
contradiction.

In the low-coherence canonical cases:

- `career_transition_coach_8_b`
  - `coherence = 5.0`
  - `persistence_anxious = 8.0`
- `career_transition_coach_16_b`
  - `coherence = 5.0`
  - `persistence_anxious = 8.0`

The responses start reusing the same generic anxious/supportive frame across
many turns instead of responding specifically to each new user question. Common
patterns include:

- repeated "it's completely normal"
- repeated "be honest and transparent"
- repeated "one step at a time"

Concrete repetition signals:

- in `8_b`, the phrase `one step at a time` appears in all `8` assistant turns
- in `16_b`, the phrase `one step at a time` appears in all `16` assistant
  turns
- other reusable stock phrases also recur multiple times, especially in the
  longer cases

So the current interpretation of `v2` at budget `8000` is:

- selector repair helped trait persistence
- but the steering is too permissive and encourages a stable anxious template
  rather than faithful turn-by-turn dialogue
- the model becomes more anxious, but less responsive and less coherent

This strengthens the case for the next `v3` change:

- run both selection and generation with `mix_mode = contrastive_opposite`
- score heads with `non_trait_route_margin`
- gate on `q25_non_trait_route_margin > 0`

That stricter objective should prefer heads whose routing beats prompt plus
reference mass, rather than heads that merely sustain a generic anxious mode.

### Later 04/10 Update: `v3` With `contrastive_opposite` At `negative_gain = 0.1`

The first `v3` budget-`0` run used:

- `mix_mode = contrastive_opposite`
- selector score based on `non_trait_route_margin`
- selector gate based on `q25_non_trait_route_margin > 0`
- `negative_gain = 0.1`

This run completed successfully under:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v3_ng0p1`

The result is that `v3` solves the worst `v2` coherence pathology, but at
`negative_gain = 0.1` it becomes too weak.

Observed outcomes at budget `0`:

- `prompt`
  - mean `persistence_anxious = 7.25`
  - mean `coherence = 9.0`
- `canonical_memory` with `v3`
  - mean `persistence_anxious = 6.5`
  - mean `coherence = 9.0`

So `canonical_memory` no longer shows the strong budget-`0` gain seen in `v2`.
It is now:

- weaker than the `prompt` baseline
- only slightly above the original plain baseline
- much cleaner behaviorally, with no obvious coherence penalty

Interpretation:

- `v2` was too permissive and found many heads that supported a stable anxious
  template, which helped trait persistence but hurt coherence badly at long
  context
- `v3` is much stricter and appears to remove many of those overly generic
  heads
- but with `negative_gain = 0.1`, the resulting steering is too conservative
  and does not move the model enough

Selector-level evidence for this overcorrection:

- `v2`
  - `258` positive heads
  - `113` heads passing the selector gate
  - selected `6` layers with `22` total heads
- `v3 ng0p1`
  - `235` positive heads
  - only `34` heads passing the stricter non-trait gate
  - selected `5` layers with only `11` total heads

In effect:

- `v2` selected a broad, high-energy steering set
- `v3 ng0p1` selects a much smaller and cleaner steering set
- that cleaner set preserves coherence but currently under-steers the trait

This means the current next step is still the same one already planned:

- keep the `v3` selector objective
- increase `negative_gain` beyond `0.1`
- compare `0.5` and `1.0` before judging whether `contrastive_opposite`
  fundamentally works or fails

### Later 04/10 Update: Completed `v3` Gain Sweep Readout

Additional `v3` budget-`0` canonical runs completed with:

- `negative_gain = 0.5`
- `negative_gain = 1.0`

Their output roots are:

- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v3_ng0p5`
- `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v3_ng1p0`

The important result is that increasing `negative_gain` did **not** materially
recover the lost steering strength.

Budget-`0` conversation-level summaries:

- `v3 prompt`
  - mean `persistence_anxious = 7.25`
  - mean `coherence = 9.0`
- `v3 canonical`, `negative_gain = 0.1`
  - mean `persistence_anxious = 6.75`
  - mean `coherence = 9.0`
- `v3 canonical`, `negative_gain = 0.5`
  - mean `persistence_anxious = 6.75`
  - mean `coherence = 9.0`
- `v3 canonical`, `negative_gain = 1.0`
  - mean `persistence_anxious = 6.75`
  - mean `coherence = 9.0`

Turn-level anxious scores tell the same story:

- `v3 prompt`
  - mean turn anxious score about `2.54`
- `v3 canonical`, `negative_gain = 0.1`
  - mean turn anxious score about `2.33`
- `v3 canonical`, `negative_gain = 0.5`
  - mean turn anxious score about `2.19`
- `v3 canonical`, `negative_gain = 1.0`
  - mean turn anxious score about `2.38`

So among the finished `v3` canonical runs:

- `negative_gain = 1.0` is slightly best on turn-level anxious score
- but all three `negative_gain` settings are clustered tightly
- none beats the `prompt` baseline at budget `0`

This changes the earlier interpretation. The issue is no longer just
`negative_gain = 0.1` being too small. The broader pattern is:

- `v3` removes the worst `v2` coherence collapse
- but the stricter `contrastive_opposite` plus pure
  `non_trait_route_margin` objective appears to over-prune useful steering
  heads
- increasing `negative_gain` does not fix that by itself

Selector evidence is consistent with this:

- `v2`
  - `258` positive heads
  - `113` gate-passing heads
  - selected `6` layers with `22` total steering heads
- `v3 ng0p1`
  - `235` positive heads
  - `34` gate-passing heads
  - selected `5` layers with `11` total steering heads
- `v3 ng0p5`
  - `232` positive heads
  - `36` gate-passing heads
  - selected `6` layers with `12` total steering heads
- `v3 ng1p0`
  - `233` positive heads
  - `36` gate-passing heads
  - selected `6` layers with `12` total steering heads

So the gain sweep only nudges the selected set. It does not restore the broad,
high-energy steering set seen in `v2`.

### Practical Readout After The Completed Sweep

At this point the best interpretation is:

- `v2 canonical`
  - strongest anxious steering
  - poor stability, especially at long context
- `v3 canonical`
  - clean and coherent
  - underpowered relative to both `v2 canonical` and `v3 prompt`
- `v3 prompt`
  - currently the best stable method among finished runs

This suggests the next iteration should probably **not** be another simple
`negative_gain` sweep. A better next change would be to keep the stricter
non-trait gate but soften the ranking objective, for example:

- require `q25_non_trait_route_margin > 0`
- but rank by `route_margin`, or by a weighted blend of `route_margin` and
  `non_trait_route_margin`

That would preserve the main anti-collapse filter from `v3` without making the
selector as sparse and weak as the current pure non-trait objective.

### Output-Directory Caveat

One operational issue also showed up:

- job `7802229` and job `7802228` both wrote to
  `long_context/dialogue_persistence_track_a_runs_smoke_anxious_v3_ng0p1`

So the later mixed `prompt + canonical` run can overwrite the earlier dedicated
budget-`0` canonical outputs in that directory. Future comparisons should use
unique `OUTPUT_DIR` values per run to avoid ambiguous provenance.

### Later 04/10 Update: `v4` Selector Implementation

Based on the completed `v3` readout, the next selector revision was implemented
in code as a configurable blended objective.

`v4` keeps the strict `v3` gate:

- require positive score
- require `non_trait_route_positive_fraction >= 0.5`
- require `q25_non_trait_route_margin > 0`

But it no longer has to rank heads purely by `non_trait_route_margin`. Instead
it supports a blend:

- `route_blend_margin = alpha * route_margin + (1 - alpha) * non_trait_route_margin`
- score = `mean(max(route_blend_margin, 0) * max(alignment_margin, 0))`

This was added as a new runtime knob:

- `--selector-route-blend-alpha`

with Slurm-wrapper support via:

- `SELECTOR_ROUTE_BLEND_ALPHA`

Interpretation of the new knob:

- `0.0`
  - current `v3` behavior
  - pure non-trait ranking
- `> 0.0`
  - `v4` behavior
  - still filtered by the strict non-trait gate, but ranked by a softer
    blended score

The intended first probe is to use a moderately route-heavy blend such as:

- `SELECTOR_ROUTE_BLEND_ALPHA = 0.7`

This should preserve the main anti-collapse filter from `v3` while restoring
some of the stronger steering signal that `v2` captured.

## 04/10 Benchmark Redesign: Fairer Long-Context Interference Benchmark

After reviewing the initial `v4` smoke results, the main conclusion was:

- the benchmark was not obviously broken
- but it was mostly a **neutral occupancy** test, not yet a strong
  **interference** test

That distinction matters because the observed result was:

- one-shot `prompt` did **not** show the expected collapse under long context

The most plausible explanation was not "prompt is secretly perfect." It was
that the current setup was still too kind to prompt:

- the long prefix was neutral and unrelated
- the prompt baseline lived in a privileged initial position
- the dialogue did not have to retrieve much from the prefix
- and the quality scalar was too saturated to expose subtle failures

So on 04/10 the benchmark was redesigned to make the long-context story more
honest and more diagnostic.

The key principle for this redesign was:

- **do not weaken prompting artificially**

The goal is not to "make prompt fail on purpose." The goal is to create a fair
set of harder conditions where prompt brittleness can show up naturally if it
is real.

### High-Level Reframing

The benchmark is now structured around two conceptually different split
families:

- `neutral_occupancy`
  - the old benchmark, kept as the controlled baseline
  - asks whether a target style can persist when a large amount of irrelevant
    cached text occupies context
- `opposite_style_interference`
  - the new harder split
  - asks whether a target style can persist when the long cached prefix is
    written in the **opposite** interaction style

This means the current benchmark family should be described as:

- a fair occupancy benchmark

not as:

- a definitive test where prompt "should obviously fail"

The redesign keeps that fair occupancy control while adding a much stronger
interference condition.

### Why This Redesign Was Needed

The redesign was motivated by three concrete issues in the initial benchmark.

First, the benchmark could not tell whether the model was really using the long
prefix at all. Since the prefix was intentionally irrelevant, the model could
mostly ignore it and still look strong.

Second, the strongest prompt baseline was too structurally privileged:

- read once at the front
- inserted as a strong prompt-role message
- never forced into a weaker or position-sensitive setting

Third, the existing quality evaluation was too ceilinged:

- `mean_coherence = 9.0` across all methods and budgets
- judge-ok rates saturated at `1.0`

That makes it very hard to distinguish:

- robust high-quality persistence

from:

- mildly repetitive, generic, or weak-but-stable style control

### New Benchmark Axes

The workflow was extended so the benchmark now has explicit controls for:

- prefill mode
- prompt-delivery mode
- case sampling mode
- canary probing
- benchmark presets
- human-validation export

New CLI knobs in
`long_context/run_dialogue_persistence_track_a_workflow.py` include:

- `--benchmark-preset current|smoke_v2|paper_v1`
- `--prefill-mode neutral_occupancy|opposite_style_interference`
- `--case-sampling in_order|stratified`
- `--stylized-context-text-dir`
- `--prompt-middle-fraction`
- `--prompt-refresh-every-k`
- `--enable-canary-probe`
- `--canary-position-fraction`
- `--context-underfill-tolerance`
- `--human-validation-export`

The benchmark presets currently mean:

- `current`
  - preserve legacy behavior unless the new knobs are requested explicitly
- `smoke_v2`
  - force stratified case sampling
  - default `case_limit = 6` if no explicit case limit is given
- `paper_v1`
  - force stratified case sampling
  - automatically enable canary probing
  - tighten context underfill tolerance to at most `5%`

This gives a clean progression from:

- backwards-compatible development runs

to:

- paper-facing runs with stricter integrity checks

### Prompt Baseline Redesign

One major design decision on 04/10 was:

- compare hidden steering against **multiple** explicit prompting baselines,
  not just one strong head-position baseline

The benchmark now supports these explicit prompt baselines:

- `prompt_once_head`
  - the old prompt baseline, preserved as the strongest fair one-shot prompt
- `prompt_once_middle`
  - prompt inserted once around the midpoint of the prefill
- `prompt_refresh_k4`
  - head prompt plus reminder messages before turns `5`, `9`, `13`, and so on

The legacy method name:

- `prompt`

is now normalized internally to:

- `prompt_once_head`

This was an important fairness decision. The benchmark should let prompting be
genuinely strong. If hidden steering only wins against a deliberately weak
prompt, that result is not convincing.

### Prompt Placement And Runtime Implementation

Prompt placement is now a first-class runtime concept rather than a single
hard-coded special case.

The workflow builds a prompt-delivery plan for each case using:

- initial messages
- optional prompt insertions inside the prefill
- optional per-turn prefix reminders

Implementation details:

- `prompt_once_head`
  - inserts the style prompt before the system preamble and before the long
    prefill
- `prompt_once_middle`
  - splits the prefill at a token fraction, default `0.5`
  - emits the left half of the prefill
  - inserts the prompt
  - then emits the rest of the prefill as a continuation
- `prompt_refresh_k4`
  - inserts the prompt initially
  - then adds identical reminder messages as turn-prefix messages every
    `k = 4` turns by default

To support this on the Qwen incremental path, `long_context/style_track_a.py`
was extended with:

- `encode_incremental_qwen_messages`
- `turn_prefix_messages` support in token estimation
- `turn_prefix_messages` support in incremental generation

This means the benchmark can now test not only:

- whether prompting works

but also:

- whether it only works when it occupies the most privileged position

and:

- how much explicit prompt refreshing recovers performance

The workflow now records prompt-delivery audit fields such as:

- `prompt_delivery_requested`
- `prompt_delivery`
- `prompt_message_count`
- `prompt_initial_token_fraction`
- `prompt_insertions_json`

That audit trail matters because prompt placement is now part of the benchmark
condition itself, not just an implementation detail.

### Case Sampling Fix

The initial smoke setup had a subtle but important problem:

- `case_limit` was applied after filtering in a way that tended to select only
  the earliest cases

That made the benchmark look larger and more time-extended than it really was.
In practice, a small smoke run could end up containing only `8`-turn and
`16`-turn cases and skip `24`-turn cases entirely.

To fix that, case sampling now supports:

- `in_order`
- `stratified`

The new stratified mode round-robins across conversation lengths before
applying `case_limit`. So a smoke run can now cover:

- `8`-turn
- `16`-turn
- `24`-turn

cases in the same small budget.

This was necessary because one of the headline claims under discussion was:

- does prompting degrade over time?

That question cannot be answered well if the smoke benchmark never actually
includes the longest conversations.

### Prefix-Use Manipulation Check: Canary Probe

Another major 04/10 design choice was to stop assuming that the model is using
the long prefix just because it was present in cache.

The workflow now supports a canary-based manipulation check.

For nonzero-budget runs, if `--enable-canary-probe` is set:

- a deterministic verification code is inserted into the prefill
- the code is placed at a controlled token fraction, default `0.5`
- after the real dialogue, the workflow asks for that exact code
- the probe is scored separately from the style benchmark

Important implementation details:

- the canary answer is normalized before exact-match scoring
- the canary probe is excluded from the normal style transcript evaluation
- probe results are tracked per conversation and aggregated at run level

New recorded fields include:

- `probe_response`
- `probe_exact_match`
- `probe_correct`
- `canary_expected_answer`
- `canary_position_fraction`

And the workflow summary now reports:

- `canary_probe_accuracy`

This manipulation check is important because it distinguishes:

- "the model carried a large occupied cache"

from:

- "the model actually retained and could access the prefix"

### Opposite-Style Interference Assets

The second major new split added on 04/10 is:

- `opposite_style_interference`

The idea is to keep the semantic content of the long prefix roughly fixed while
changing its **surface style** into the opposite pole of the target trait.

Examples:

- target `anxious/high` uses long-context documents rewritten in
  `anxious/low`
- target `formal/high` uses documents rewritten in `formal/low`

This creates stylistic competition without turning the benchmark into a
semantic retrieval task or a document-topic confound.

To support that, `long_context/neutral_contexts.py` now includes stylized
asset-loading helpers:

- `default_stylized_context_text_dir()`
- `read_downloaded_stylized_contexts(...)`

and a new asset builder was added:

- `long_context/prepare_stylized_contexts.py`

That script rewrites downloaded neutral source texts into controlled style
variants using the generation descriptor for each trait and pole.

Each generated stylized asset stores metadata including:

- source id and source metadata
- trait and pole
- rewrite model
- source and rewritten SHA1
- character counts
- length ratio
- digit-signature match
- descriptor used for the rewrite

This design tries to preserve:

- topic content
- facts
- entities
- numbers

while changing:

- tone
- framing
- style surface form

That is much closer to a real interference benchmark than neutral filler.

### Context Underfill Honesty

The original benchmark had another issue:

- `16000` and `24000` could silently collapse onto the same realized context
  length when the neutral corpus was too small

That makes the headline context-budget curve misleading.

The workflow now treats underfilled context budgets explicitly.

For each budget, it compares:

- requested token budget

against:

- actual realized prefill tokens

If actual tokens fall below the tolerance threshold determined by
`--context-underfill-tolerance`, the workflow:

- prints a warning in normal runs
- raises an error under `paper_v1`

This was an important integrity change. It prevents a paper-facing sweep from
quietly treating:

- "asked for 24k but only got about 14k"

as though it were a valid 24k condition.

### Evaluation Overhaul

The judge was extended because the old conversation-level coherence scalar was
too saturated to be useful.

Turn-level evaluation now scores not only the six target style traits but also:

- `usefulness`
- `specificity`
- `current_turn_relevance`
- `non_genericness`

Conversation-level evaluation still records target-trait persistence and
legacy `coherence`, but now also adds:

- `user_state_consistency`
- `non_repetitiveness`
- `overall_quality`

This makes it easier to catch cases where stronger style persistence is being
achieved by:

- repetitive phrasing
- generic canned coaching
- drift away from user-specific context

The workflow aggregation was updated accordingly. In addition to the earlier
means, it now reports:

- `mean_turn_target_score`
- `mean_turn_usefulness`
- `mean_turn_specificity`
- `mean_turn_current_turn_relevance`
- `mean_turn_non_genericness`
- `mean_user_state_consistency`
- `mean_non_repetitiveness`
- `mean_overall_quality`
- `last_quarter_minus_first_quarter`
- `canary_probe_accuracy`

The `last_quarter_minus_first_quarter` metric is especially important because
it is a direct summary of within-conversation late-vs-early style change.

### Analysis And Human-Validation Tooling

Two additional pieces of support tooling were added so the redesigned benchmark
can be analyzed more like an actual benchmark and less like a one-off smoke
script.

First, a paired-bootstrap analysis script was added:

- `long_context/analyze_dialogue_persistence_results.py`

It loads workflow outputs, normalizes legacy method names, joins per-case
conversation metrics with turn-derived metrics, and writes:

- `paired_bootstrap_summary.csv`

The default baselines are:

- `plain`
- `prompt_once_head`

and the analysis now supports paired deltas for metrics such as:

- target persistence
- quality fields
- canary correctness
- turn-level quality means
- late-minus-early style change

Second, a human-validation subset export path was added:

- `long_context/human_validation.py`
- `long_context/export_human_validation_subset.py`
- `long_context/compare_human_validation.py`

The current export target is a balanced subset over:

- 6 traits
- 2 prefill modes
- 5 methods

at:

- budget `16000`

with one `16`-turn conversation per stratum where available.

The exported files include:

- blinded transcripts
- a key file
- an annotation template

and the comparison script can join averaged human labels against automatic
judge outputs.

### Slurm And Workflow Integration

The main Slurm wrapper was updated so the redesigned benchmark can be launched
without rewriting the command line manually each time.

`long_context/submit_dialogue_persistence_track_a.sh` now defaults to:

- `plain`
- `prompt_once_head`
- `prompt_once_middle`
- `prompt_refresh_k4`
- `canonical_memory`

and also exposes environment or CLI support for:

- `BENCHMARK_PRESET`
- `PREFILL_MODE`
- `CASE_SAMPLING`
- `STYLIZED_CONTEXT_TEXT_DIR`
- `ENABLE_CANARY_PROBE`
- `CONTEXT_UNDERFILL_TOLERANCE`
- `HUMAN_VALIDATION_EXPORT`

This keeps the new benchmark matrix runnable through the same launcher rather
than creating yet another ad hoc driver.

### Files Changed Or Added On 04/10

Core workflow changes landed in:

- `long_context/run_dialogue_persistence_track_a_workflow.py`
- `long_context/judge.py`
- `long_context/style_track_a.py`
- `long_context/neutral_contexts.py`
- `long_context/submit_dialogue_persistence_track_a.sh`

New support files added on 04/10:

- `long_context/prepare_stylized_contexts.py`
- `long_context/human_validation.py`
- `long_context/export_human_validation_subset.py`
- `long_context/compare_human_validation.py`
- `long_context/analyze_dialogue_persistence_results.py`

### What This Redesign Does And Does Not Claim Yet

What it now enables:

- a fair occupancy control split
- a harder opposite-style interference split
- multiple explicit prompt baselines
- a manipulation check for prefix use
- stronger quality evaluation
- uncertainty reporting support
- human-validation export support

What it does **not** yet prove by itself:

- that prompt will definitely fail
- that hidden steering will definitely win in every harder split
- that the current long-context corpus is already sufficient for a final
  paper-quality `24k` sweep

Two practical caveats remain after the 04/10 implementation:

- `opposite_style_interference` requires precomputed stylized assets under
  `long_context/data/stylized_contexts/`
- `paper_v1` will now reject materially underfilled budgets until the context
  corpus is expanded enough to fill them honestly

So the 04/10 work should be understood as:

- the benchmark redesign and implementation landing in code

not yet as:

- the final empirical readout from the new benchmark matrix

### Practical Summary

The benchmark was redesigned because the original result:

- "prompt does not degrade much under long neutral context"

was plausible but too easy to over-interpret.

The revised design keeps that original test as the occupancy control, but adds
the pieces that were missing for a stronger paper claim:

- explicit interference
- prompt-position ablations
- prompt-refresh baselines
- prefix-use manipulation checks
- stricter context-budget honesty
- better quality metrics
- and bootstrap / human-validation support

That is the right next step if the real research question is not just:

- "can style survive a large cached prefix?"

but more specifically:

- "when long context creates real competition, does hidden steering remain more
  stable than prompting?"

### Later 04/10 Update: Assistant-Register Interference Corpus

After inspecting the first stylized-context assets more closely, the main issue
was **not** that the files were literally identical. The issue was that the
underlying source family was still wrong for the main interference claim.

The first `opposite_style_interference` design rewrote objective expository
sources such as:

- NASA atmosphere pages
- USGS water-cycle pages
- NIST AI RMF pages

Those rewrites do differ across traits and poles, but they still remain:

- factual reference prose
- relatively objective in discourse mode
- and somewhat distant from the register the model later uses in dialogue

That is probably too weak for the strongest paper-facing interference split.
The model can plausibly treat:

- "long factual article voice"

as separate from:

- "how I should answer the user right now"

especially for traits like:

- `dismissive`
- `cautious`
- `anxious`

So the benchmark was modified again on 04/10 to add a new corpus family:

- **assistant-register long-context corpora**

The core idea is:

- keep the long prefix semantically unrelated to the benchmark dialogue
- but make its discourse mode look like real assistant answers rather than
  encyclopedia-style reference pages

This should create a much stronger and cleaner interference test.

#### New Prefill Modes

The workflow now supports two additional prefill modes:

- `assistant_register_occupancy`
- `assistant_register_interference`

Interpretation:

- `assistant_register_occupancy`
  - long prefix is neutral assistant-style advice text
- `assistant_register_interference`
  - long prefix is assistant-style advice text rewritten into the **opposite**
    style pole of the target trait

This keeps the occupancy/interference structure from the earlier redesign, but
with a better-matched text family.

#### New Context Assets

A new manifest was added:

- `long_context/data/assistant_register_contexts/manifest.json`

This manifest contains synthetic but realistic long-form assistant-answer topic
prompts on unrelated practical topics such as:

- home Wi-Fi dead zones
- cast-iron skillet care
- indoor herb care
- rain-jacket layering
- family photo backup
- bike tire pressure
- desk lamps
- car emergency kits
- meal-prep containers
- router upgrades
- travel laundry
- beginner composting

These topics were chosen to be:

- clearly outside the benchmark conversation domain
- useful enough to support long-form assistant answers
- and plausible as actual assistant-response content

#### New Builder Script

A new builder was added:

- `long_context/prepare_assistant_register_contexts.py`

This script has two stages.

Stage 1:

- generate a **neutral assistant-register base corpus**
- one long standalone answer per manifest topic
- assistant-like, practical, calm, useful, but not trait-biased

Stage 2:

- rewrite each base answer into trait/pole-specific variants
- preserve content and assistant-answer structure
- but push wording, framing, rhythm, hedging, directness, and emotional tone
  toward the target style

This means the interference corpus is now:

- genre-matched to assistant dialogue
- semantically off-task
- and explicitly stylized for the target trait contrast

The new builder also includes:

- retry logic for malformed or truncated JSON responses
- configurable `response_max_tokens`
- resumable behavior by skipping already-written files

#### Loader And Workflow Changes

`long_context/neutral_contexts.py` was extended with:

- default assistant-register manifest and text-dir helpers
- assistant-register base-text loaders
- assistant-register stylized-text loaders

`long_context/run_dialogue_persistence_track_a_workflow.py` was updated to:

- accept the two new prefill modes
- accept assistant-register manifest / text-dir CLI arguments
- route context loading through a generalized prefill-mode loader
- emit more accurate error messages when assistant-register assets are missing

`long_context/submit_dialogue_persistence_track_a.sh` was updated to:

- pass assistant-register manifest and directory arguments through to the
  workflow
- avoid auto-downloading the old neutral HTML corpus when an assistant-register
  prefill mode is selected

#### Generation Status On 04/10

The new assistant-register corpus generation was started immediately after the
new builder landed.

Local run status before switching to Slurm:

- base assistant-register texts completed: `12 / 12`
- stylized assistant-register texts completed locally: `2`

Because the full stylized pass is longer-running, it was then handed off to a
cluster job so it can finish resumably.

Submitted Slurm job:

- `7858331`

Requested resources:

- account `pi_amk266`
- partition `gpu_h200`

The submitted job continues the same command and skips any already-generated
base or stylized files.

#### Current Direction

At this point the intended benchmark progression is:

1. keep the original neutral factual corpus for the old occupancy control
2. treat the old expository stylized corpus as a plumbing/debug path, not the
   main paper split
3. use the new assistant-register corpora for the main harder occupancy /
   interference comparison

The first recommended smoke comparison after the corpus finishes is:

- `formal`
- `assistant_register_occupancy`
- `assistant_register_interference`
- methods:
  - `plain`
  - `prompt_once_head`
  - `prompt_once_middle`
  - `prompt_refresh_k4`
  - `canonical_memory`

Reason:

- `formal` is a cleaner diagnostic trait than `anxious`
- it should reveal prompt-position and interference effects without the same
  degree of quality entanglement as the more adversarial styles
