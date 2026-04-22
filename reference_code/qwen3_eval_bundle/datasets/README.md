# Datasets

This bundle keeps evaluation data in plain local JSONL so runs do not depend on the Hugging Face cache layout.

## Included now

- `gsm8k/test.jsonl`
  - Source: local cached `gsm8k` parquet snapshot that was already present in the packaging workspace.
  - Schema: `question`, `answer`, `gold_final_answer`

## MMLU status

- `mmlu/dev.jsonl`
- `mmlu/test.jsonl`

Those files are placeholders in this bundle. The packaging workspace did not contain a local cached copy of MMLU, so I could not vendor the raw dev/test rows directly.

The included runner expects this schema:

- `subject`: string
- `question`: string
- `choices`: JSON list of 4 answer strings
- `answer`: `A` / `B` / `C` / `D`

To populate the files from the official Hugging Face dataset:

```bash
python datasets/scripts/prepare_mmlu_jsonl.py
```

That script targets the official `cais/mmlu` dataset and its `all` config, writing:

- `datasets/mmlu/dev.jsonl`
- `datasets/mmlu/test.jsonl`

After that, the MMLU runner works fully offline against the local JSONL files.
