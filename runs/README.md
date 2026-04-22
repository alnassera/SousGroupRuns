# Runs

Tracked eval result folders live here.

The expected folder layout for downstream evals is:

```text
runs/<run_name>/
  config.json
  all_generations.csv
  all_summaries.csv
  all_summaries.json
  <trait>/high/...
```

The expected folder layout for long-context evals is:

```text
runs/<run_name>/
  workflow_summary.csv
  <trait>/workflow_summary.csv
  ...
```

Result folders are ignored by default so large scratch runs do not enter Git accidentally. Add intentional results with `git add -f runs/<run_name>`.
