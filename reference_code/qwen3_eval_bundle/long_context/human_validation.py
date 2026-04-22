from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


_VALIDATION_METHODS = (
    "plain",
    "prompt_once_head",
    "prompt_once_middle",
    "prompt_refresh_k4",
    "canonical_memory",
)
_VALIDATION_PREFILL_MODES = ("neutral_occupancy", "opposite_style_interference")
_VALIDATION_TRAITS = ("warm", "formal", "assertive", "cautious", "dismissive", "anxious")


def _normalize_method_name(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized == "prompt":
        return "prompt_once_head"
    return normalized


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_human_validation_subset(
    *,
    output_root: str | Path,
    export_dir: str | Path,
    target_budget: int = 16000,
) -> Dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    export_root = Path(export_dir).expanduser().resolve()
    summary_rows = _read_csv(root / "workflow_summary.csv")
    if not summary_rows:
        raise FileNotFoundError(f"No workflow_summary.csv found under {root}.")

    blinded_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    annotation_template_rows: List[Dict[str, Any]] = []

    for trait in _VALIDATION_TRAITS:
        for prefill_mode in _VALIDATION_PREFILL_MODES:
            for method in _VALIDATION_METHODS:
                candidates = [
                    row
                    for row in summary_rows
                    if str(row.get("target_trait", "")) == trait
                    and str(row.get("prefill_mode", "neutral_occupancy")) == prefill_mode
                    and _normalize_method_name(str(row.get("method", ""))) == method
                    and int(float(row.get("context_token_budget", 0) or 0)) == int(target_budget)
                ]
                if not candidates:
                    continue
                run_dir = Path(candidates[0]["run_dir"]).expanduser().resolve()
                detailed_rows = _read_csv(run_dir / "dialogue_persistence_rows.csv")
                chosen = None
                for row in detailed_rows:
                    if int(float(row.get("num_turns", 0) or 0)) == 16:
                        chosen = row
                        break
                if chosen is None and detailed_rows:
                    chosen = detailed_rows[0]
                if chosen is None:
                    continue

                sample_id = "HV-" + hashlib.sha1(
                    f"{trait}|{prefill_mode}|{method}|{chosen.get('case_id','')}".encode("utf-8")
                ).hexdigest()[:12].upper()
                transcript = str(chosen.get("transcript", "")).strip()
                if not transcript:
                    continue
                auto_target_key = f"persistence_{trait}"
                blinded_rows.append(
                    {
                        "sample_id": sample_id,
                        "transcript": transcript,
                    }
                )
                key_rows.append(
                    {
                        "sample_id": sample_id,
                        "target_trait": trait,
                        "prefill_mode": prefill_mode,
                        "method": method,
                        "case_id": str(chosen.get("case_id", "")),
                        "num_turns": int(float(chosen.get("num_turns", 0) or 0)),
                        "run_dir": str(run_dir),
                        "auto_target_persistence": chosen.get(auto_target_key, ""),
                        "auto_overall_quality": chosen.get("overall_quality", ""),
                        "auto_user_state_consistency": chosen.get("user_state_consistency", ""),
                        "auto_non_repetitiveness": chosen.get("non_repetitiveness", ""),
                    }
                )
                annotation_template_rows.append(
                    {
                        "sample_id": sample_id,
                        "rater_id": "",
                        "target_trait_persistence": "",
                        "overall_quality": "",
                        "user_state_consistency": "",
                        "non_repetitiveness": "",
                        "notes": "",
                    }
                )

    _write_csv(export_root / "blinded_samples.csv", blinded_rows)
    _write_csv(export_root / "key.csv", key_rows)
    _write_csv(export_root / "annotation_template.csv", annotation_template_rows)
    manifest = {
        "output_root": str(root),
        "target_budget": int(target_budget),
        "sample_count": len(blinded_rows),
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
