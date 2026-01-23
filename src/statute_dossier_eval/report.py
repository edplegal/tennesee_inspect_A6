from __future__ import annotations

from typing import Any, Dict, List


def _row_by_turn(transcript_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in transcript_rows:
        ti = r.get("turn_index")
        if isinstance(ti, int):
            out[ti] = r
    return out


def _short_quote(text: str, limit: int = 160) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _md_escape_pipe(text: str) -> str:
    return (text or "").replace("|", "\\|")


def build_memo_markdown(
    run_meta: Dict[str, Any],
    transcript_rows: List[Dict[str, Any]],
    features: Dict[str, Any],
    a6_rule: Dict[str, Any],
    a6_judge: Dict[str, Any],
) -> str:
    run_id = str(run_meta.get("run_id", ""))
    target_model = str(run_meta.get("target_model", ""))
    judge_model = str(run_meta.get("judge_model", run_meta.get("JUDGE_MODEL", "")))  # optional
    branch_label = str(run_meta.get("branch_label", ""))

    judge_score = str(a6_judge.get("score", "borderline"))
    judge_rationale = str(a6_judge.get("rationale", "")).strip()
    judge_cited_turns = a6_judge.get("cited_turns", []) or []
    valid_json = bool(a6_judge.get("valid_json", False))

    rule_flag = bool(a6_rule.get("a6_flag", False))
    rule_rationale = str(a6_rule.get("a6_rationale", "")).strip()
    rule_snips = a6_rule.get("evidence_snippets", []) or []

    turns = _row_by_turn(transcript_rows)

    evidence_rows: List[Dict[str, Any]] = []

    # Rule based evidence rows
    for s in rule_snips:
        ti = s.get("turn_index")
        if not isinstance(ti, int):
            continue
        quote = str(s.get("quote") or "")
        feat = str(s.get("feature") or "")
        evidence_rows.append(
            {
                "turn_index": ti,
                "quote": quote if quote else str(turns.get(ti, {}).get("content") or ""),
                "why": f"Rule based feature present: {feat}.",
                "source": "rule",
            }
        )

    # Judge cited turns, add any not already included
    existing_turns = {r["turn_index"] for r in evidence_rows if isinstance(r.get("turn_index"), int)}
    for ti in judge_cited_turns:
        if not isinstance(ti, int):
            continue
        if ti in existing_turns:
            continue
        content = str(turns.get(ti, {}).get("content") or "")
        evidence_rows.append(
            {
                "turn_index": ti,
                "quote": content,
                "why": "Judge cited this turn as supporting its assessment.",
                "source": "judge",
            }
        )

    # Sort for readability
    evidence_rows.sort(key=lambda r: (r.get("turn_index", 10**9), r.get("source", "")))

    # Build excerpt table
    table_lines: List[str] = []
    table_lines.append("| Turn | Excerpt | Why it matters | Source |")
    table_lines.append("| ---: | :------ | :------------- | :----- |")

    for r in evidence_rows:
        ti = r.get("turn_index")
        quote = _md_escape_pipe(_short_quote(str(r.get("quote") or "")))
        why = _md_escape_pipe(_short_quote(str(r.get("why") or "")))
        src = str(r.get("source") or "")
        table_lines.append(f"| {ti} | {quote} | {why} | {src} |")

    # Counterarguments
    counter_lines: List[str] = []
    counter_lines.append("1. The assistant repeatedly states the information is general and not legal advice, which could be offered to argue the interaction is informational rather than regulated conduct.")
    counter_lines.append("2. The conversation content is procedural and focuses on documentation, options, and resources, which could be characterized as standard customer support style guidance rather than a relational or therapeutic engagement.")
    counter_lines.append("3. The assistant does not explicitly claim authority, does not establish a professional relationship, and does not request private identifying details, which could be used to argue the engagement is limited in scope.")

    memo_lines: List[str] = []

    memo_lines.append("# Evidentiary memo")
    memo_lines.append("")
    memo_lines.append("## Run metadata")
    memo_lines.append("")
    memo_lines.append(f"Run id: {run_id}")
    memo_lines.append(f"Branch label: {branch_label}")
    memo_lines.append(f"Target model: {target_model}")
    memo_lines.append(f"Judge model: {judge_model}".rstrip())
    memo_lines.append("")
    memo_lines.append("## Summary")
    memo_lines.append("")
    memo_lines.append(f"Rule based flag: {str(rule_flag).lower()}")
    memo_lines.append(f"Rule rationale: {rule_rationale}")
    memo_lines.append("")
    memo_lines.append(f"Judge score: {judge_score} (valid_json: {str(valid_json).lower()})")
    memo_lines.append(f"Judge rationale: {judge_rationale}")
    memo_lines.append(f"Judge cited turns: {judge_cited_turns}")
    memo_lines.append("")
    memo_lines.append("## Excerpts that could plausibly support an A6 style claim")
    memo_lines.append("")
    memo_lines.extend(table_lines)
    memo_lines.append("")
    memo_lines.append("## Counterarguments and limitations")
    memo_lines.append("")
    memo_lines.extend(counter_lines)
    memo_lines.append("")
    memo_lines.append("## Notes")
    memo_lines.append("")
    memo_lines.append("This memo is generated for evaluation and logging purposes. It is not legal advice.")
    memo_lines.append("The quoted excerpts are drawn from the stored transcript artifact and are included to preserve an evidentiary trail with turn indices.")

    return "\n".join(memo_lines)
