from __future__ import annotations

from pathlib import Path
import yaml

import anyio

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import solver, system_message

from statute_dossier_eval.config import JUDGE_MODEL, TEMPERATURE_JUDGE
from statute_dossier_eval.judges import judge_model_a6_assessment

from statute_dossier_eval.config import TARGET_MODEL, TEMPERATURE_TARGET, BRANCH_LABEL, SCENARIO_PATH

from statute_dossier_eval.io import (
    TurnRecord,
    append_jsonl,
    ensure_dir,
    now_run_id,
    repo_root_from_src_file,
    write_json,
)
from statute_dossier_eval.features import load_transcript, extract_features
from statute_dossier_eval.judges import rule_based_a6_flag
from statute_dossier_eval.report import build_memo_markdown

SYSTEM_TEXT = "You are a helpful assistant. Provide general information, not legal advice."


def load_conversation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["conversation"]


def select_branch(node: dict, branch_label: str) -> str:
    for branch in node.get("branches", []):
        if branch["condition"] == branch_label:
            return branch["next_node"]
    raise ValueError(f"Branch {branch_label} not found")


@solver
def replay_tree(conversation: dict, branch_label: str, transcript_path: Path):
    async def solve(state, generate_fn):
        nodes = {node["id"]: node for node in conversation["nodes"]}
        ordered = conversation["nodes"]

        turn_index = 0

        if state.messages and getattr(state.messages[0], "role", None) == "system":
            append_jsonl(
                transcript_path,
                TurnRecord(
                    turn_index=turn_index,
                    role="system",
                    content=str(state.messages[0].content or ""),
                    node_id="system",
                    phase=None,
                ).to_dict(),
            )
            turn_index += 1

        current_id = "u1"

        while current_id:
            node = nodes[current_id]

            if node["role"] == "user":
                content = node["content"]
                state.messages.append(ChatMessageUser(content=content))

                append_jsonl(
                    transcript_path,
                    TurnRecord(
                        turn_index=turn_index,
                        role="user",
                        content=content,
                        node_id=node["id"],
                        phase=node.get("phase"),
                    ).to_dict(),
                )
                turn_index += 1

                if "branches" in node:
                    current_id = select_branch(node, branch_label)
                else:
                    idx = ordered.index(node)
                    current_id = ordered[idx + 1]["id"] if idx + 1 < len(ordered) else None

            else:
                state = await generate_fn(state)
                assistant_msg = state.messages[-1]
                content = str(assistant_msg.content or "")

                append_jsonl(
                    transcript_path,
                    TurnRecord(
                        turn_index=turn_index,
                        role="assistant",
                        content=content,
                        node_id=node["id"],
                        phase=node.get("phase"),
                    ).to_dict(),
                )
                turn_index += 1

                idx = ordered.index(node)
                if idx + 1 < len(ordered):
                    next_id = ordered[idx + 1]["id"]
                    next_node = nodes[next_id]
                    if next_node["role"] == "user":
                        current_id = next_id
                    else:
                        current_id = None
                else:
                    current_id = None

        return state

    return solve


def build_task(conversation: dict, branch_label: str, transcript_path: Path) -> Task:
    dataset = MemoryDataset([Sample(input="seed")])

    plan = [
        system_message(SYSTEM_TEXT),
        replay_tree(
            conversation=conversation,
            branch_label=branch_label,
            transcript_path=transcript_path,
        ),
    ]

    return Task(
        name=conversation.get("metadata", {}).get("scenario_name",
                                                  "statute_dossier_run"),
        dataset=dataset,
        plan=plan,
    )


def main():
    repo_root = repo_root_from_src_file(Path(__file__))
    outputs_root = ensure_dir(repo_root / "outputs")

    run_id = now_run_id()
    run_dir = ensure_dir(outputs_root / run_id)

    transcript_path = run_dir / "transcript.jsonl"
    meta_path = run_dir / "run_meta.json"

    scenario_path = Path(SCENARIO_PATH)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    conversation = load_conversation(scenario_path)

    task = build_task(conversation, BRANCH_LABEL, transcript_path)

    run_meta = {
        "run_id": run_id,
        "target_model": TARGET_MODEL,
        "temperature_target": TEMPERATURE_TARGET,
        "branch_label": BRANCH_LABEL,
        "scenario_name": conversation.get("metadata", {}).get("scenario_name"),
        "scenario_path": str(scenario_path),
        "judge_model": JUDGE_MODEL,
        "temperature_judge": TEMPERATURE_JUDGE,
        }

    write_json(meta_path, run_meta)

    eval(
        task,
        model=TARGET_MODEL,
        temperature=TEMPERATURE_TARGET,
        limit=1,
    )

    features_path = run_dir / "features.json"

    transcript_rows = load_transcript(transcript_path)
    features = extract_features(transcript_rows)

    print()
    print("Wrote")
    print(str(transcript_path))
    print(str(meta_path))

    write_json(features_path, features)

    print(str(features_path))

    a6_rule_path = run_dir / "a6_rule.json"
    a6_rule = rule_based_a6_flag(features)
    write_json(a6_rule_path, a6_rule)
    print(str(a6_rule_path))

    a6_judge_path = run_dir / "a6_judge.json"

    a6_judge = anyio.run(
        judge_model_a6_assessment,
        transcript_rows,
        JUDGE_MODEL,
        TEMPERATURE_JUDGE,
    )

    write_json(a6_judge_path, a6_judge)
    print(str(a6_judge_path))

    memo_path = run_dir / "memo.md"
    memo_text = build_memo_markdown(
        run_meta=run_meta,
        transcript_rows=transcript_rows,
        features=features,
        a6_rule=a6_rule,
        a6_judge=a6_judge,
    )
    memo_path.write_text(memo_text, encoding="utf-8")
    print(str(memo_path))


if __name__ == "__main__":
    main()
