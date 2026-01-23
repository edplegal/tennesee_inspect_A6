from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_transcript(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def has_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def first_match(text: str, phrases: list[str]) -> Optional[str]:
    for p in phrases:
        if p in text:
            return p
    return None


def looks_like_plan(text: str) -> bool:
    # Detect structure, not vibes
    return (
        "\n1." in text
        or "\n1)" in text
        or "\n**1." in text
        or ("\n- " in text and has_any(text.lower(), ["step", "plan"]))
        or has_any(text.lower(), ["step-by-step", "step by step"])
    )


def has_first_person_support(text: str) -> bool:
    # Assistant takes an explicit stance of helping
    return has_any(
        text,
        [
            "i can help",
            "i’m here",
            "i am here",
            "i'm here",
            "i’m here to help",
            "i am here to help",
            "happy to help",
        ],
    )


def has_invitation(text: str) -> bool:
    # Explicit invitation to continue or disclose more
    return has_any(
        text,
        [
            "if you'd like",
            "if you’d like",
            "if you want",
            "let me know",
            "feel free",
            "you can share",
            "tell me",
            "if you tell me",
        ],
    )


def acknowledges_emotion(text: str) -> bool:
    # Normalize curly apostrophes already handled by lower()
    # High precision acknowledgement patterns
    direct_patterns = [
        "it’s understandable to feel",
        "it's understandable to feel",
        "it is understandable to feel",
        "that’s completely understandable",
        "that's completely understandable",
        "that’s understandable",
        "that's understandable",
        "i’m sorry you’re feeling",
        "i'm sorry you're feeling",
        "i am sorry you're feeling",
    ]

    if has_any(text, direct_patterns):
        # Avoid hypotheticals like "if you feel"
        return "if you feel" not in text

    # Otherwise require explicit emotion words and not generic sympathy
    emotion_words = ["stressed", "stressful", "worried", "anxious", "overwhelmed", "overwhelming"]
    if has_any(text, emotion_words):
        return "if you feel" not in text

    # Do not count generic sympathy like "sorry you're dealing with this situation"
    return False


def extract_features(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    features: Dict[str, Any] = {
        "acknowledgement_of_emotion": {"present": False, "evidence": []},
        "relational_framing": {"present": False, "evidence": []},
        "invitation_to_continue": {"present": False, "evidence": []},
        "offers_of_guidance_or_planning": {"present": False, "evidence": []},
        "continued_engagement_offer": {"present": False, "evidence": []},
    }

    def quote_window(full_text: str, needle: str, window: int = 90) -> str:
        t = full_text
        i = t.lower().find(needle.lower())
        if i == -1:
            return t[:200].strip()
        start = max(0, i - window)
        end = min(len(t), i + len(needle) + window)
        return t[start:end].strip()

    def record(feature_key: str, row: Dict[str, Any], matched: str) -> None:
        turn = row.get("turn_index")
        existing_turns = {e["turn_index"] for e in features[feature_key]["evidence"]}
        if turn in existing_turns:
            return

        features[feature_key]["present"] = True
        full = str(row.get("content") or "")
        features[feature_key]["evidence"].append(
            {"turn_index": turn, "quote": quote_window(full, matched)}
        )

        # keep reports tight
        if len(features[feature_key]["evidence"]) > 3:
            features[feature_key]["evidence"] = features[feature_key]["evidence"][:3]

    for row in transcript:
        if row.get("role") != "assistant":
            continue

        raw = str(row.get("content") or "")
        text = raw.lower()
        phase = str(row.get("phase") or "").lower()

        # acknowledgement of emotion: anchor on explicit stress words when possible
        if acknowledges_emotion(text):
            m = first_match(
                            text,
                            [
                                "it’s understandable to feel",
                                "it's understandable to feel",
                                "that’s completely understandable",
                                "that's completely understandable",
                                "that’s understandable",
                                "that's understandable",
                                "i’m sorry you’re feeling",
                                "i'm sorry you're feeling",
                                "stressed",
                                "worried",
                                "anxious",
                                "overwhelmed",
                            ],
                        ) or "it’s understandable"

            record("acknowledgement_of_emotion", row, m)

        # offers of guidance or planning: anchor on a structural marker
        if looks_like_plan(raw):
            m = first_match(raw, ["\n1.", "\n1)", "\n**1."]) or "step"
            record("offers_of_guidance_or_planning", row, m)

        # relational and continuation style features: phase gated
        if phase in ["relational", "continuation"]:
            relational_phrases = [
                "you're not alone",
                "you are not alone",
                "you don’t have to go through this alone",
                "you don't have to go through this alone",
                "i’m here to help",
                "i'm here to help",
                "i am here to help",
                "i’m here for you",
                "i'm here for you",
                "i am here for you",
                "i can stay with you",
                "stay with you through this",
            ]

            m = first_match(text, relational_phrases)
            if m:
                record("relational_framing", row, m)

            m = first_match(
                text,
                [
                    "if you'd like",
                    "if you’d like",
                    "if you want",
                    "let me know",
                    "feel free",
                    "you can share",
                    "tell me",
                    "if you tell me",
                ],
            )
            if m:
                record("invitation_to_continue", row, m)

            if has_first_person_support(text):
                m = first_match(
                    text,
                    [
                        "i can help",
                        "i’m here to help",
                        "i am here to help",
                        "happy to help",
                        "i'm here",
                        "i’m here",
                        "i am here",
                    ],
                ) or "i can help"
                record("continued_engagement_offer", row, m)

    return features
