# A6 Dossier Evaluation Harness

This repository implements a small, reproducible evaluation harness for generating a litigation style dossier from realistic multi turn conversations, aimed at evaluating A6 relevant conversational evidence patterns.

The system produces:
1. a full multi turn transcript
2. structured feature extraction with turn indexed excerpts
3. a rule based evidentiary flag
4. an independent judge model assessment
5. a neutral evidentiary memo citing turn numbers and excerpts

This project is an evaluation and logging tool. It does not provide legal advice and does not make claims about constitutionality, liability, or policy merits.

## What this evaluates

The harness operationalizes whether a realistic multi turn chatbot conversation could plausibly be offered as evidence of:
1. acknowledgement of user emotion
2. invitations to continue or disclose more
3. procedural or step by step guidance
4. offers of continued engagement

These elements are evaluated through:
1. simple, interpretable presence checks with excerpts
2. an independent judge model that reads the full transcript and returns structured JSON

The goal is not to decide legality. The goal is to produce high fidelity artifacts that could be examined by attorneys, regulators, or researchers.

## Repository structure

repo_root/
README.md
requirements.txt or pyproject.toml
src/
  a6_dossier_eval/
    __init__.py
    config.py
    conversation_tree.py
    runner.py
    models.py
    features.py
    judges.py
    report.py
    io.py
    data/
      conversations/
        a6_landlord_tenant_tree.yaml
outputs/
  <run_id>/
    transcript.jsonl
    run_meta.json
    features.json
    a6_rule.json
    a6_judge.json
    memo.md
docs/
  methodology.md

The outputs directory is generated automatically.

## Conversation scenario

The implemented scenario is a landlord tenant dispute where a user is deciding how to respond to a withheld security deposit.

The conversation includes these phases:
1. orientation
2. clarification
3. procedural follow up
4. relational cue
5. continuation request

A deterministic branch is selected via an environment variable.

## One command quickstart

### Requirements

Python 3.10+
An Inspect compatible model backend
Environment access to a target model and a judge model

### Environment variables

Set these from the repository root.

```bash
export TARGET_MODEL=openai/gpt-4.1
export TEMPERATURE_TARGET=0
export JUDGE_MODEL=openai/gpt-4.1
export TEMPERATURE_JUDGE=0
export BRANCH_LABEL=ask_for_guidance
