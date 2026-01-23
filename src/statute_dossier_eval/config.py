import os

TARGET_MODEL = os.getenv("TARGET_MODEL", "openai/gpt-4.1")
TEMPERATURE_TARGET = float(os.getenv("TEMPERATURE_TARGET", "0"))
BRANCH_LABEL = os.getenv("BRANCH_LABEL", "ask_for_guidance")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4.1")
TEMPERATURE_JUDGE = float(os.getenv("TEMPERATURE_JUDGE", "0"))

# Path to a conversation scenario YAML, relative to the repository root by default
SCENARIO_PATH = os.getenv(
    "SCENARIO_PATH",
    "src/statute_dossier_eval/data/conversations/a6_landlord_tenant_tree.yaml",
)
