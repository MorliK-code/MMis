from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, list[str]] = {
    "memory": [
        "tests/test_memory_architecture_audit.py",
        "tests/test_memory_backfill.py",
        "tests/test_memory_claims.py",
        "tests/test_memory_dialog_episode_builder.py",
        "tests/test_memory_dialog_episode_models.py",
        "tests/test_memory_dialog_episode_retriever.py",
        "tests/test_memory_document_ingest.py",
        "tests/test_memory_document_models.py",
        "tests/test_memory_document_retrieval.py",
        "tests/test_memory_e2e_scenarios.py",
        "tests/test_memory_ingest_analyzer.py",
        "tests/test_memory_replay_cases.py",
        "tests/test_memory_retrieval_foundation.py",
        "tests/test_memory_storage_profile.py",
        "tests/test_memory_write_policy.py",
        "tests/test_response_pipeline_factual_isolation.py",
    ],
    "web": [
        "tests/test_search_layer_policy.py",
        "tests/test_response_pipeline_fx_format.py",
        "tests/test_web_currency_post_search.py",
        "tests/test_web_factual_numeric_safety.py",
        "tests/test_web_fx_routing.py",
        "tests/test_web_geo_hint_policy.py",
        "tests/test_web_intent_alignment.py",
        "tests/test_web_policy_challenge.py",
        "tests/test_web_post_search_numeric_pipeline.py",
        "tests/test_web_query_core_extraction.py",
        "tests/test_web_source_filtering.py",
    ],
    "task_models": [
        "tests/test_task_model_migration.py",
        "tests/test_task_model_registry.py",
        "tests/test_task_model_router.py",
    ],
    "all": ["tests"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predefined pytest suites for MMis.",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="memory",
        help="Predefined suite to run. Default: memory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the files for the selected suite and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command without running it.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra pytest args. Example: --suite memory -- -q -k recall",
    )
    return parser.parse_args()


def _normalized_pytest_args(values: list[str]) -> list[str]:
    args = list(values or [])
    if args[:1] == ["--"]:
        return args[1:]
    return args


def main() -> int:
    args = _parse_args()
    targets = list(SUITES.get(str(args.suite or "memory"), []))
    pytest_args = _normalized_pytest_args(list(args.pytest_args or []))

    if args.list:
        print(f"Suite: {args.suite}")
        for item in targets:
            print(item)
        return 0

    cmd = [sys.executable, "-m", "pytest", *targets, *pytest_args]
    printable = subprocess.list2cmdline(cmd)

    if args.dry_run:
        print(printable)
        return 0

    print(f"Running suite: {args.suite}")
    print(printable)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
