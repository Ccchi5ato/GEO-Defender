from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geo_defensebench"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Accept a BOM as some JSONL exports were written by Windows tools.
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def validate_dataset() -> None:
    clean_construction = read_jsonl(DATA / "clean" / "construction.jsonl")
    clean_test = read_jsonl(DATA / "clean" / "test.jsonl")
    attack_construction = read_jsonl(DATA / "attack_injected" / "construction.jsonl")
    attack_test = read_jsonl(DATA / "attack_injected" / "test.jsonl")
    preferences = read_jsonl(DATA / "construction" / "shield_reranker_preferences.jsonl")
    tfsg = read_jsonl(DATA / "construction" / "tfsg_top5.jsonl")

    expected = {
        "clean construction": (clean_construction, 12),
        "clean test": (clean_test, 88),
        "attack construction": (attack_construction, 36),
        "attack test": (attack_test, 616),
        "preference triplets": (preferences, 36),
        "TFSG construction": (tfsg, 36),
    }
    for label, (rows, count) in expected.items():
        if len(rows) != count:
            raise ValueError(f"Expected {count} {label} rows, found {len(rows)}")

    construction_ids = {str(row["query_id"]) for row in clean_construction}
    test_ids = {str(row["query_id"]) for row in clean_test}
    if construction_ids & test_ids:
        raise ValueError("Construction and test query groups overlap")

    methods: set[str] = set()
    for row in attack_construction + attack_test:
        documents = row.get("documents")
        if not isinstance(documents, list) or len(documents) != 10:
            raise ValueError(f"{row.get('instance_id')} does not contain 10 documents")
        attack_docs = [doc for doc in documents if doc.get("document_type") == "attack"]
        if len(attack_docs) != 1:
            raise ValueError(f"{row.get('instance_id')} does not contain exactly one attack document")
        if int(attack_docs[0]["document_id"]) != int(row["attack_document_id"]):
            raise ValueError(f"{row.get('instance_id')} has an inconsistent attack document id")
        methods.add(str(row["geo_method"]))
    if len(methods) != 7:
        raise ValueError(f"Expected seven GEO methods, found {sorted(methods)}")


def validate_artifacts() -> None:
    libraries = sorted((ROOT / "artifacts" / "experience_libraries").glob("*.json"))
    if len(libraries) != 5:
        raise ValueError(f"Expected five experience libraries, found {len(libraries)}")
    for path in libraries:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
            raise ValueError(f"Invalid experience library schema: {path}")

    adapter = ROOT / "artifacts" / "shield_reranker_adapter"
    json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8-sig"))
    model = adapter / "adapter_model.safetensors"
    if not model.is_file() or model.stat().st_size == 0:
        raise ValueError("Missing Shield Reranker adapter weights")


def scan_release_metadata() -> None:
    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
        "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
        "project-local path": re.compile(r"PycharmProjects|Documents\\paper", re.IGNORECASE),
    }
    suffixes = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        # This validator necessarily contains the detection expressions below.
        if path.resolve() == Path(__file__).resolve():
            continue
        if DATA in path.parents:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise ValueError("Potential release metadata found:\n" + "\n".join(findings))


def main() -> None:
    validate_dataset()
    validate_artifacts()
    scan_release_metadata()
    print("Release validation passed.")
    print("Dataset: 12/88 clean queries; 36/616 attack-injected instances; 7 GEO methods.")
    print("Artifacts: 1 Shield Reranker adapter; 5 experience libraries.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Release validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
