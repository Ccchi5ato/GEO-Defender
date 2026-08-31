from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


METHOD_NAMES = {
    "agentic_geo": "AgenticGEO",
    "autogeo": "AutoGEO",
    "mageo": "MAGEO",
    "geo_adaptive": "GEO",
    "geo_sfe_isg_strict": "GEO-SFE",
    "if_geo": "IF-GEO",
    "raid_gseo": "RAID-GSEO",
}
SEEN_METHODS = {"AgenticGEO", "AutoGEO", "MAGEO"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def nonempty_fields(doc: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "document_id": int(doc["doc_id"]),
        "document_type": "attack" if doc.get("doc_type") == "attack" else "benign",
        "source": doc.get("source", ""),
        "text": doc["text"],
    }
    for key in ("title", "url", "domain", "provider", "score", "search_query"):
        value = doc.get(key)
        if value not in (None, ""):
            result[key] = value
    return result


def restored_clean_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    originals = row["original_text_list"]
    target_id = int(row["target_id"])
    documents: list[dict[str, Any]] = []
    for raw_doc in sorted(row["docs"], key=lambda item: int(item["doc_id"])):
        doc = nonempty_fields(raw_doc)
        if int(doc["document_id"]) == target_id:
            if target_id >= len(originals):
                raise ValueError(f"Invalid target document index in query {row['query_id']}")
            doc["text"] = originals[target_id]
        doc["document_type"] = "benign"
        documents.append(doc)
    return documents


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_attack_instance(row: dict[str, Any], split: str) -> dict[str, Any]:
    method = METHOD_NAMES[row["dataset_name"]]
    documents = [nonempty_fields(doc) for doc in sorted(row["docs"], key=lambda item: int(item["doc_id"]))]
    attack_ids = [doc["document_id"] for doc in documents if doc["document_type"] == "attack"]
    if attack_ids != [int(row["attack_doc_id"])]:
        raise ValueError(f"Expected one aligned attack document in query {row['query_id']} ({method})")
    return {
        "instance_id": f"{split}-{row['query_id']}-{row['dataset_name']}",
        "split": split,
        "query_id": str(row["query_id"]),
        "query": row["query"],
        "geo_method": method,
        "method_seen_during_construction": method in SEEN_METHODS,
        "attack_document_id": attack_ids[0],
        "preference_benign_document_id": int(row["benign_id"]),
        "documents": documents,
    }


def normalize_tfsg_top5(row: dict[str, Any]) -> dict[str, Any]:
    method = METHOD_NAMES.get(row.get("attack_method", ""), row.get("attack_method", ""))
    docs = []
    for raw_doc in row["top5_docs"]:
        doc = {
            "doc_id": int(raw_doc["doc_id"]),
            "doc_type": "attack" if raw_doc.get("doc_type") == "attack" else "benign",
            "source": raw_doc.get("source", ""),
            "text": raw_doc["text"],
        }
        for key in ("title", "url"):
            value = raw_doc.get(key)
            if value not in (None, ""):
                doc[key] = value
        docs.append(doc)
    return {
        "sample_id": str(row["sample_id"]),
        "query_id": str(row.get("query_id", "")),
        "query": row["query"],
        "attack_method": method,
        "attack_doc_ids": [int(value) for value in row.get("attack_doc_ids", [])],
        "top5_docs": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the public GEO-DefenseBench release.")
    parser.add_argument("--construction-top10", required=True, type=Path)
    parser.add_argument("--test-top10-dir", required=True, type=Path)
    parser.add_argument("--tfsg-construction-top5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    construction_rows = read_jsonl(args.construction_top10)
    test_rows: list[dict[str, Any]] = []
    for path in sorted(args.test_top10_dir.glob("*.jsonl")):
        test_rows.extend(read_jsonl(path))

    if len(construction_rows) != 36 or len(test_rows) != 616:
        raise ValueError(f"Unexpected split sizes: construction={len(construction_rows)}, test={len(test_rows)}")

    construction_queries = {str(row["query_id"]) for row in construction_rows}
    test_queries = {str(row["query_id"]) for row in test_rows}
    if construction_queries & test_queries:
        raise ValueError("Construction and test query groups overlap")

    clean_by_split: dict[str, dict[str, dict[str, Any]]] = {"construction": {}, "test": {}}
    clean_digests: dict[str, dict[str, str]] = {"construction": {}, "test": {}}
    for split, rows in (("construction", construction_rows), ("test", test_rows)):
        for row in rows:
            query_id = str(row["query_id"])
            clean = {
                "query_id": query_id,
                "split": split,
                "query": row["query"],
                "documents": restored_clean_documents(row),
            }
            digest = canonical_digest(clean)
            previous = clean_digests[split].get(query_id)
            if previous is not None and previous != digest:
                raise ValueError(f"Clean candidates differ across GEO variants for query {query_id}")
            clean_digests[split][query_id] = digest
            clean_by_split[split][query_id] = clean

    construction_doc_hashes = {
        canonical_digest(doc["text"])
        for row in clean_by_split["construction"].values()
        for doc in row["documents"]
    }
    test_doc_hashes = {
        canonical_digest(doc["text"])
        for row in clean_by_split["test"].values()
        for doc in row["documents"]
    }
    if construction_doc_hashes & test_doc_hashes:
        raise ValueError("Construction and test splits share at least one exact clean document")

    construction_instances = [normalize_attack_instance(row, "construction") for row in construction_rows]
    test_instances = [normalize_attack_instance(row, "test") for row in test_rows]

    preferences = []
    for index, row in enumerate(construction_rows):
        target_id = int(row["target_id"])
        benign_id = int(row["benign_id"])
        docs_by_id = {int(doc["doc_id"]): doc for doc in row["docs"]}
        preferences.append(
            {
                "source_row_index": index,
                "query": row["query"],
                "qwen3_instruct_1": row["qwen3_instruct_1"],
                "pos1_text": row["original_text_list"][target_id],
                "neg_text": docs_by_id[target_id]["text"],
                "pos2_text": docs_by_id[benign_id]["text"],
            }
        )

    tfsg_rows = [normalize_tfsg_top5(row) for row in read_jsonl(args.tfsg_construction_top5)]
    if len(tfsg_rows) != 36:
        raise ValueError(f"Expected 36 TFSG construction rows, found {len(tfsg_rows)}")

    out = args.output_dir
    write_jsonl(out / "clean" / "construction.jsonl", sorted(clean_by_split["construction"].values(), key=lambda r: r["query_id"]))
    write_jsonl(out / "clean" / "test.jsonl", sorted(clean_by_split["test"].values(), key=lambda r: r["query_id"]))
    write_jsonl(out / "attack_injected" / "construction.jsonl", construction_instances)
    write_jsonl(out / "attack_injected" / "test.jsonl", test_instances)
    write_jsonl(out / "construction" / "shield_reranker_preferences.jsonl", preferences)
    write_jsonl(out / "construction" / "tfsg_top5.jsonl", tfsg_rows)

    method_counts = Counter(row["geo_method"] for row in construction_instances + test_instances)
    stats = {
        "clean_queries": {
            "construction": len(clean_by_split["construction"]),
            "test": len(clean_by_split["test"]),
        },
        "attack_injected_instances": {
            "construction": len(construction_instances),
            "test": len(test_instances),
        },
        "geo_method_counts": dict(sorted(method_counts.items())),
        "construction_query_overlap_with_test": 0,
        "exact_clean_document_overlap_with_test": 0,
        "sha256": {
            "clean_construction": canonical_digest(list(clean_by_split["construction"].values())),
            "clean_test": canonical_digest(list(clean_by_split["test"].values())),
            "attack_construction": canonical_digest(construction_instances),
            "attack_test": canonical_digest(test_instances),
        },
    }
    (out / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
