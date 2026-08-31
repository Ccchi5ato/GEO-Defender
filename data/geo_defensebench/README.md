# GEO-DefenseBench

GEO-DefenseBench contains paired clean and attack-injected candidate lists for
evaluating defenses against malicious Generative Engine Optimization (GEO).

## Splits

| File | Unit | Count |
| --- | --- | ---: |
| `clean/construction.jsonl` | clean query with 10 benign documents | 12 |
| `clean/test.jsonl` | clean query with 10 benign documents | 88 |
| `attack_injected/construction.jsonl` | query with 9 benign + 1 attack document | 36 |
| `attack_injected/test.jsonl` | query with 9 benign + 1 attack document | 616 |
| `construction/shield_reranker_preferences.jsonl` | Shield Reranker triplet | 36 |
| `construction/tfsg_top5.jsonl` | reranked TFSG construction instance | 36 |

The construction and test partitions are disjoint at the query level. Exact
clean-document text is also disjoint across the two partitions. The checks and
content hashes are recorded in `statistics.json`.

## GEO methods

The construction partition uses AgenticGEO, AutoGEO, and MAGEO. The test
partition contains these three methods together with GEO, GEO-SFE, IF-GEO, and
RAID-GSEO. Each test query has one independently constructed attack-injected
instance for every method.

## Schemas

An attack-injected record contains:

```json
{
  "instance_id": "test-...-if_geo",
  "split": "test",
  "query_id": "...",
  "query": "...",
  "geo_method": "IF-GEO",
  "method_seen_during_construction": false,
  "attack_document_id": 3,
  "preference_benign_document_id": 0,
  "documents": [
    {
      "document_id": 0,
      "document_type": "benign",
      "source": "original_top5",
      "text": "..."
    }
  ]
}
```

The `document_type` field is an evaluation/construction annotation. It must not
be included in the target LLM input during defense execution.

## Data provenance and use

Queries originate from GEO-Bench. Candidate documents were constructed from
the original candidates and additional Tavily search results, after which one
source document was independently rewritten with each GEO method. Web-page
text remains attributable to its original source when a URL is available.
The dataset is released for non-commercial research and evaluation; users are
responsible for respecting the rights and terms associated with source pages.

`scripts/prepare_geo_defensebench.py` documents the normalization and split
validation used to create this public layout from the experiment records.
