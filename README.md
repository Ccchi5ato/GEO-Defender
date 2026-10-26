# GEO Defender

Reference implementation and resources for **GEO Defender**, a two-stage defense for generative search against malicious Generative Engine Optimization (GEO).

- **Shield Reranker** is a LoRA-adapted reranker that demotes GEO attack documents using preference-based defensive correction.
- **Training-Free Shield Generation (TFSG)** uses an external experience library to guide source use by a fixed target LLM.

## Repository layout

```text
geo_defender/     Core two-stage implementation
scripts/          Training, library construction, and inference entry points
artifacts/        Released LoRA adapter and six experience libraries
data/             GEO-DefenseBench
examples/         Minimal input examples
```

## Dataset

GEO-DefenseBench provides 36 construction instances and 616 test instances. Each attack-injected instance contains one query, nine benign candidate web documents, and one GEO attack document. The construction and test partitions contain disjoint query groups and exact clean-document texts. See [`data/geo_defensebench/README.md`](data/geo_defensebench/README.md) for the schema and split statistics.

## Installation

Python 3.10+ and a CUDA-capable environment are recommended.

```bash
python -m pip install -e .
```

The base reranker, `Qwen/Qwen3-Reranker-0.6B`, is downloaded on first use. The released adapter does not include base-model weights.

## Run GEO Defender

Set the target-LLM credential through an environment variable, then run:

```bash
python scripts/run_geo_defender.py \
  --input examples/inference.jsonl \
  --output outputs/defended_answers.jsonl \
  --adapter artifacts/shield_reranker_adapter \
  --experience-library artifacts/experience_libraries/gpt-5.5.json \
  --generator-model gpt-5.5
```

The runner reranks candidates, retains top-$k$ evidence, and produces an experience-guided answer in one target-LLM call. Use the corresponding JSON file under `artifacts/experience_libraries/` for another target LLM.

## Main entry points

- `scripts/train_shield_reranker.py`: train Shield Reranker from preference triplets.
- `scripts/precompute_reference_scores.py`: prepare frozen-reranker reference scores.
- `scripts/build_tfsg_library.py`: construct a library from structured diagnostics.
- `scripts/build_tfsg_library_reasoning.py`: construct a library from exposed reasoning content.
- `scripts/run_geo_defender.py`: run the full inference pipeline.

The released adapter and final libraries can be used directly; API credentials must never be committed to the repository.

## License

The code is released under the [MIT License](LICENSE). The dataset contains research records derived from public web pages; users are responsible for complying with the terms of the original sources and any applicable laws.
