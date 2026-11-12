# Phase 2 development pilot

The Windows CPU pilot ran on November 11, 2026 on the i5-6300HQ with four logical cores, 16 GiB installed RAM, Python 3.12.14, and ONNX Runtime's CPU provider. Seed 42 sampled 20 paper IDs from the sorted 281-paper QASPER v0.3 development split. All 66 questions on those papers were queried in each condition. Retrieval relevance metrics used the 50 questions with a majority answerable label and at least one completely mapped text evidence set. The other 16 comprised eight without a complete text set, five answerability ties, and three majority-unanswerable questions. The held-out split was never loaded.

## Measured retrieval results

These are single development-subset observations. The split-wide condition searches the 20 selected papers and includes the target title in the query. All 66 questions retrieved their target paper in the first ten chunks under both split-wide methods. This does not identify the title's effect, which requires a paired query-only ablation.

| Method | Scope | Recall@5 | Recall@10 | MRR@20 | nDCG@5 | Complete annotated set@10 | Warm query p50 / p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | Paper | .287 | .482 | .256 | .201 | .520 | 0.33 / 0.87 |
| BGE | Paper | .449 | .570 | .415 | .357 | .580 | 26.22 / 64.92 |
| BM25 | Split with title | .135 | .287 | .162 | .100 | .280 | 9.57 / 19.96 |
| BGE | Split with title | .172 | .307 | .176 | .123 | .340 | 20.84 / 39.20 |

Quality denominators are 50 and latency denominators are 66 per row. Search timings are from the initial run after one warmup per method, include query encoding for BGE, and exclude scoring and file I/O. BM25 uses split-index document frequencies even under paper scope. BGE uses a retrieval prefix for queries, CLS pooling and float32 L2 normalization for both queries and passages, and an exact dot product on normalized vectors. Both methods break ties by ascending chunk ID.

## Measured resource use

| Measurement | Initial build | Cache reuse |
| --- | ---: | ---: |
| Archive verification, dev JSON load and schema validation | 0.57 s | 0.61 s |
| Chunking and paper lookup | 1.35 s | 1.20 s |
| BM25 build | 0.08 s | 0.10 s |
| ONNX session load | 2.17 s | 0.89 s |
| Passage embeddings or validated cache load | 116.29 s | 0.016 s |
| Evaluation, all four conditions | 4.52 s | 3.81 s |
| Sum of timed stages | 125.61 s | 7.28 s |
| Process CPU time, sum of timed stages | 215.50 CPU-s | 8.02 CPU-s |
| Process peak Windows working set | 364 MiB | 260 MiB |
| Available system memory at start / end | 2.75 / 2.35 GiB | 3.00 / 2.74 GiB |

The run indexed **1,214 chunks** at **10.44 chunks/s** during the initial passage embedding and cache-write stage. The dense matrix is 1,864,832 bytes (1.78 MiB), and the text-free chunk manifest is 182,527 bytes. BM25 is rebuilt in memory, so it has no separate persisted index size. The first run's evidence and ranking files are 78,752 and 996,337 bytes. Both pilot runs completed without a memory or inference error. System-wide memory was sampled at the start and end, so transient system memory and paging were not measured. The run makes no full-corpus runtime estimate.

Downloads were the official QASPER train/dev archive (10,835,856 bytes), BGE ONNX graph (133,093,490 bytes), BGE tokenizer (711,396 bytes), the model repository metadata response (149,130 bytes), and free PyPI wheels for the requirements and their dependencies (about 215 MB installed). There were no paid services, generator weights, or QASPER test downloads.

## Evidence mapping and reproducibility

Original source dictionaries retain raw answer fields, annotation IDs, evidence strings, and the document structure. Matching uses Unicode NFC and collapsed whitespace, within the target paper only. Every original evidence label receives a status. Matching repeated paragraphs is ambiguous, and unresolved or non-text labels cannot form a complete mapped set. This pilot yielded 206 unique text matches, 41 unmatched labels, and 15 explicit non-text labels. A diagnostic found two unmatched strings as substrings of paragraphs. No heuristic remapping was applied.

The audit checked all 1,214 chunk offsets against 1,141 source paragraphs, including 60 paragraphs spanning multiple windows. Each indexed paragraph's windows covered its full original span, and all 206 mapped labels matched the referenced source text after normalization. The cache passed its content hash and vector norm checks. Cold and warm runs produced byte-identical chunk and evidence manifests and identical ranks, scores, and metrics across all 264 condition-query rows after removing measured latency. This verifies replay on one machine. Independent embedding rebuilds and cross-machine bitwise reproducibility were outside this check.

All **12 unit tests passed**. They cover archive integrity and schema, annotation preservation, deterministic chunk windows and offsets, evidence ambiguity, partial and overlapping windows, metric formulas, paper filtering, score ties, exact similarity, and cache invalidation or corruption. The first test attempt exposed Windows temporary-directory access and a retained memory-map handle in the cache test. The successful run used a project-local temporary directory and released the test handle before cleanup.

## Reproduce the pilot

Use Python 3.12 on Windows. Run these commands from the project root. The CLI verifies the archive and model hashes and can run offline after the setup files are available.

```powershell
python -m venv .venv
$pilotPython = '.\.venv\Scripts\python.exe'
& $pilotPython -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, model, artifacts/test-temp | Out-Null
$qasperBase = 'https://qasper-dataset.s3.us-west-2.amazonaws.com'
$bgeBase = 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
curl.exe --fail --location "$qasperBase/qasper-train-dev-v0.3.tgz" -o data/qasper-train-dev-v0.3.tgz
curl.exe --fail --location "$bgeBase/onnx/model.onnx" -o model/model.onnx
curl.exe --fail --location "$bgeBase/tokenizer.json" -o model/tokenizer.json
$env:TEMP = (Resolve-Path artifacts/test-temp).Path
$env:TMP = $env:TEMP
$env:OPENBLAS_NUM_THREADS = '2'
$env:TOKENIZERS_PARALLELISM = 'false'
& $pilotPython -m unittest discover -s tests -v
& $pilotPython -m rag_eval --cache artifacts/replay-cache --output artifacts/replay-build
& $pilotPython -m rag_eval --cache artifacts/replay-cache --output artifacts/replay-reuse
```

Use a fresh cache directory to measure a cold embedding build. The report records input and output hashes, timing, runtime versions, eligibility counts, and the selected paper IDs. Each run saves IDs, ranks and scores for every evaluated question, along with a chunk-offset manifest and annotation mapping. Keep the input assets, caches, and generated artifacts local. The actual pilot used the bundled Python 3.12.14 interpreter and workspace-local dependencies because the Windows Store Python alias was unavailable.

## Phase 3 review

Audit the unmatched evidence labels before freezing any mapping changes, including whether apparent matches represent whole paragraphs, fragments, or absent text. Run paired RRF and reranking comparisons on development data with the same chunk index and report latency by stage. Include the title-cue ablation on an identical candidate set and distinguish retrieval of an annotated set from human judgment of sufficiency. Audit cross-split duplicates before any held-out evaluation. Keep Qwen2.5-1.5B provisional until a later phase measures its memory use, throughput, and technical answer quality.
