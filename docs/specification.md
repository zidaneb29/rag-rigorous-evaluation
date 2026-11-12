# Technical specification

## Corpus decision

Use QASPER for the primary study. The public release contains 5,049 questions over 1,585 NLP papers. Independent annotators supplied free-text, extractive, yes/no, or unanswerable responses and supporting evidence where available. Questions were written with the target title and abstract in view. The dataset card lists CC BY 4.0. Preserve attribution and verify underlying paper rights and source URLs before redistributing any text or derived artifacts containing excerpts. Use the benchmark's structured text locally to avoid PDF extraction variability. [QASPER paper](https://aclanthology.org/2021.naacl-main.365/) · [Dataset card](https://huggingface.co/datasets/allenai/qasper/blob/9834d7e5a315da3a8c5df41024929af4462b80ce/README.md)

Use the original v0.3 train/dev and test archives. Ai2's loader metadata lists 888 train, 281 dev, and 416 test papers, with about 14.7 MB of compressed downloads. Verify the listed checksums and confirm actual counts during ingestion. [Ai2 release metadata](https://huggingface.co/datasets/allenai/qasper/discussions/4/files)

| Candidate | Evidence and fit | Decision |
| --- | --- | --- |
| QASPER | Full technical papers, human QA, evidence, and native unanswerable annotations. Evidence is anchored to one paper and can include non-text material. | Core benchmark. Use text evidence for retrieval metrics and report exclusions. |
| SciFact | Scientific abstracts with sentence rationales for support or refutation and claims without evidence. Small and suitable for claim verification. | Possible separate citation-verification transfer study. Its claim labels do not directly evaluate free-form technical QA. [Paper](https://aclanthology.org/2020.emnlp-main.609/) · [Release](https://github.com/allenai/scifact) |
| HotpotQA | Public Wikipedia-based benchmark with sentence evidence and multi-document questions. Distractor setting limits candidate scope. | Possible separate multi-document study. Domain and answerability design differ from the main study. [Official benchmark](https://hotpotqa.github.io/) |
| Curated GNN or optimization papers | Strong topical fit. Multi-document questions, evidence sets, distractors, and negative labels require independent annotation and document-level rights review. | Defer until the core study is complete. Report any future labels as project-created. |

## Pipeline

1. **Ingest.** Record the release ID, source URL and file hashes. Store title, source identifier or verified link, section, paragraph ordinal, and raw text. Give paragraphs stable IDs independent of chunking. Keep question text, labels, and reference evidence in a separate evaluation store. Index body passages and abstracts with title and section metadata. Keep source text and indexed normalization traceable through offsets.
2. **Chunk.** Keep chunks within one source paragraph to simplify evidence alignment. Use the BGE tokenizer and a 256-token ceiling including title and section prefixes and special tokens. Split long paragraphs into windows with 32 tokens of overlap by default. Keep raw character offsets. Index the abstract as its own paragraph and body paragraphs in source order. Preserve table and figure records in the source data and omit standalone captions and visual contents from the index. Retain text and placeholders already embedded in body paragraphs.
3. **Retrieve.** A uses BM25Okapi with a deterministic Unicode tokenizer that preserves technical terms and numbers. B uses normalized BGE-small vectors and exact cosine search via float32 NumPy matrix-vector multiplication. C fuses the top 50 from each using equal-weight RRF with constant 60 and one-based ranks. Absent items contribute zero. D rescores C's top 20 with the optional MiniLM cross-encoder and leaves ranks beyond that pool in their original order. Resolve baseline and fusion ties by ascending chunk ID and cross-encoder ties by original fusion rank. Apply any paper scope before ranking, with BM25 corpus statistics fixed for a given index. [BM25 implementation](https://github.com/dorianbrown/rank_bm25) · [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
4. **Evaluate and persist.** The Python CLI ingests, indexes, evaluates and aggregates retrieval results. Versioned TOML settings and hashed protocol and population manifests identify each run. SQLite transactions preserve query-level results for resume and replay. Save stage timings, ranks, candidate scores, chunk offsets, and label mappings.

Generation remains an archived feasibility investigation. The [1.5B pilot](phase4.md), [prompt recovery](phase4b.md), and [7B Colab sanity check](phase4c.md) record the experimental context budgets, backend, citation validation, and outcomes.

## Streamlit

**Evidence search** offers a question field, required development-paper selection, retrieval method, and top-k selection. It shows ranked passage text with paper and section provenance and retrieval timings. It caches verified index resources and loads retrieval models lazily.

**Evaluation dashboard** reads completed CLI artifacts and shows split and scope, sample counts, configuration, paired metric differences and intervals, latency distributions, and sampled successes and failures with evidence IDs. Empty states explain how to restore results. See the [Phase 6 guide](phase6.md) for setup and verification. [Streamlit documentation](https://docs.streamlit.io/)
