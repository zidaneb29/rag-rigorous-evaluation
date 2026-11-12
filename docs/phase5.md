# Phase 5 held-out retrieval evaluation

## Protocol and provenance

The protocol and development diagnostics were frozen before downloading or opening the QASPER v0.3 test archive. The canonical protocol digest is `b8e8a70faf83fcc3a42f85658000944b6a379f039fb41689dc8c478e1f59b9a0`. The manifest records dependency and implementation versions, QASPER archive hashes, model revisions and asset hashes, and the complete Phase 2 and 3 settings. Phase 5 reused the existing BGE and MiniLM assets. The official test archive was the only new data download. Its SHA-256 is `72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72`.

The four configurations use BM25Okapi, normalized BGE-small exact NumPy cosine search, equal-weight RRF over the top 50 of each baseline with constant 60, and MiniLM reranking of the first 20 fused candidates. The reranker preserves the fused tail. Chunking stays within source paragraphs, uses a 256-token cap including metadata and special tokens, and uses 32-token overlap on long paragraphs. Tie rules and the CPU backends match Phase 3. All methods use the same corpus and questions within each scope. Paper-scoped retrieval searches the supplied target paper. Split-wide retrieval searches the sampled 40-paper pool and appends the original question after the target title. The scopes are reported separately.

The resource preflight verified 888 train, 281 development and 416 test papers. It compared cross-split IDs, normalized titles, exact normalized body fingerprints, near-title token Jaccard of at least 0.85 with five or more tokens, and at least 80% overlap relative to the smaller set of long paragraph fingerprints with at least three per paper. It found zero matches at the frozen thresholds. This audit cannot detect all paraphrased versions, smaller shared excerpts, or model pretraining exposure.

The preflight used a five-paper document-only chunk-count sample (247 chunks), measured Phase 3 resource rates, and question counts to project workload. The projection was **88.5 minutes** for all 416 papers and 1,451 questions across both scopes, above the predefined 20-minute budget. The frozen fallback shuffled sorted eligible paper IDs with seed 42, selected 40 papers, and kept all 156 questions from those papers. Its projection was **9.2 minutes**. These are estimates and do not enter retrieval outcomes. Question or annotation outcomes were not used to select the sample. The population digest is `3102c170ba3d5036c9d5c5030ab7784980f32454bffd04ff7cb4d7bef43e3ce0`.

### Scoring

Use the audited Phase 3 mapping. Exact unique section headings map to all nonempty indexed paragraphs in that section. An annotation contributes an evidence set only if every required label maps to indexed text. Eligibility requires a majority answerable label and at least one complete mapped set. Ties, majority-unanswerables and questions without a complete text set remain in coverage and latency denominators. Report the union of eligible sets for Recall, MRR and nDCG, and completion of any eligible set as a separate metric. A paragraph counts only when retrieved offsets cover its full span. At each chunk rank, nDCG grants one binary gain for a newly completed annotated paragraph. It uses the ideal with one completed paragraph at each early rank, which can be unattainable for a multi-window paragraph. Unannotated passages have no observed gain and may still support an answer.

Recall@1/5/10, MRR@20 and nDCG@5 are question averages. Paired intervals use 2,000 paper-cluster bootstrap draws with seed 42 and percentile endpoints. Each sampled paper contributes all its paired eligible questions, and the mean weights questions within the resample. The intervals are descriptive and unadjusted across contrasts.

## Development robustness checks

The frozen checks reused Phase 3 runs. They did not fit any new parameters.

- **Title cue.** With the same 30 development papers, the title cue increased target-paper Hit@10 for all four methods. Its Recall@10 effects were +.077 for BM25, +.070 for BGE, +.026 for RRF, and +.101 for RRF + MiniLM. Only the last interval excluded zero in this sample [+0.010, +0.190]. RRF nDCG@5 fell by .074 even as target-paper identification improved.
- **Section labels.** Among 77 paper-scoped eligible development questions, four had at least one mapped heading label. BGE Recall@10 was .416 on that panel and .616 on the other 73. The heading panel had just two questions under the paragraph-only rule. The two protocols cover different populations there, and the four-case panel is too small to isolate a heading effect.

## Held-out results

### Paper scoped

126 eligible questions across 40 papers. Latency covers all 156 questions across 40 papers.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@20 | nDCG@5 | Annotated set@10 | p50 / p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.134 | 0.394 | 0.557 | 0.388 | 0.317 | 0.563 | 0.48 / 0.69 |
| BGE | 0.189 | 0.492 | 0.614 | 0.511 | 0.442 | 0.643 | 11.25 / 15.13 |
| RRF | 0.159 | 0.477 | 0.593 | 0.467 | 0.401 | 0.603 | 11.87 / 15.87 |
| RRF + MiniLM | 0.222 | 0.536 | 0.695 | 0.543 | 0.470 | 0.706 | 900.81 / 1190.27 |

| Paired contrast | Δ Recall@10 [95% CI] | Δ MRR@20 [95% CI] | Δ nDCG@5 [95% CI] |
| --- | --- | --- | --- |
| BGE − BM25 | +0.057 [-0.017, +0.128] | +0.124 [+0.053, +0.190] | +0.125 [+0.072, +0.178] |
| RRF − BM25 | +0.035 [-0.036, +0.111] | +0.079 [+0.043, +0.119] | +0.085 [+0.045, +0.127] |
| RRF − BGE | -0.022 [-0.063, +0.022] | -0.044 [-0.102, +0.016] | -0.040 [-0.082, +0.002] |
| RRF + MiniLM − BM25 | +0.138 [+0.074, +0.202] | +0.155 [+0.106, +0.199] | +0.153 [+0.116, +0.194] |
| RRF + MiniLM − BGE | +0.081 [+0.033, +0.129] | +0.032 [-0.042, +0.100] | +0.028 [-0.031, +0.081] |
| RRF + MiniLM − RRF | +0.103 [+0.052, +0.155] | +0.076 [+0.021, +0.127] | +0.069 [+0.022, +0.114] |

### Sampled split, title cued

126 eligible questions across 40 papers. Latency covers all 156 questions across 40 papers.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@20 | nDCG@5 | Annotated set@10 | p50 / p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.043 | 0.251 | 0.400 | 0.237 | 0.189 | 0.381 | 12.01 / 17.30 |
| BGE | 0.093 | 0.273 | 0.439 | 0.325 | 0.236 | 0.429 | 17.85 / 22.05 |
| RRF | 0.057 | 0.264 | 0.400 | 0.286 | 0.211 | 0.357 | 30.12 / 39.01 |
| RRF + MiniLM | 0.200 | 0.465 | 0.593 | 0.501 | 0.422 | 0.595 | 792.99 / 1143.54 |

| Paired contrast | Δ Recall@10 [95% CI] | Δ MRR@20 [95% CI] | Δ nDCG@5 [95% CI] |
| --- | --- | --- | --- |
| BGE − BM25 | +0.039 [-0.047, +0.116] | +0.088 [+0.029, +0.147] | +0.047 [-0.008, +0.105] |
| RRF − BM25 | +0.000 [-0.042, +0.045] | +0.050 [+0.009, +0.087] | +0.022 [-0.016, +0.059] |
| RRF − BGE | -0.038 [-0.108, +0.031] | -0.038 [-0.078, +0.003] | -0.025 [-0.063, +0.016] |
| RRF + MiniLM − BM25 | +0.193 [+0.121, +0.265] | +0.264 [+0.196, +0.333] | +0.232 [+0.175, +0.289] |
| RRF + MiniLM − BGE | +0.154 [+0.097, +0.217] | +0.177 [+0.111, +0.241] | +0.186 [+0.111, +0.259] |
| RRF + MiniLM − RRF | +0.192 [+0.123, +0.263] | +0.215 [+0.149, +0.275] | +0.210 [+0.149, +0.269] |

## Evidence coverage and failure analysis

Of the 156 questions, 126 (80.8%) were eligible. Exclusions were 12 without a complete mapped text set, nine answerability ties, and nine majority-unanswerable questions. Across the 156, the audit counted 610 mapped paragraph labels, 18 mapped section labels, 57 non-text labels, and two ambiguous labels. Paragraph-only eligibility was 125 questions. On that population, paper-scoped Recall@10 was .563, .625, .596, and .713 for BM25, BGE, RRF, and reranking respectively. The method order matches the audited paper-scoped result, and the populations differ.

The analysis shuffles eligible paper-scoped rows with seed 42 and records the first matching question per predefined pattern. Patterns overlap. Counts cover all 126 eligible rows.

| Pattern | Count | Sample and rank observation |
| --- | ---: | --- |
| No annotated paragraph completed anywhere in the fused pool | 0 | Every eligible query completed at least one annotated paragraph somewhere in the pool. Complete sets can still be missed, and evidence can rank below the useful cutoff. |
| First fused evidence after rank 5 and by rank 20 | 32 | Test-set domain, `338a3758…`. Fusion first found a gold paragraph at rank 6. MiniLM moved another gold paragraph to rank 2. |
| BGE beats BM25 on Recall@10 | 34 | Same domain question. BGE covered two of three gold paragraphs by rank 10. BM25 covered one. |
| BM25 beats BGE on Recall@10 | 22 | Prior approaches, `eccbbe36…`. BM25 placed the annotated paragraph 10th and BGE 30th. |
| RRF beats both baselines on Recall@10 | 3 | Classification features, `35d2eae3…`. The baselines found different paragraphs by rank 10. RRF placed both at ranks 5 and 6. |
| RRF falls below the better baseline on Recall@10 | 36 | Prior approaches, `eccbbe36…`. RRF moved the BM25 hit from rank 10 to rank 16. MiniLM recovered it at rank 7. |
| MiniLM improves nDCG@5 | 50 | Test-set domain, `338a3758…`. The rank-2 hit raised nDCG@5 from 0 to .296. |
| MiniLM lowers nDCG@5 | 22 | Human judgments of simplification, `511e46b5…`. The annotated paragraph moved from rank 1 to rank 3. |

For the incomplete case `4dc41801…`, all three annotation sets contain non-text labels, including sets with some mapped text. This question remains visible in the audit and latency counts. It cannot be scored using the complete text-set protocol. None of these rank checks establishes that an annotated or unannotated passage is sufficient on its own.

## Measured efficiency

The Windows i5-6300HQ CPU run used two model threads, embedding batches of eight and reranking batches of eight. Nonoverlapping measured stages took **446.2 seconds** of wall time (7.44 minutes), excluding report serialization and process startup. The model weights were loaded sequentially. No services were billed.

| Measurement | Observed |
| --- | ---: |
| Resource preflight and duplicate audit | 5.77 s |
| Chunking and annotation mapping | 1.77 s |
| BGE session load | 0.97 s |
| Embedding, shard writes and vector-index assembly | 165.85 s for 1,925 chunks, 11.61 chunks/s |
| BM25 index construction | 0.13 s |
| Base query stage | 8.79 s for both scopes |
| MiniLM session load | 0.30 s |
| Fusion and rerank stage, including MiniLM load | 262.93 s |
| Peak process working set | 412.3 MiB |
| Vector shard storage, including checksum files | 2,964,560 bytes |

The p50/p95 table uses warm query timing. Dense time includes query embedding and exact search. Fusion sums sparse, dense, and fusion time. The reranked method adds MiniLM scoring and reorder time. Model loads, warmups, index construction and artifact writes are outside query timings. Reranking alone took 889/1,174 ms p50/p95 in paper scope and 763/1,109 ms in the title-cued sampled pool. This describes the measured run on one machine. System paging was not instrumented.

## Conclusions and limitations

On the paper-scoped sample, BGE improved early ranking over BM25 with a median of 11.25 ms. Its Recall@10 contrast with BM25 remained uncertain. RRF alone had lower point estimates than BGE on the reported metrics, with paired intervals spanning zero for the three main contrasts. MiniLM improved Recall@10 and nDCG@5 over RRF in both scopes. Compared with BGE, its paper-scoped Recall@10 gain was +.081 [+.033, +.129], while its nDCG@5 interval included zero. The median cost rose from about 11 ms to 901 ms. BGE is a reasonable responsive default. Expose reranking as an explicit quality–latency option.

The 40-paper sample and the title-cued candidate pool constrain generalization. The title cue, paper-scope restriction, partial annotation coverage, and section-to-paragraph mapping all shape the measured quantities. The duplicate screen and published benchmarks cannot rule out training exposure. The study evaluates retrieval and provides no validated answer-generation results.

## Reproduction

Use the [Windows CPU setup](phase2.md) and [MiniLM assets](phase3.md). On a fresh checkout, prepare the development artifacts first. The resource freeze reads the cold Phase 3 run at `artifacts/phase3-dev`. The command below uses it for the development diagnostics as well. If the original verified runs are already available, use them in place of new measurements. Keep all archives and full experiment artifacts local.

```powershell
# On a fresh checkout, once the train/dev archive and all model assets exist:
& $pilotPython -m rag_eval --output artifacts/pilot-cold
& $pilotPython -m rag_eval.phase3 --output artifacts/phase3-dev
& $pilotPython -m rag_eval.phase5 freeze --dev artifacts/phase3-dev --output artifacts/phase5-reproduction
curl.exe --fail --location https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz -o data/qasper-test-and-evaluator-v0.3.tgz
& $pilotPython -m rag_eval.phase5 run --output artifacts/phase5-reproduction
& $pilotPython -m pip install -r requirements-figures.txt
& $pilotPython -m rag_eval.phase5_figures artifacts/phase5-reproduction/report.json artifacts/phase5-reproduction/figures
```

Reuse the output directory to resume. The runner validates the frozen manifest and content hashes, checksums each per-paper embedding shard, and records base and reranked query rows transactionally. An uninterrupted second invocation replays saved work and writes a separate replay report. Keep the protocol and population manifests with the result artifacts. A new freeze derives its workload estimate from local development measurements. Check the resulting population manifest before comparing a fresh run with the recorded 40-paper sample.

All **43 tests passed**, including the 39 earlier tests. The replay recovered 312 base and 312 rerank checkpoints without loading either model. Its timed stages took 8.05 seconds, including a fresh duplicate audit and chunk/evidence reconstruction. Rankings, candidate records, metrics, paired intervals, failure samples and artifact hashes matched. A separate audit verified all 1,925 vector rows and checksums, all 312 RRF calculations and candidate orders, paper-scope filtering, and rerank preservation of the fused tail and Recall@20. This establishes checkpoint replay and rank invariants on the recorded run. It does not test an independent rerun of model inference.

The two SVG figures were byte-identical when regenerated. Previous Phase 2, 3, 4 and 4B recorded hashes matched, and the original Phase 4C checkpoint and packet were preserved.

## Dashboard handoff

Use the saved report for point estimates, percentile intervals, eligibility counts and timing. Label the two scopes explicitly and show the 40-paper candidate-pool limit next to any split-wide result. Provide a quality–latency view with median and p95 kept distinct. Use saved ranking and evidence IDs for examples. Resolve raw passages locally from the chunk manifest and source offsets. Mark annotation matches as benchmark labels and describe other passages as unlabeled. The generator feasibility results can appear in a separate read-only view.
