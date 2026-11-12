# Phase 3 retrieval experiments

## Design

The experiment retains the 20 Phase 2 development papers and samples 10 from the remaining sorted development IDs with seed 42. It runs every question in those 30 papers, with identical questions and chunk populations across methods within each scope. The expansion increases the paper clusters available for paired analysis while keeping the laptop run bounded. At Phase 2's observed 10.44 chunks/s and 60.7 chunks per paper, 10 additional papers were expected to add about 607 vectors, roughly 58 seconds of embedding and 0.93 MB of vector storage. This estimate assumed similar document lengths and throughput. MiniLM latency had no Phase 2 measurement. No held-out data was loaded.

The primary scope searches the supplied paper using the original question. The secondary experiment searches the same 30-paper index twice, using the original question and the same question prefixed with the target title. Indexed titles and section headings appear in both conditions. Questions may already mention paper-specific terminology or methods, and their authors saw the title and abstract. This comparison measures the incremental effect of an explicit title prefix on this pool.

BM25Okapi and BGE retain their Phase 2 scoring. For each chunk, equal-weight reciprocal rank fusion sums `1 / (60 + rank)` across each input's top 50 results, with ranks starting at one and absent chunks contributing zero. Fusion and baseline score ties use ascending chunk ID. The predeclared sensitivity check uses input depth 25 on paper-scoped questions. MiniLM scores the first 20 fused chunks, breaks tied logits by fusion rank, and preserves the remaining tail. The optional reranker uses the official AVX2 quantized ONNX graph on the CPU with two threads, batches of eight, and a 320-token pair limit. Each result retains the original sparse and dense ranks and scores, fused ordering and scores, and reranker input ranks, scores and logits. [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) · [MiniLM release](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a/onnx)

## Evidence audit

All **206** originally mapped paragraph labels remain mapped. All **41** unmatched labels match unique section headings exactly after the existing normalization. They span 22 distinct strings, including two headings that also occur as fragments in body text. Phase 3 resolves headings by exact section metadata and assigns all nonempty body paragraphs in the section. Sections with a non-indexed body, duplicate headings, and paragraph-heading ambiguity remain unresolved. This conservative mapping can require several paragraphs even if a smaller excerpt suffices. The **15** explicit non-text labels remain non-text. No dataset content or Phase 2 artifact was changed. The audit identifies each label by question ID, original annotation ID and label ordinal, and records its old and new status and source coordinates.

| Eligibility | Original 20-paper pilot | Audited 20-paper pilot | Audited 30-paper expansion |
| --- | ---: | ---: | ---: |
| Eligible | 50 | 52 | 77 |
| No complete text set | 8 | 6 | 8 |
| Answerability tie | 5 | 5 | 7 |
| Majority unanswerable | 3 | 3 | 5 |
| Total | 66 | 66 | 97 |

Two questions became eligible because their heading labels now resolve. Of the remaining six original questions with no complete text set, five had mixed paragraph and non-text labels across answerable annotations and one had only non-text evidence. Every answerable evidence set on those questions lacked at least one required text mapping because it included visual evidence. In the expansion the corresponding counts are six mixed and two non-text-only. Across all 30 papers there are 281 mapped paragraph labels, 41 mapped section labels, and 23 explicit non-text labels. The 20 ineligible questions remain in latency and target-paper hit summaries. Paragraph-only sensitivity uses the original evidence rule on the same rankings and candidate pools.

The ranking conventions remain those in [experiments.md](experiments.md). A gold paragraph is found when returned source spans cover its full text. Metrics report observed annotations, which can omit valid supporting passages. Complete annotated-set coverage records whether all paragraphs from at least one eligible annotation were found.

## Retrieval results

The table reports the four predeclared configurations on the same 77 eligible questions. Latency uses all 97 queries per condition in the initial run, after warmup. Hybrid latency sums sequential sparse, dense and fusion timings. Reranked latency adds cross-encoder scoring and reordering.

| Scope | Method | Recall@1/5/10/20 | nDCG@1/5/10/20 | MRR@20 | Annotated set@10 | p50 / p95 (ms) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Paper | BM25 | 0.079 / 0.307 / 0.470 / 0.653 | 0.143 / 0.225 / 0.288 / 0.345 | 0.285 | 0.455 | 0.46 / 0.89 |
| Paper | BGE | 0.162 / 0.471 / 0.606 / 0.805 | 0.273 / 0.384 / 0.438 / 0.496 | 0.441 | 0.597 | 11.87 / 17.92 |
| Paper | RRF | 0.189 / 0.432 / 0.595 / 0.761 | 0.325 / 0.368 / 0.426 / 0.477 | 0.451 | 0.584 | 12.41 / 18.59 |
| Paper | RRF + MiniLM | 0.225 / 0.528 / 0.656 / 0.761 | 0.377 / 0.438 / 0.489 / 0.522 | 0.515 | 0.649 | 955.04 / 1175.01 |
| Split, question | BM25 | 0.023 / 0.106 / 0.223 / 0.295 | 0.065 / 0.085 / 0.127 / 0.149 | 0.139 | 0.221 | 6.36 / 11.81 |
| Split, question | BGE | 0.065 / 0.206 / 0.254 / 0.352 | 0.117 / 0.171 / 0.190 / 0.215 | 0.204 | 0.247 | 13.98 / 20.82 |
| Split, question | RRF | 0.108 / 0.242 / 0.305 / 0.399 | 0.208 / 0.214 / 0.235 / 0.263 | 0.278 | 0.312 | 20.52 / 31.28 |
| Split, question | RRF + MiniLM | 0.115 / 0.274 / 0.360 / 0.399 | 0.221 / 0.231 / 0.266 / 0.278 | 0.298 | 0.364 | 943.23 / 1219.82 |
| Split, title | BM25 | 0.034 / 0.142 / 0.300 / 0.505 | 0.091 / 0.113 / 0.169 / 0.230 | 0.178 | 0.260 | 11.48 / 16.94 |
| Split, title | BGE | 0.075 / 0.207 / 0.325 / 0.543 | 0.130 / 0.170 / 0.212 / 0.276 | 0.238 | 0.325 | 18.67 / 25.23 |
| Split, title | RRF | 0.035 / 0.186 / 0.332 / 0.546 | 0.065 / 0.140 / 0.187 / 0.252 | 0.192 | 0.312 | 30.36 / 41.68 |
| Split, title | RRF + MiniLM | 0.173 / 0.383 / 0.461 / 0.546 | 0.273 / 0.337 / 0.361 / 0.384 | 0.399 | 0.442 | 881.16 / 1310.92 |

### Paired uncertainty

Differences use the same questions in each pair. Sample papers with replacement, preserving all their paired question differences, and recompute the question-weighted mean. Use 2,000 draws with NumPy seed 42 and percentile endpoints at 2.5% and 97.5%. Evidence metrics cover 77 questions in 29 eligible papers. Target-paper hit uses 97 questions in 30 papers. Intervals are descriptive and unadjusted for multiple comparisons.

| Paper-scoped contrast | Δ Recall@10 [95% CI] | Δ nDCG@5 [95% CI] | Δ MRR@20 [95% CI] |
| --- | --- | --- | --- |
| BGE − BM25 | +0.136 [+0.059, +0.201] | +0.160 [+0.105, +0.216] | +0.156 [+0.086, +0.225] |
| RRF − BM25 | +0.125 [+0.075, +0.179] | +0.144 [+0.101, +0.193] | +0.166 [+0.107, +0.232] |
| RRF − BGE | -0.011 [-0.060, +0.048] | -0.016 [-0.052, +0.023] | +0.009 [-0.043, +0.066] |
| RRF + MiniLM − RRF | +0.060 [-0.004, +0.113] | +0.069 [+0.005, +0.132] | +0.064 [-0.012, +0.134] |
| RRF + MiniLM − BM25 | +0.186 [+0.108, +0.249] | +0.213 [+0.150, +0.278] | +0.230 [+0.145, +0.316] |
| RRF + MiniLM − BGE | +0.050 [-0.012, +0.111] | +0.054 [-0.003, +0.111] | +0.074 [+0.019, +0.136] |

RRF improved over BM25, but its paper-scoped Recall@10 and nDCG@5 were slightly below BGE. At rank 10, both baselines found some annotated evidence for 47 questions, BGE alone for 13, BM25 alone for one, and neither for 16. That complementarity was asymmetric. The reranker improved nDCG@5 over RRF, with an interval above zero in this sample. Its Recall@10 interval overlapped zero.

At fusion depth 25, paper-scoped Recall@10 was .552 versus .595 at depth 50, and nDCG@5 was .368 at both depths. Retain the predeclared depth of 50 for follow-up. Paragraph-only sensitivity on 75 questions gave Recall@10 of .467, .600, .589, and .649 in method order. The method ordering on Recall@10 was the same with both evidence rules.

### Title cue

No original question contained its full target title as a normalized case-insensitive substring. Short title fragments, acronyms, method names, and topic cues may still appear. Both runs shared the same passages, title-bearing index, and retrieval configurations.

| Method | Target-paper hit@10, question → title | Δ target hit@10 [95% CI] | Δ Recall@10 [95% CI] | Δ nDCG@5 [95% CI] |
| --- | --- | --- | --- | --- |
| BM25 | 0.711 → 1.000 | +0.289 [+0.182, +0.395] | +0.077 [-0.025, +0.178] | +0.028 [-0.009, +0.067] |
| BGE | 0.536 → 1.000 | +0.464 [+0.344, +0.582] | +0.070 [-0.021, +0.170] | -0.002 [-0.087, +0.074] |
| RRF | 0.660 → 1.000 | +0.340 [+0.240, +0.434] | +0.026 [-0.056, +0.113] | -0.074 [-0.146, -0.006] |
| RRF + MiniLM | 0.701 → 1.000 | +0.299 [+0.211, +0.392] | +0.101 [+0.010, +0.190] | +0.106 [+0.041, +0.186] |

The title prefix put the target paper in the first ten chunks for every question under all four methods in this pool. It also reduced RRF nDCG@5 and MRR@20, showing that paper identification and within-paper evidence placement can move in opposite directions. RRF + MiniLM improved both evidence metrics with the cue. The reproduction command below generates the paired Recall@10 figure directly from the local report.

### Example rankings

- **Recovered long-paragraph evidence.** For the annotation-bias question on paper `1910.12574` (question `893ec40b…`), RRF ranked the three windows of the gold paragraph at 1, 4, and 12. MiniLM moved the final required window into the first ten and completed coverage at rank 7.
- **Sparse complement.** For the dataset question on paper `1810.12085` (question `c571deef…`), BM25 found the gold paragraph at rank 10 and BGE at rank 13. Fusion kept it at 10 and MiniLM moved it to 2.
- **Reranker regression.** For the recurrent-network variant question on paper `1605.07333` (question `e2e977d7…`), MiniLM demoted the annotated paragraph from fused rank 6 to rank 17.
- **Candidate limit.** For the language-pair question on paper `1608.01084` (question `1f07e837…`), BGE ranked the annotated paragraph 14th. Fusion ranked it 44th, beyond the MiniLM candidate pool, where it stayed.

## Measured resources

The CPU-only run reused 1,214 Phase 2 vectors and computed 440 additional vectors. The additional-embedding and index-build stage took 37.12 seconds, including validation and cache writes, for 11.85 new chunks/s. The ten new embedding shards and checksum files occupied 677,780 bytes. The initial run's chunk manifest, evidence audit, rankings, and checkpoints occupied 248,445, 139,971, 8,020,307, and 3,452,928 bytes respectively.

On the i5-6300HQ, MiniLM loaded in 0.37 seconds. Reranking alone took 942 / 1,157 ms p50 / p95 in paper scope, 925 / 1,193 ms for uncued split-wide queries, and 855 / 1,277 ms for title-cued split-wide queries. The first run took approximately 319 seconds across nonoverlapping timed stages and 605 CPU-seconds. Its process working set peaked at 618 MiB. RSS was 359 MiB before MiniLM load, 391 MiB after load, and 618 MiB at the end of evaluation. The initial measurement retained the BGE session through reranking, so its RSS values reflect both sessions. The final runner releases the BGE session first. System-available memory at the first run's start and end was 2.75 and 2.87 GiB. No paging trace was collected.

The second run reused all 440 expansion vectors and recomputed every ranking with fresh model sessions. It took 296 seconds across timed stages and 539 CPU-seconds. With sequential model residency, working set peaked at **360 MiB**, and RSS went from 98 MiB before MiniLM load to 133 MiB after it. MiniLM loaded in 0.30 seconds. The final paper-scoped reranked p50 / p95 was 984 / 1,482 ms. The two runs had identical rankings and different latency distributions.

The MiniLM graph and tokenizer occupied 23,200,716 and 711,396 bytes. Phase 3 downloaded those two public artifacts, a 5,794-byte model metadata response, and the optional free Matplotlib 3.10.7 wheel and dependencies for rendering. It reused the existing QASPER archive and BGE model. The service cost was $0.

## Verification

All 20 tests passed, including the 12 Phase 2 tests. The additional tests cover rank-fusion arithmetic and ties, reranker input preservation, section evidence and unresolved labels, metric consistency, checkpoint integrity and configuration validation, and deterministic paper-cluster bootstrap.

An independent artifact audit verified both original Phase 2 manifests and all recorded Phase 3 artifact hashes. It checked the 1,654 chunk spans across 1,563 paragraphs and validated the checksums and norms of the ten new vector shards. All 291 query–scope rows preserved the original fused candidate set and tail under reranking. As expected for top-20 reordering, reranking preserved each eligible question's Recall@20.

The initial run, second run, and checkpoint replay had identical ranks, scores, candidate records, metrics and paired intervals for all 291 query–scope rows, with measured timings excluded from the equality check. The replay read 291 base and 291 reranker checkpoints and loaded neither inference model. Its timed stages took 3.07 seconds. Chunk and evidence manifests were byte-identical across runs, and the two independently rendered SVG figures were byte-identical.

## Limitations and Phase 4 recommendations

The 30-paper sample is development data, and its split-wide index is much smaller than the full development split. Sparse and dense statistics and the effect of a title cue may change at full scale. The 29 eligible paper clusters and multiple descriptive comparisons limit the precision and generality of the intervals. The experiment used one reranker checkpoint and quantization, and compared reranking only on fused candidates. A dense-only reranking arm would isolate the contribution of fusion. A top-20 reranker cannot recover evidence omitted by its candidate set. Section-expanded labels are coarse and benchmark annotations can omit valid passages.

Carry BGE forward as the low-latency generation retrieval default and retain RRF + MiniLM as a separate development comparison. Measure whether its improvement in early evidence placement survives context packing and improves grounded answers or citations enough to justify the latency. Use the agreed 40–60-question initial review budget and keep evidence sufficiency distinct from annotated set coverage. Audit split duplicates and freeze the held-out protocol before accessing test data. Qwen2.5-1.5B remains provisional until a separately authorized phase measures quality and hardware feasibility.

## Reproduction

Use the [Phase 2 setup](phase2.md) and original `artifacts/pilot-cold` and `artifacts/cache`. With a fresh Phase 2 build from that guide, pass `--phase2 artifacts/replay-build --phase2-cache artifacts/replay-cache` to the Phase 3 command. With the same Python environment and thread settings, add the pinned MiniLM assets and run:

```powershell
New-Item -ItemType Directory -Force model/minilm | Out-Null
$miniLM = 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a'
curl.exe --fail --location "$miniLM/onnx/model_quint8_avx2.onnx" -o model/minilm/model.onnx
curl.exe --fail --location "$miniLM/tokenizer.json" -o model/minilm/tokenizer.json
& $pilotPython -m unittest discover -s tests -v
& $pilotPython -m rag_eval.phase3 --output artifacts/phase3-reproduction
& $pilotPython -m pip install matplotlib==3.10.7
& $pilotPython -m rag_eval.phase3_figure artifacts/phase3-reproduction/report.json artifacts/phase3-reproduction/recall-effects.svg
```

Rerun the same CLI command against the same output directory to resume. Set `rerank_depth = 0` in a copied configuration to run the three retrieval methods without MiniLM. Per-paper embedding caches verify the array checksum and shape. Per-query SQLite checkpoints verify the payload checksum and include a configuration, source, dependency, and implementation fingerprint. Reports retain original stage latencies from checkpointed base searches and reranking. Fusion is timed again during replay. The full results, manifests, and caches remain in ignored local artifacts.
