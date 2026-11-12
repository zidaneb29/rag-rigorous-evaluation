# Implementation roadmap

Advance after each phase passes its verification gate. Any design changes after the freeze require a dated protocol revision. Changes made after held-out inspection are exploratory.

| Phase | Deliverables | Acceptance criteria and verification |
| --- | --- | --- |
| 1. Specification and benchmark design | Overview, technical design, evaluation protocol, resource budget, roadmap | Review the dataset choice, split rules, metrics, resource gates and source links. Obtain the user's review before implementing. |
| 2. Ingestion and retrieval baselines | Source and split manifests, normalized corpus, paragraph mappings, BM25 and dense indexes, CLI, baseline artifacts | Verify source IDs, split isolation and passage offsets, including examples with long or duplicate paragraphs. Check metrics against hand-calculated fixtures. Rebuild an index from the same inputs and compare ranks. Run an offline Windows CPU pilot and record resource use. |
| 3. Fusion and reranking | RRF and cross-encoder, evidence audit, development ablations and retrieval artifacts | Completed the bounded [Phase 3 development comparison](phase3.md). Hand-check fusion, verify candidate preservation and replay, and review paired effects and resource costs. Freeze all held-out settings before test access. |
| 4. Generation feasibility | Local adapter, context packer, citation schema and bounded feasibility pilots | The [1.5B pilot](phase4.md), [prompt recovery](phase4b.md), and [7B Colab sanity check](phase4c.md) documented failed feasibility gates. Full generation evaluation remains incomplete. Generator work is paused. |
| 5. Retrieval reliability and final evaluation | Frozen retrieval protocol, development robustness check, held-out run, paired estimates, and error audit | Verify split isolation and evidence mapping before scoring. Compare BM25, BGE, RRF and RRF + MiniLM on identical questions and candidate scope. Check metric denominators, timings, intervals, and example ranks against saved results. |
| 6. Streamlit UI | Local ranked search, expandable evidence passages, and measured evaluation dashboard | Verify search and passage provenance on the Windows CPU setup. Reconcile dashboard values with CLI artifacts and test missing-index, missing-artifact and empty-result states. |
| 7. Publication | Portfolio README, concise technical report, reproducible figures, tests, setup and GitHub presentation | Reproduce setup from a clean checkout, verify figure values and artifact hashes, and review attribution and redistribution rights. Publication requires separate authorization. |

## Remaining scope

### Phase 5 · Retrieval reliability and final evaluation

- Freeze dataset and index hashes, duplicate handling, chunking, evidence mappings, eligibility, method settings, metric cutoffs, and the held-out protocol before inspecting test results. Audit cross-split paper IDs and near duplicates, and preserve section-heading, non-text, ambiguity and exclusion counts. Report paragraph-only sensitivity and annotated-set coverage separately from human evidence sufficiency.
- Keep paper-scoped retrieval primary and split-wide retrieval secondary. Use the existing title-cue ablation and a small, seeded, manually checked development paraphrase panel to test robustness. Freeze configurations on development data. Run the authorized held-out comparison once, with the same questions and scope across methods.
- Report Recall@k, MRR, nDCG, paired paper-cluster bootstrap intervals, warm p50/p95 latency, index size and peak memory. Keep uncertainty and efficiency visible alongside point estimates. Review a small fixed set of successes and failures covering missing candidates, split windows, and reranking regressions. Prefer one comparison figure and one latency–quality figure over a larger tuning grid.

### Phase 6 · Streamlit UI

- Build local search with a question field, optional paper scope, method selection, ranked results, and expandable passages showing title, section, source link and rank. Use existing retrievers and artifacts. Display latency and distinguish known annotations from unlabeled passages.
- Build a dashboard from recorded CLI results with explicit split, retrieval scope, configuration, sample size and uncertainty. A read-only feasibility panel may show saved generator examples and their outcomes. Live generation is outside the interface scope.

### Phase 7 · Publication

- Lead with the retrieval research question and measured findings. Add a short technical report, reproducible figures, a setup and replay guide, a Streamlit screenshot, and test instructions. Include the failed generation feasibility gates and the limits of the retrieval evidence.
- Publish only after a clean Windows replay and a rights review. Keep weights, corpora, and raw excerpts out of the public repository unless redistribution rights are established. Use IDs, hashes, aggregates, and download instructions for reproducibility.

The [Phase 5 report](phase5.md) records the held-out sample and replay. The [Phase 6 guide](phase6.md) records the local app and verification. Phase 7 awaits authorization.
