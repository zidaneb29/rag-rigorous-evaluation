# Phase 4B generator recovery

## Protocol

The recovery experiment used the same Apache-2.0 Qwen2.5-1.5B-Instruct Q4_K_M GGUF, llama.cpp b7300 build, and CPU settings as Phase 4. It verified the original generation records, checkpoint database, model, and runtime executable by SHA-256. The original eight development questions retained their IDs, answerability labels, passage IDs, order, text and source offsets. No held-out data was accessed.

| Configuration | Instruction change | Output contract |
| --- | --- | --- |
| A | Exact Phase 4 instruction, including the literal abstention example | Original JSON schema and parser |
| B | Concise grounding instruction that prioritizes answering from sufficient evidence and describes both paths in prose | Same schema and parser |
| C | B plus one synthetic answered example and one synthetic abstention example, in that order | Same schema and parser |

The two demonstrations use a cedar garden marker and a storage box with an unknown painter. They share no answers with the eight evaluation questions or the synthetic checks. The three development conditions share the same final user message, frozen evidence and passage IDs, 2,048-token context, 1,250-token evidence allowance, five-passage limit, 176-token output limit, greedy decoding, seed 42 and disabled prompt reuse. Added prompt text is counted separately from the evidence increment. The server receives exact token IDs, and the recovery runner checks prompt lengths and runtime truncation.

The six synthetic calls are a separate diagnostic. Each of three original passages is tested once with ordinary text and once with A's structured instructions and grammar. That comparison changes both instruction and output contract, so it cannot isolate the grammar. The passages cover a stated color, a simple inference from opening hours, and a missing person's name.

The recovery decision requires at least four of six reference questions answered correctly with supporting citations, both native negatives abstained, and valid structured output for the eight cases. A prompt meeting that screen would justify at most one disjoint development confirmation sample. The screen deliberately counts supported answers rather than valid JSON or answer frequency alone.

## Original implementation audit

- The original system text instructs the model to use only supplied passages, abstain if the passages do not establish the answer, and return one or two cited claims. Its only literal output example is `{"abstain":true,"claims":[]}`. The final user turn is a JSON object with the question and an array of passage IDs and raw texts. Chat control markers in the source strings are escaped.
- The server's embedded Qwen chat template renders the system and user roles and opens the assistant turn. A requires the exact token SHA-256 saved in Phase 4 for each prompt. The runner checks the same role rendering, prompt length, and budget against the live server.
- The schema requires a boolean `abstain` and a `claims` array. Each claim has `text` and `citations`. The schema permits contradictory combinations such as an abstention with nonempty claims. The parser rejects those and JSON failures as `malformed` with `abstained = null`. It only accepts an explicit boolean true with an empty array as abstention. Citation IDs outside the visible set are recorded as errors and removed from the resolved links.
- The packer numbered included passages consecutively from P1, kept source offsets, and recorded any dropped passages. The six original reference inputs had full mapped evidence coverage. Recovery reuses those exact contexts. The request uses the model's EOS handling with no extra stop string or forced abstention fallback. Phase 4 outputs stopped normally at EOS after 11 tokens, well below the cap.
- The original raw output explicitly set `abstain` to true on all eight questions. Parsing and validation did not create those abstentions. The original model record remains unchanged.

## Synthetic results

| Synthetic question | Ordinary text | Original structured approach |
| --- | --- | --- |
| Stated color | Said “Insufficient evidence.” The passage explicitly gave blue. | Extracted blue and cited P1, but set `abstain: true` alongside the claim. |
| Collection after closing | Said “Insufficient evidence.” The two rules imply no. | Cited the closing and collection rules, but did not state the answer and set `abstain: true` alongside two claims. |
| Owner's name absent | Said “Insufficient evidence.” | Cited the bicycle's location while setting `abstain: true` with a nonempty claim. |

Ordinary text resolved one of three cases as expected and failed both supported questions. The structured outputs were JSON in all three cases, but zero passed the parser's abstention contract. All three stopped at EOS. The structured result for the color shows that the model could extract the simple fact while failing to choose a coherent answer state. Removing the grammar together with changing the instruction did not restore answering under the ordinary-text prompt tested here.

## Eight-question reassessment

| Prompt | Answered, 6 answerable | Abstained, 2 native negatives | Valid contract | Resolving citation references | Latency p50 / p95 | Median decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, original | 0/6 | 2/2 | 8/8 | No links | 4.58 / 12.88 s | 19.38 |
| B, simplified | 0/6 | 1/2 | 8/8 | 2/2 | 5.56 / 16.60 s | 14.75 |
| C, balanced few-shot | 5/6 | 1/2 | 8/8 | 7/7 | 8.24 / 20.24 s | 14.94 |

All 24 development calls passed the parser, with no missing or invalid citation IDs among emitted claims. All 30 calls, including the six synthetic calls, stopped normally at EOS without context truncation or token-limit failures. The recovery test verified that A reproduced the same prompt token hashes and exact raw outputs as the original pilot, and that all 24 development contexts exactly matched their saved originals.

Adding the examples in C changed behavior while the schema, parser, formatting of the evaluation evidence, decoder, and checkpoint remained fixed. This identifies prompt sensitivity and demonstrates that the same model can answer some supported technical questions. The simplified wording in B did not recover any answerable case, so the original literal abstention example alone does not explain the failure. The synthetic checks exposed an answer-state inconsistency permitted by the schema. They do not isolate the effects of the grammar and text prompt. The run did not independently vary evidence formatting, decoding, or quantization.

## Technical answer and citation inspection

The following is an assistant inspection against the visible passages and saved QASPER references. It is a diagnostic assessment and has no independent human adjudication.

| Reference question | C output assessment |
| --- | --- |
| `1ed49a8c07…`, state-of-the-art dataset | Correct. P2 explicitly identifies the English–German dataset. |
| `45a5961a4e…`, number of question types | Correct. P1 explicitly states seven. |
| `3e432d7151…`, datasets used | Incomplete. “The training set is used” does not identify the self-collected Portuguese financial-intent data described in P1 and the references. |
| `07b70b2b79…`, dataset size | Abstained. P2 gives the Wikipedia count. The annotated context omits the arXiv count in the reference, so full contextual sufficiency is uncertain. |
| `1097768b89…`, cross-domain transfer of best methods | Unsupported as stated. P1 gives comparative results but does not establish the same best method across domains. The two benchmark references disagree. |
| `52f9cd05d8…`, robustness and general knowledge | Correct. P1 attributes the robustness gap in part to humans' ability to use general knowledge. |

Both B and C correctly declined the question asking which fonts were best. Both answered the other native negative, which asked whether an indexing method transferred beyond Wikipedia. B repeated a positive transfer claim with P1 and P2. C issued both positive and negative transfer claims and cited P1 for both. The cited passages describe a Wikipedia procedure and establish neither assertion. Citation IDs resolved in each case, but the links did not substantiate the answers.

**Decision.** C yielded three clearly correct, supported answers out of six and abstained on one of two native negatives. The recovery screen required four supported answers and two abstentions. No prompt passed, so no independent confirmation sample was run. The balanced prompt recovers some evidence use but this checkpoint and protocol remain unreliable for the full paired generation study.

## Resources and verification

The server loaded in **2.57 seconds** on the i5-6300HQ with four threads and zero GPU layers. The 30 completion calls took **220.86 seconds** combined, excluding startup, prompt tokenization, and artifact writes. Server peak working set was **1.89 GiB**. Records occupied 248,104 bytes and the checkpoint database 262,144 bytes. The model and runtime were reused and service cost was $0.

All **34 tests passed**, including the 30 earlier tests. New tests verify exact A message construction, fixed development evidence and schema, balanced and distinct demonstrations, and rejection of inconsistent abstention states. Replay recovered all 30 checkpoints without launching the server. Original Phase 4 record and checkpoint hashes still match. No other phases were run.

## Recommended next step

If another generator trial is authorized, test the Apache-2.0 Qwen2.5-7B-Instruct Q4_K_M release in a small interactive free-Colab GPU pilot. Its two weight files total about 4.68 GB. As a planning allowance, seek at least 8 GiB of free device memory for weights, context, and runtime buffers, then measure actual use with the same 2,048-token budget. A larger model increases CPU and memory cost, and its technical QA and abstention behavior on this task are unverified. Free GPU access is variable. Pin the weights and runtime, export checkpoints, and measure load and latency on a few new development questions before allocating a full batch. [Official model and license](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) · [Weight files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main) · [Colab resource limits](https://research.google.com/colaboratory/faq.html)

## Reproduction

Use the pinned assets and environment in [Phase 4](phase4.md). No new model or runtime downloads are required. Run from the repository root:

```powershell
& $pilotPython -m unittest discover -s tests -v
& $pilotPython -m rag_eval.phase4b --output artifacts/phase4b-reproduction
```

Repeat with the same output directory to replay completed records. Records include full chat turns, rendered prompts, token counts and hashes, generation settings, raw output, parser status, and context provenance. The local runtime uses loopback. The output directory is separate from the original Phase 4 pilot.
