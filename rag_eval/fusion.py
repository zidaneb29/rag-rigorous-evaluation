"""Deterministic rank fusion and a modular CPU cross-encoder."""

from collections import defaultdict
import math

import numpy as np


def reciprocal_rank_fusion(rankings, constant, depth):
    if constant <= 0 or depth < 1:
        raise ValueError("RRF constant and depth must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, (chunk_id, _) in enumerate(ranking[:depth], 1):
            if chunk_id in seen:
                raise ValueError("Duplicate chunk ID in ranking")
            seen.add(chunk_id)
            scores[chunk_id] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))


def rerank(query, fused, chunks, model, depth):
    candidates = fused[:depth]
    if not candidates:
        return [], fused
    scores = model.score(query, [chunks[cid].text for cid, _ in candidates])
    if len(scores) != len(candidates) or not all(map(math.isfinite, scores)):
        raise ValueError("Invalid cross-encoder logits")
    inputs = [{"chunk_id": cid, "fusion_rank": rank, "fusion_score": prior,
               "logit": float(logit)}
              for rank, ((cid, prior), logit) in enumerate(zip(candidates, scores), 1)]
    head = sorted(inputs, key=lambda row: (-row["logit"], row["fusion_rank"], row["chunk_id"]))
    return inputs, [(row["chunk_id"], row["logit"]) for row in head] + fused[depth:]


class MiniLM:
    def __init__(self, model_path, tokenizer, threads, batch, max_tokens):
        import onnxruntime as ort

        tokenizer.enable_truncation(max_length=max_tokens, strategy="longest_first")
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
        self.tokenizer = tokenizer
        self.batch = batch
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.inputs = [field.name for field in self.session.get_inputs()]

    def score(self, query, passages):
        scores = []
        for start in range(0, len(passages), self.batch):
            rows = self.tokenizer.encode_batch([(query, text) for text in passages[start:start + self.batch]])
            fields = {"input_ids": [row.ids for row in rows],
                      "attention_mask": [row.attention_mask for row in rows],
                      "token_type_ids": [row.type_ids for row in rows]}
            logits = self.session.run(None, {name: np.asarray(fields[name], dtype=np.int64)
                                             for name in self.inputs})[0]
            if logits.shape != (len(rows), 1):
                raise ValueError(f"Expected one logit per pair, got {logits.shape}")
            scores.extend(logits[:, 0].tolist())
        return scores
