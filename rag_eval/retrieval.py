"""BM25 and CPU-only BGE with exact cosine similarity."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import sha256


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def terms(text):
    return re.findall(r"\w+", text.casefold())


class BGE:
    def __init__(self, model, tokenizer, batch_size, threads):
        import onnxruntime as ort

        if batch_size < 1 or threads < 1:
            raise ValueError("Batch size and thread count must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [field.name for field in self.session.get_inputs()]
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def encode(self, texts, *, query=False):
        vectors = np.empty((len(texts), 384), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if query:
                batch = [QUERY_PREFIX + text for text in batch]
            encoded = self.tokenizer.encode_batch(batch)
            fields = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            width = min(512, max(len(row.ids) for row in encoded))
            pad_id = self.tokenizer.token_to_id("[PAD]")
            for row in encoded:
                padding = max(0, width - len(row.ids))
                ids = row.ids[:width]
                if len(row.ids) > width:
                    ids[-1] = self.tokenizer.token_to_id("[SEP]")
                fields["input_ids"].append(ids + [pad_id] * padding)
                fields["attention_mask"].append(row.attention_mask[:width] + [0] * padding)
                fields["token_type_ids"].append(row.type_ids[:width] + [0] * padding)
            output = self.session.run(None, {name: np.asarray(fields[name], dtype=np.int64)
                                             for name in self.input_names})[0]
            pooled = output[:, 0] if output.ndim == 3 else output
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                raise ValueError("BGE produced an invalid embedding")
            vectors[start:start + len(batch)] = pooled / norms
        return vectors


def cached_vectors(directory, chunks, signature, encoder):
    identity = hashlib.sha256(signature.encode())
    for chunk in chunks:
        for field in (chunk.id, chunk.text):
            value = field.encode()
            identity.update(len(value).to_bytes(8, "big"))
            identity.update(value)
    key = identity.hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npy"
    digest_path = directory / f"{key}.sha256"
    hit = path.exists() and digest_path.exists()
    if hit:
        if sha256(path) != digest_path.read_text(encoding="ascii").strip():
            raise ValueError(f"Embedding cache checksum mismatch: {path}")
    else:
        matrix = encoder.encode([chunk.text for chunk in chunks])
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, matrix, allow_pickle=False)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        with NamedTemporaryFile(dir=directory, mode="w", encoding="ascii", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(sha256(path) + "\n")
        try:
            temporary.replace(digest_path)
        finally:
            temporary.unlink(missing_ok=True)
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(chunks), 384) or matrix.dtype != np.float32:
        raise ValueError(f"Invalid embedding cache shape or dtype: {path}")
    return matrix, path, hit


class Retrieval:
    def __init__(self, chunks, *, method, vectors=None, encoder=None):
        if method not in {"bm25", "dense"} or not chunks:
            raise ValueError("A valid method and nonempty chunk index are required")
        self.method = method
        self.chunks = chunks
        self.vectors = vectors
        self.encoder = encoder
        self.by_paper = defaultdict(list)
        for position, chunk in enumerate(chunks):
            self.by_paper[chunk.paper_id].append(position)
        if method == "bm25":
            self.bm25 = BM25Okapi([terms(chunk.text) for chunk in chunks])

    def search(self, query, k, paper_id=None):
        positions = range(len(self.chunks)) if paper_id is None else self.by_paper.get(paper_id, ())
        if not positions:
            return []
        if self.method == "bm25":
            scores = self.bm25.get_batch_scores(terms(query), positions)
        else:
            vector = self.encoder.encode([query], query=True)[0]
            matrix = self.vectors if paper_id is None else self.vectors[positions]
            scores = matrix @ vector
        ranked = sorted(zip(positions, scores), key=lambda pair: (-pair[1], self.chunks[pair[0]].id))
        return [(self.chunks[position], float(value)) for position, value in ranked[:k]]
