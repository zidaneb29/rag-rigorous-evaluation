"""Deterministic paragraph windows and source character coordinates."""

from dataclasses import dataclass

from .corpus import paragraphs


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    paragraph_id: str
    start: int
    end: int
    text: str


def build_chunks(papers, tokenizer, max_tokens, overlap):
    if not 0 <= overlap < max_tokens - 2 or max_tokens > 512:
        raise ValueError("Invalid token limit or overlap")
    chunks = []
    for paper_id, paper in sorted(papers.items()):
        for paragraph in paragraphs(paper_id, paper):
            raw = paragraph.text
            offsets = tokenizer.encode(raw, add_special_tokens=False).offsets
            if not offsets:
                continue
            # Leave room for the body, overlap, and BERT special tokens.
            heading = f"{paper['title']}\n{paragraph.section}".strip()
            prefix_cap = min(48, max_tokens - overlap - 3)
            heading_offsets = tokenizer.encode(heading, add_special_tokens=False).offsets
            kept_heading = heading_offsets[:prefix_cap]
            prefix = heading[:kept_heading[-1][1]] + "\n" if kept_heading else ""
            capacity = max_tokens - len(tokenizer.encode(prefix).ids)
            start_token = 0
            while start_token < len(offsets):
                end_token = min(start_token + capacity, len(offsets))
                start = offsets[start_token][0] if start_token else 0
                while True:
                    end = offsets[end_token][0] if end_token < len(offsets) else len(raw)
                    body = raw[start:end]
                    if len(tokenizer.encode(prefix + body).ids) <= max_tokens:
                        break
                    end_token -= 1
                    if end_token <= start_token:
                        raise ValueError(f"Cannot fit body token in {paragraph.id}")
                chunks.append(Chunk(f"{paragraph.id}@{start}:{end}", paper_id,
                                    paragraph.id, start, end, prefix + body))
                if end_token == len(offsets):
                    break
                start_token = max(start_token + 1, end_token - overlap)
    return chunks
