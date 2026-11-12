"""Map benchmark evidence and score retrieval against observed annotations."""

from collections import defaultdict
from dataclasses import dataclass
from math import log2

from .corpus import normalize


@dataclass
class Gold:
    status: str
    sets: list
    bounds: dict
    annotations: list


def map_evidence(question, paragraphs, chunks):
    by_text = defaultdict(list)
    bounds = {}
    for paragraph in paragraphs:
        if paragraph.text.strip():
            by_text[normalize(paragraph.text)].append(paragraph.id)
            bounds[paragraph.id] = (len(paragraph.text) - len(paragraph.text.lstrip()),
                                    len(paragraph.text.rstrip()))
    indexed = {chunk.paragraph_id for chunk in chunks}
    annotations, sets = [], []
    answerable = 0
    for annotation in question["answers"]:
        answer = annotation["answer"]
        answer_type = ("unanswerable" if answer["unanswerable"] else
                       "extractive" if answer["extractive_spans"] else
                       "yes_no" if answer["yes_no"] is not None else
                       "free_form" if answer["free_form_answer"] else "unspecified")
        answerable += not answer["unanswerable"]
        labels = []
        for index, evidence in enumerate(answer["evidence"]):
            matches = by_text[normalize(evidence)]
            if evidence.strip().upper().startswith("FLOAT SELECTED"):
                status = "non_text"
            elif len(matches) > 1:
                status = "ambiguous"
            elif not matches:
                status = "unmatched"
            elif matches[0] not in indexed:
                status = "not_indexed"
            else:
                status = "mapped"
            labels.append({"label_index": index, "status": status, "paragraph_ids": matches})
        annotations.append({"annotation_id": annotation["annotation_id"],
                            "answer_type": answer_type, "labels": labels})
        if not answer["unanswerable"] and labels and all(row["status"] == "mapped" for row in labels):
            sets.append({row["paragraph_ids"][0] for row in labels})
    count = len(annotations)
    if not count:
        status = "no_annotations"
    elif answerable * 2 == count:
        status = "answerability_tie"
    elif answerable * 2 < count:
        status = "unanswerable"
    else:
        status = "eligible" if sets else "no_complete_text_set"
    return Gold(status, sets, bounds, annotations)


def _covers(spans, bounds):
    covered_until, stop = bounds
    for start, end in sorted(spans):
        if start > covered_until:
            break
        covered_until = max(covered_until, end)
        if covered_until >= stop:
            return True
    return False


def score(ranked_chunks, gold, cutoffs):
    if gold.status != "eligible":
        return {}
    relevant = set.union(*gold.sets)
    coverage = defaultdict(list)
    found = set()
    gains, result = [], {}
    first_rank = 0
    for rank in range(1, max(cutoffs) + 1):
        gain = 0
        if rank <= len(ranked_chunks):
            chunk = ranked_chunks[rank - 1]
            pid = chunk.paragraph_id
            if pid in relevant and pid not in found:
                coverage[pid].append((chunk.start, chunk.end))
                if _covers(coverage[pid], gold.bounds[pid]):
                    found.add(pid)
                    gain = 1
                    first_rank = first_rank or rank
        gains.append(gain)
        if rank in cutoffs:
            dcg = sum(value / log2(i + 2) for i, value in enumerate(gains))
            ideal = sum(1 / log2(i + 2) for i in range(min(rank, len(relevant))))
            result[f"recall@{rank}"] = len(found) / len(relevant)
            result[f"ndcg@{rank}"] = dcg / ideal
            result[f"annotated_set_coverage@{rank}"] = float(any(s <= found for s in gold.sets))
    result[f"mrr@{max(cutoffs)}"] = 1 / first_rank if first_rank else 0.0
    return result
