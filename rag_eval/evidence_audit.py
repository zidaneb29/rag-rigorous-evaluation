"""Audit heading annotations without changing Phase 2 mapping conventions."""

from collections import defaultdict
from dataclasses import replace

from .corpus import normalize
from .evaluation import map_evidence


RESOLVED = {"mapped", "mapped_section"}


def audited_evidence(paper_id, paper, question, paragraphs, chunks):
    original = map_evidence(question, paragraphs, chunks)
    headings = defaultdict(list)
    indexed = {chunk.paragraph_id for chunk in chunks}
    for index, section in enumerate(paper["full_text"]):
        heading = normalize(section["section_name"] or "")
        if heading:
            ids = [f"{paper_id}:{index}:{i}" for i, text in enumerate(section["paragraphs"])
                   if text.strip()]
            headings[heading].append((index, ids))
    annotations, sets = [], []
    for source, row in zip(question["answers"], original.annotations):
        labels = []
        for label, prior in zip(source["answer"]["evidence"], row["labels"]):
            current = {**prior, "phase2_status": prior["status"]}
            if prior["status"] in {"unmatched", "mapped"}:
                matches = headings.get(normalize(label), [])
                if matches:
                    current["section_ordinals"] = [ordinal for ordinal, _ in matches]
                    if prior["status"] == "mapped":
                        current.update(status="ambiguous_heading_or_paragraph")
                    elif len(matches) > 1:
                        current.update(status="ambiguous_section")
                    else:
                        ids = matches[0][1]
                        current.update(paragraph_ids=ids,
                                       status="mapped_section" if ids and set(ids) <= indexed
                                       else "section_without_indexable_body")
                elif prior["status"] == "unmatched":
                    caption_matches = [i for i, item in enumerate(paper["figures_and_tables"])
                                       if normalize(item.get("caption") or "") == normalize(label)]
                    if caption_matches:
                        current.update(status="standalone_caption", caption_ordinals=caption_matches)
                    else:
                        fragments = [p.id for p in paragraphs if normalize(label)
                                     and normalize(label) in normalize(p.text)]
                        current.update(status="paragraph_fragment" if fragments else "absent_exact_match",
                                       fragment_paragraph_ids=fragments)
            labels.append(current)
        annotations.append({**row, "labels": labels})
        if row["answer_type"] != "unanswerable" and labels and all(
            label["status"] in RESOLVED for label in labels
        ):
            sets.append({pid for label in labels for pid in label["paragraph_ids"]})
    status = original.status
    if status in {"eligible", "no_complete_text_set"}:
        status = "eligible" if sets else "no_complete_text_set"
    return original, replace(original, status=status, sets=sets, annotations=annotations)
