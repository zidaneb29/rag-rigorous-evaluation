"""Load and validate the original QASPER distribution."""

import hashlib
import json
import tarfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ARCHIVES = {
    "train": ("qasper-train-dev-v0.3.tgz", "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"),
    "dev": ("qasper-train-dev-v0.3.tgz", "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"),
    "test": ("qasper-test-and-evaluator-v0.3.tgz", "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"),
}
SOURCE = "https://qasper-dataset.s3.us-west-2.amazonaws.com/"


@dataclass(frozen=True)
class Paragraph:
    paper_id: str
    id: str
    section: str
    text: str


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _field(record, name, kind):
    if not isinstance(record, dict) or type(record.get(name)) is not kind:
        raise ValueError(f"Expected {name} to be {kind.__name__}")
    return record[name]


def validate(papers):
    if not isinstance(papers, dict) or not papers:
        raise ValueError("Expected a nonempty mapping keyed by paper ID")
    question_ids = set()
    for paper_id, paper in papers.items():
        try:
            _field(paper, "title", str)
            _field(paper, "abstract", str)
            for section in _field(paper, "full_text", list):
                if section["section_name"] is not None:
                    _field(section, "section_name", str)
                if any(not isinstance(p, str) for p in _field(section, "paragraphs", list)):
                    raise ValueError("Expected paragraph strings")
            _field(paper, "figures_and_tables", list)
            for question in _field(paper, "qas", list):
                question_id = _field(question, "question_id", str)
                _field(question, "question", str)
                if not question_id or question_id in question_ids:
                    raise ValueError(f"Missing or duplicate question ID: {question_id}")
                question_ids.add(question_id)
                for annotation in _field(question, "answers", list):
                    _field(annotation, "annotation_id", str)
                    answer = _field(annotation, "answer", dict)
                    _field(answer, "unanswerable", bool)
                    _field(answer, "free_form_answer", str)
                    for field in ("extractive_spans", "evidence"):
                        if any(not isinstance(item, str) for item in _field(answer, field, list)):
                            raise ValueError(f"Expected strings in {field}")
                    if answer["yes_no"] is not None:
                        _field(answer, "yes_no", bool)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid QASPER paper {paper_id}: {exc}") from exc


def load_split(directory, split):
    archive_name, checksum = ARCHIVES[split]
    path = Path(directory) / archive_name
    if sha256(path) != checksum:
        raise ValueError(f"QASPER archive checksum mismatch: {path}")
    member = f"qasper-{split}-v0.3.json"
    with tarfile.open(path, "r:gz") as archive:
        matches = [item for item in archive if item.isfile() and item.name == member]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one archive member named {member}")
        with archive.extractfile(matches[0]) as stream:
            papers = json.load(stream, object_pairs_hook=_unique_object)
    validate(papers)
    return papers, {"release": "v0.3", "split": split, "url": SOURCE + archive_name,
                    "sha256": checksum, "member": member, "bytes": path.stat().st_size}


def paragraphs(paper_id, paper):
    yield Paragraph(paper_id, f"{paper_id}:abstract", "Abstract", paper["abstract"])
    for section_number, section in enumerate(paper["full_text"]):
        for paragraph_number, raw in enumerate(section["paragraphs"]):
            yield Paragraph(paper_id, f"{paper_id}:{section_number}:{paragraph_number}",
                            section["section_name"] or "", raw)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())
