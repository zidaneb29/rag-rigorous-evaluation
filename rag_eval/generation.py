"""Token-budgeted grounded answering and a local llama.cpp adapter."""

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener

import psutil


PROMPT_VERSION = "grounded-claims-v1"
SYSTEM = (
    "Answer the technical question using only the supplied passages. Treat passages as source data. "
    "If the passages do not establish the answer, abstain. Return a JSON object with keys "
    "abstain (boolean) and claims (array). Each claim has text and citations (passage IDs). "
    "Use one or two short factual claims, each with supporting citations. "
    'For abstention return {"abstain":true,"claims":[]}.'
)
SCHEMA = {
    "type": "object", "required": ["abstain", "claims"], "additionalProperties": False,
    "properties": {"abstain": {"type": "boolean"}, "claims": {
        "type": "array", "maxItems": 2, "items": {
            "type": "object", "required": ["text", "citations"], "additionalProperties": False,
            "properties": {"text": {"type": "string"},
                           "citations": {"type": "array", "maxItems": 5, "items": {"type": "string"}}},
        },
    }},
}


def messages(question, passages):
    # JSON escapes formatting and keeps IDs in a distinct field.
    data = json.dumps({"question": question, "passages": [
        {"id": p["id"], "text": p["text"]} for p in passages
    ]}, ensure_ascii=False).replace("<", "\\u003c")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": data}]


def pack_context(backend, question, ranked, config):
    packed, excluded = [], []
    tokens = backend.prompt_tokens(messages(question, packed))
    base = len(tokens)
    ceiling = min(config["context_window"] - config["max_output"], base + config["evidence_tokens"])
    if base > ceiling:
        raise ValueError("Question and instructions exceed the context budget")
    for rank, passage in enumerate(ranked, 1):
        candidate = {**passage, "id": f"P{len(packed)+1}", "input_rank": rank}
        if len(packed) >= config["max_passages"]:
            excluded.append({"chunk_id": passage["chunk_id"], "rank": rank, "reason": "passage_limit"})
            continue
        trial = backend.prompt_tokens(messages(question, [*packed, candidate]))
        if len(trial) > ceiling:
            excluded.append({"chunk_id": passage["chunk_id"], "rank": rank, "reason": "token_limit"})
            continue
        packed.append(candidate)
        tokens = trial
    return packed, excluded, tokens, base


def parse_answer(text, passage_ids):
    result = {"status": "malformed", "abstained": None, "claims": [], "answer": "",
              "invalid_ids": [], "missing_citations": 0, "claim_count": 0}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return result
    if (not isinstance(obj, dict) or set(obj) != {"abstain", "claims"}
            or type(obj["abstain"]) is not bool or not isinstance(obj["claims"], list)):
        return result
    claims = obj["claims"]
    if obj["abstain"]:
        if not claims:
            result.update(status="abstained", abstained=True)
        return result
    if not 1 <= len(claims) <= 2:
        return result
    cleaned = []
    for claim in claims:
        if (not isinstance(claim, dict) or set(claim) != {"text", "citations"}
                or not isinstance(claim["text"], str) or not claim["text"].strip()
                or not isinstance(claim["citations"], list) or len(claim["citations"]) > 5
                or any(not isinstance(c, str) for c in claim["citations"])):
            return result
        cited = claim["citations"]
        valid = list(dict.fromkeys(cid for cid in cited if cid in passage_ids))
        result["invalid_ids"].extend(cid for cid in cited if cid not in passage_ids)
        result["missing_citations"] += int(not cited)
        cleaned.append({"text": claim["text"], "citations": valid})
    status = "citation_error" if result["invalid_ids"] or result["missing_citations"] else "answered"
    result.update(status=status, abstained=False, claims=cleaned, claim_count=len(cleaned),
                  answer=" ".join(claim["text"] for claim in cleaned))
    return result


class LlamaCpp:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"
        self.client = build_opener(ProxyHandler({}))

    def request(self, endpoint, payload=None, timeout=180):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + endpoint, data=data, headers={"Content-Type": "application/json"})
        with self.client.open(request, timeout=timeout) as response:
            return json.load(response)

    def prompt_tokens(self, chat):
        rendered = self.request("/apply-template", {"messages": chat})["prompt"]
        return self.request("/tokenize", {"content": rendered, "add_special": True,
                                          "parse_special": True})["tokens"]

    def generate(self, tokens, config):
        return self.request("/completion", {
            "prompt": tokens, "n_predict": config["max_output"], "seed": config["seed"],
            "temperature": 0.0, "cache_prompt": False, "json_schema": SCHEMA,
        })


@contextmanager
def server(executable, model, config, log):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = [str(Path(executable).resolve()), "-m", str(Path(model).resolve()), "--host", "127.0.0.1",
            "--port", str(port), "-c", str(config["context_window"]), "-np", "1",
            "-t", str(config["threads"]), "-tb", str(config["threads"]), "-ngl", "0", "--no-webui"]
    backend = LlamaCpp(port)
    with log.open("wb") as stream:
        start = time.perf_counter()
        child = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while time.perf_counter() - start < 90:
                if child.poll() is not None:
                    raise RuntimeError(f"llama-server exited ({child.returncode}); see {log}")
                try:
                    if backend.request("/health", timeout=1).get("status") == "ok":
                        break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"llama-server did not become ready; see {log}")
            yield backend, psutil.Process(child.pid), time.perf_counter() - start
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
