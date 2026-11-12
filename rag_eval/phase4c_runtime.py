"""Standalone, standard-library checkpoint protocol used by the Colab notebook."""

import hashlib
import json
import math
from pathlib import Path


STAGES = ("sanity", "original", "confirmation")
RUNTIME_ID = "llama-cpp-python-0.3.16-cu124-wheel-v1"
WHEEL_BASE = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16-cu124/"
WHEELS = {
    "cp311": ("llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl",
              "a455263e7608a51c26079e82fe524f9b7f1f73ee9fc563221a297de8da2a8c51", 551261692),
    "cp312": ("llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl",
              "431fd9e68ff699bfa9ae6a01b05a0b6ddc0de97aa63c503e84141ae408502a11", 551261698),
}


def select_wheel(system, machine, python_version):
    abi = f"cp{python_version[0]}{python_version[1]}"
    if system != "Linux" or machine != "x86_64" or abi not in WHEELS:
        raise ValueError("Use a free Linux x86_64 Colab GPU runtime with Python 3.11 or 3.12")
    return WHEELS[abi]


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def render(chat):
    return "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in chat) + (
        "<|im_start|>assistant\n")


def validate_packet(packet):
    payload = packet["payload"]
    if digest(payload) != packet["sha256"] or payload["config"]["protocol"] != "phase4c-v1":
        raise ValueError("Packet fingerprint or protocol mismatch")
    cases = payload["cases"]
    if (len({case["id"] for case in cases}) != len(cases)
            or [case["stage"] for case in cases] != sorted(
                (case["stage"] for case in cases), key=STAGES.index)
            or {stage: sum(c["stage"] == stage for c in cases) for stage in STAGES}
            != {"sanity": 3, "original": 8, "confirmation": 15}):
        raise ValueError("Invalid case manifest")
    return payload


def validate_checkpoint(packet, checkpoint):
    payload = validate_packet(packet)
    if checkpoint["packet_sha256"] != packet["sha256"]:
        raise ValueError("Checkpoint belongs to a different packet")
    cases = payload["cases"]
    rows = checkpoint["rows"]
    config = payload["config"]
    environment = None
    for session in checkpoint["sessions"]:
        if session.get("runtime_id") != RUNTIME_ID:
            raise ValueError("Use a separate checkpoint for this binary runtime")
        wheel = session["wheel"]
        gpu = session["gpu_before"]
        probe = session["gpu_probe"]
        if ((wheel["name"], wheel["sha256"], wheel["bytes"]) not in WHEELS.values()
                or session["model_revision"] != config["revision"] or session["weight_shards"] != config["shards"]
                or session["versions"]["llama-cpp-python"] != config["backend_version"]
                or "T4" not in gpu["name"] or gpu["compute_capability"] != "7.5"
                or gpu["free_mib"] < config["minimum_free_gpu_mib"]
                or session["cuda_driver_api"] < 12040
                or probe["process_used_mib"] < 3072 or probe["tokens_evaluated"] < 1
                or session["settings"] != {"n_gpu_layers": -1, "n_batch": 256, "n_ctx": config["context_window"]}):
            raise ValueError("Runtime session differs from the frozen configuration")
        key = (wheel["sha256"], gpu["name"], gpu["driver"], session["cuda_driver_api"], session["versions"])
        if environment is not None and key != environment:
            raise ValueError("Do not combine records from different runtime environments")
        environment = key
    if len(rows) > len(cases):
        raise ValueError("Too many result records")
    for case, wrapped in zip(cases, rows):
        row = wrapped["record"]
        if (digest(row) != wrapped["sha256"] or row["id"] != case["id"]
                or row["stage"] != case["stage"] or row["prompt_sha256"] != digest(render(case["chat"]))):
            raise ValueError(f"Invalid result record for {case['id']}")
        if not 0 < row["prompt_tokens"] <= payload["config"]["context_window"] - payload["config"]["max_output"]:
            raise ValueError("Invalid prompt token count")
        if not 0 <= row["output_tokens"] <= payload["config"]["max_output"]:
            raise ValueError("Invalid output token count")
        if not math.isfinite(row["latency_s"]) or row["latency_s"] <= 0:
            raise ValueError("Invalid measured latency")
        if type(row["session"]) is not int or not 0 <= row["session"] < len(checkpoint["sessions"]):
            raise ValueError("Invalid result session")
    for stage in ("original", "confirmation"):
        if any(row["record"]["stage"] == stage for row in rows) and stage not in checkpoint["approvals"]:
            raise ValueError(f"Missing manual stage approval: {stage}")
    return rows


def append_result(packet, checkpoint, path, row):
    candidate = checkpoint | {"rows": [*checkpoint["rows"], {"record": row, "sha256": digest(row)}]}
    validate_checkpoint(packet, candidate)
    save(path, candidate)
    checkpoint["rows"] = candidate["rows"]
