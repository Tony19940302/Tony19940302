"""Score trained ONNX classifiers on the same automatic replay candidates."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from battletracker.model import LABELS, clip_sequence, features_from_sequence, softmax
from battletracker.runtime import bootstrap_native_runtime
from evaluate import RESOURCE_TYPES, evaluate, read_jsonl
from predict_clean import ROOT, load_json

DESTINATION = ROOT / "outputs/phase-05"
MODELS = {
    "linear": "linear/model.onnx",
    "cnn": "temporal/temporal.onnx",
    "centroid": "temporal/centroid.onnx",
}


def working_set():
    if os.name != "nt":
        return None
    from ctypes.wintypes import DWORD, HANDLE

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", DWORD), ("faults", DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = HANDLE
    get_memory = kernel.K32GetProcessMemoryInfo
    get_memory.argtypes = [HANDLE, ctypes.POINTER(ProcessMemoryCounters), DWORD]
    info = ProcessMemoryCounters()
    info.cb = ctypes.sizeof(info)
    if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return info.working_set


def compare(name, model, predictions, clips, source, duration_ms, threshold, tolerance):
    bootstrap_native_runtime()
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    memory_before = working_set()
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    extraction_seconds = inference_seconds = 0.0
    observations = 0
    peak_sampled = working_set()
    adjusted = []
    for event in predictions:
        kind = event["event_type"]
        if kind not in RESOURCE_TYPES:
            adjusted.append(event)
            continue
        start = max(0, event["start_ms"] - 1100)
        end = min(duration_ms, event["end_ms"] + 500)
        started = perf_counter()
        clip = clip_sequence(source, start, end)
        model_input = clip if name == "cnn" else features_from_sequence(clip)
        extraction_seconds += perf_counter() - started
        started = perf_counter()
        output = session.run(None, {input_name: model_input[None, ...]})[0]
        probabilities = softmax(output)[0] if name == "cnn" else output[0]
        inference_seconds += perf_counter() - started
        label = LABELS[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        observations += 1
        row = dict(event, model_label=label, confidence=confidence,
                   model_id=name, model_window_ms=[start, end])
        if confidence < threshold or label != kind:
            row["event_type"] = "UNKNOWN"
        adjusted.append(row)
        snapshot = working_set()
        if snapshot is not None:
            peak_sampled = max(peak_sampled, snapshot)
    result = evaluate(adjusted, clips, tolerance)
    result.update(onnx_runtime=ort.__version__, candidate_count=observations,
                  preprocessing_seconds=extraction_seconds, inference_seconds=inference_seconds,
                  model_bytes=model.stat().st_size, source_duration_ms=duration_ms,
                  working_set_before_session_bytes=memory_before,
                  highest_sampled_working_set_bytes=peak_sampled)
    return adjusted, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    output = args.output
    config = load_json(ROOT / "configs/experiments/phase-05.json")
    summary = load_json(output / "baseline/visual_summary.json")
    predictions = read_jsonl(output / "baseline/predictions.jsonl")
    clips = read_jsonl(output / "validation_index.jsonl")
    asset = next(item for item in load_json(ROOT / "data/manifests/assets.json")["assets"]
                 if item["asset_id"] == summary["source_asset"])
    source = ROOT / asset["path"]
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (asset["dataset_split"] != "validation" or
            digest != asset["sha256"] or digest != summary["source_sha256"]):
        raise ValueError("comparison requires the registered held-out clean replay")
    tolerance = 500
    baseline = evaluate(predictions, clips, tolerance)
    model = output / MODELS[args.model]
    experiment_path = ("linear/experiment.json" if args.model == "linear"
                       else "temporal/experiment-temporal.json")
    experiment = load_json(output / experiment_path)
    index_digest = hashlib.sha256((output / "grouped_index.jsonl").read_bytes()).hexdigest()
    if (experiment["clip_index_sha256"] != index_digest
            or asset["battle_id"] not in experiment["validation_battles"]
            or asset["battle_id"] in experiment["train_battles"]):
        raise ValueError("model experiment and held-out dataset versions differ")
    adjusted, result = compare(args.model, model, predictions, clips, source,
                               summary["duration_ms"], config["minimum_confidence"], tolerance)
    payload = dict(baseline=baseline, enhanced=result, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   reference_scope="sparse reviewed windows", visual_summary=summary,
                   reporting_threshold=config["minimum_confidence"], clip_index_sha256=index_digest)
    (output / f"comparison-{args.model}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / f"predictions-{args.model}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adjusted), encoding="utf-8",
    )
    print(json.dumps({"model": args.model, "baseline": baseline["metrics"],
                      "enhanced": result["metrics"], "candidate_count": result["candidate_count"]}))


if __name__ == "__main__":
    main()
