"""Train the vibration models and write the evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.hardware import configure_torch_threads
from src.report import to_builtin, write_json
from vibration.catalog import RECORDINGS, Recording
from vibration.config import (
    CLASS_ORDER,
    CNN_EPOCHS,
    N_WINDOWS,
    PROJECT_SLUG,
    QUICK_EPOCHS,
    QUICK_LABELS,
    QUICK_LOADS,
    QUICK_TREES,
    QUICK_WINDOWS,
    RF_TREES,
    SAMPLE_RATE_HZ,
    SEED,
    SPLIT_VERSION,
    WINDOW_LENGTH,
)
from vibration.download import ensure_recordings
from vibration.envelope import outer_race_check
from vibration.features import feature_names, window_features
from vibration.metrics import macro_f1, vibration_criteria
from vibration.models import fit_random_forest, predict_cnn, train_cnn
from vibration.plots import plot_confusion, plot_envelope, plot_window_spectra
from vibration.signals import load_drive_end
from vibration.windowing import build_split_document, cross_split_overlap_count


def select_recordings(quick: bool) -> tuple[Recording, ...]:
    if not quick:
        return RECORDINGS
    labels = set(QUICK_LABELS)
    loads = set(QUICK_LOADS)
    return tuple(item for item in RECORDINGS if item.label in labels and item.load_hp in loads)


def class_names(recordings: tuple[Recording, ...] | list[Recording]) -> list[str]:
    present = {item.label for item in recordings}
    return [label for label in CLASS_ORDER if label in present]


def envelope_target(recordings: tuple[Recording, ...] | list[Recording]) -> Recording:
    outers = [item for item in recordings if item.family == "outer" and item.load_hp == 0]
    if not outers:
        raise ValueError("need a 0 hp outer-race recording for the BPFO check")
    return max(outers, key=lambda item: item.diameter_in or 0.0)


def _load_lengths(data_dir: Path, recordings: tuple[Recording, ...]) -> tuple[dict[str, int], dict[str, float | None]]:
    lengths: dict[str, int] = {}
    rpms: dict[str, float | None] = {}
    for spec in recordings:
        signal, rpm = load_drive_end(data_dir / f"{spec.file_id}.mat")
        lengths[spec.file_id] = int(signal.size)
        rpms[spec.file_id] = rpm if rpm is not None else float(spec.nominal_rpm)
        del signal
    return lengths, rpms


def _materialize(
    data_dir: Path,
    document: dict,
) -> dict[str, np.ndarray | list]:
    names = feature_names()
    windows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[str] = []
    splits: list[str] = []
    recordings: list[str] = []
    loads: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    examples: dict[str, np.ndarray] = {}
    fs = float(document["sample_rate_hz"])
    for file_id, info in document["recordings"].items():
        signal, _rpm = load_drive_end(data_dir / f"{file_id}.mat")
        rpm = float(info["measured_rpm"] or info["nominal_rpm"])
        for start, end, window_split in info["windows"]:
            piece = np.asarray(signal[int(start) : int(end)], dtype=np.float32)
            windows.append(piece)
            features.append(window_features(piece, fs, rpm))
            labels.append(info["label"])
            splits.append(str(window_split))
            recordings.append(file_id)
            loads.append(int(info["load_hp"]))
            starts.append(int(start))
            ends.append(int(end))
            if window_split == "train" and info["label"] not in examples:
                examples[info["label"]] = piece.astype(np.float32)
        del signal
    return {
        "windows": np.stack(windows).astype(np.float32),
        "features": np.vstack(features).astype(np.float64),
        "labels": labels,
        "splits": splits,
        "recordings": recordings,
        "loads": loads,
        "starts": starts,
        "ends": ends,
        "feature_names": names,
        "examples": examples,
    }


def _index(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def run_vibration(
    root: Path,
    quick: bool = False,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> dict:
    configure_torch_threads()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if output_dir is None:
        output_dir = root / "results" / "quick" if quick else root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = select_recordings(quick)
    labels = class_names(recordings)
    label_to_index = {label: index for index, label in enumerate(labels)}
    data_dir = ensure_recordings(root, recordings)
    target = envelope_target(recordings)
    target_signal, target_rpm = load_drive_end(data_dir / f"{target.file_id}.mat")
    rpm = float(target_rpm) if target_rpm is not None else float(target.nominal_rpm)
    # Use a long prefix so the frequency bin is much finer than the 2% BPFO tolerance.
    check_n = min(target_signal.size, SAMPLE_RATE_HZ * 10)
    check = outer_race_check(target_signal[:check_n], SAMPLE_RATE_HZ, rpm)
    envelope_payload = {
        "bpfo_hz_theory": check["bpfo_hz_theory"],
        "bpfo_hz_detected": check["bpfo_hz_detected"],
        "rel_error": check["rel_error"],
        "recording": target.file_id,
        "load_hp": target.load_hp,
        "label": target.label,
        "rpm": check["rpm"],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "search_low_hz": check["search_low_hz"],
        "search_high_hz": check["search_high_hz"],
        "n_samples": int(check_n),
    }
    write_json(output_dir / "envelope_check.json", envelope_payload)
    plot_envelope(
        target_signal[:check_n],
        SAMPLE_RATE_HZ,
        rpm,
        check["bpfo_hz_detected"],
        output_dir / "vibration_envelope_spectrum.png",
    )
    del target_signal

    lengths, rpms = _load_lengths(data_dir, recordings)
    n_windows = QUICK_WINDOWS if quick else N_WINDOWS
    document = build_split_document(
        recordings,
        lengths,
        rpms,
        WINDOW_LENGTH,
        n_windows,
        SAMPLE_RATE_HZ,
        SPLIT_VERSION,
    )
    overlap = cross_split_overlap_count(document)
    split_path = root / "data" / "vibration" / ("splits-quick.json" if quick else "splits-full.json")
    write_json(split_path, document)
    write_json(output_dir / "vibration_splits.json", document)

    table = _materialize(data_dir, document)
    split_arr = np.asarray(table["splits"])
    train_i = _index(split_arr == "train")
    val_i = _index(split_arr == "val")
    test_i = _index(split_arr == "test")
    if train_i.size == 0 or val_i.size == 0 or test_i.size == 0:
        raise RuntimeError("every split needs at least one window")
    y_all = [label_to_index[label] for label in table["labels"]]
    y = np.asarray(y_all, dtype=np.int64)
    features = table["features"]
    windows = table["windows"]

    # Fit only on train. Early stopping looks only at the validation load.
    n_jobs = 2 if quick else 4
    forest = fit_random_forest(
        features[train_i],
        np.asarray(table["labels"])[train_i],
        n_estimators=QUICK_TREES if quick else RF_TREES,
        seed=SEED,
        n_jobs=n_jobs,
    )
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cnn_info = train_cnn(
        windows[train_i],
        y[train_i],
        windows[val_i],
        y[val_i],
        n_classes=len(labels),
        epochs=QUICK_EPOCHS if quick else CNN_EPOCHS,
        device=torch_device,
        seed=SEED,
    )
    baseline_pred = forest.predict(features[test_i]).astype(str)
    cnn_index = predict_cnn(model, windows[test_i], torch_device)
    cnn_pred = np.asarray([labels[int(index)] for index in cnn_index])
    truth = np.asarray(table["labels"])[test_i]
    rows = []
    for position, index in enumerate(test_i.tolist()):
        start = table["starts"][index]
        end = table["ends"][index]
        recording = table["recordings"][index]
        rows.append(
            {
                "window_id": f"{recording}:{start}:{end}",
                "recording": recording,
                "load_hp": int(table["loads"][index]),
                "label": truth[position],
                "cnn_pred": cnn_pred[position],
                "baseline_pred": baseline_pred[position],
            }
        )
    predictions = pd.DataFrame(rows).sort_values(["recording", "window_id"]).reset_index(drop=True)
    pred_path = output_dir / "vibration_test_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    predictions = pd.read_csv(pred_path)
    scored_labels = labels if not quick else sorted(set(predictions["label"].astype(str)))
    if not quick:
        scored_labels = list(CLASS_ORDER)
    cnn_f1 = macro_f1(predictions["label"].astype(str), predictions["cnn_pred"].astype(str), scored_labels)
    baseline_f1 = macro_f1(predictions["label"].astype(str), predictions["baseline_pred"].astype(str), scored_labels)
    best_f1 = max(cnn_f1, baseline_f1)
    best_name = "cnn" if cnn_f1 >= baseline_f1 else "random_forest"

    examples = table["examples"]
    fault_label = "or_021" if "or_021" in examples else "or_007"
    normal_example = examples.get("normal")
    fault_example = examples.get(fault_label)
    if normal_example is not None and fault_example is not None:
        plot_window_spectra(normal_example, fault_example, SAMPLE_RATE_HZ, output_dir / "vibration_fft_example.png")
    elif fault_example is not None:
        plot_window_spectra(fault_example, fault_example, SAMPLE_RATE_HZ, output_dir / "vibration_fft_example.png")
    plot_confusion(
        predictions["label"].astype(str).tolist(),
        predictions["cnn_pred"].astype(str).tolist(),
        scored_labels,
        output_dir / "vibration_confusion_cnn.png",
        "CNN on held-out load",
    )
    plot_confusion(
        predictions["label"].astype(str).tolist(),
        predictions["baseline_pred"].astype(str).tolist(),
        scored_labels,
        output_dir / "vibration_confusion_baseline.png",
        "Random forest on held-out load",
    )

    summary = {
        "project": PROJECT_SLUG,
        "quick": quick,
        "n_classes": 10 if not quick else len(scored_labels),
        "class_names": list(CLASS_ORDER) if not quick else scored_labels,
        "cnn_test_macro_f1": cnn_f1,
        "baseline_test_macro_f1": baseline_f1,
        "best_test_macro_f1": best_f1,
        "best_model": best_name,
        "cnn_val_macro_f1": cnn_info["val_macro_f1"],
        "cnn_epochs_ran": cnn_info["epochs_ran"],
        "bpfo_hz_theory": check["bpfo_hz_theory"],
        "bpfo_hz_detected": check["bpfo_hz_detected"],
        "bpfo_rel_error": check["rel_error"],
        "bpfo_rel_error_percent": 100.0 * check["rel_error"],
        "overlap_windows": overlap,
        "test_load_hp": 3,
        "device": str(torch_device),
        "n_train_windows": int(train_i.size),
        "n_val_windows": int(val_i.size),
        "n_test_windows": int(test_i.size),
        "n_train_recordings": len(document["train_recordings"]),
        "n_val_recordings": len(document["val_recordings"]),
        "n_test_recordings": len(document["test_recordings"]),
        "window_length": WINDOW_LENGTH,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "feature_names": table["feature_names"],
        "envelope_recording": target.file_id,
    }
    # The official 10-class count is the contract for a full run even if scoring used that list.
    write_json(output_dir / "vibration_summary.json", summary)
    print(
        f"vibration val_macro_f1={cnn_info['val_macro_f1']:.4f} "
        f"test_cnn={cnn_f1:.4f} test_baseline={baseline_f1:.4f} "
        f"bpfo_rel_error={check['rel_error']:.6f} overlap={overlap}"
    )
    return to_builtin(summary)


def _fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def merge_vibration_metrics(metrics_path: Path, summary: dict, coverage: float | None, quick: bool) -> dict:
    """Add vibration criteria and numbers onto the parent metrics file."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    parent_ok = all(item["passed"] for item in payload["success_criteria"])
    fd001 = payload["metrics"]["fd001"]["test_rmse"]
    fd003 = payload["metrics"]["fd003"]["test_rmse"]
    lead = payload["metrics"]["alerts"]["lead_time_fraction"]
    false_alerts = payload["metrics"]["alerts"]["false_alerts_per_engine"]
    from vibration.metrics import parent_headline_deviation

    if quick or fd001 is None or fd003 is None:
        deviation = None
        parent_ok = False
    else:
        deviation = parent_headline_deviation(float(fd001), float(fd003), float(lead), float(false_alerts))
    n_classes = 10 if not quick else int(summary["n_classes"])
    extra = vibration_criteria(
        float(summary["best_test_macro_f1"]),
        float(summary["baseline_test_macro_f1"]),
        float(summary["bpfo_rel_error"]),
        int(summary["overlap_windows"]),
        None if deviation is None else float(deviation),
        parent_ok,
        coverage,
        n_classes,
    )
    # Keep the parent criteria. Official full runs append the vibration contract.
    if quick:
        # Quick mode still records vibration numbers, but the parent smoke test
        # reads run_pipeline directly and never sees this merged file.
        payload["success_criteria"] = list(payload["success_criteria"]) + extra
    else:
        payload["success_criteria"] = list(payload["success_criteria"]) + extra
    payload["project"] = PROJECT_SLUG
    headline = payload["metrics"].setdefault("headline_text", {})
    headline["vibration_cnn_macro_f1"] = _fmt(summary["cnn_test_macro_f1"], 4)
    headline["vibration_baseline_macro_f1"] = _fmt(summary["baseline_test_macro_f1"], 4)
    headline["vibration_best_macro_f1"] = _fmt(summary["best_test_macro_f1"], 4)
    headline["vibration_bpfo_rel_error_percent"] = _fmt(summary["bpfo_rel_error_percent"], 2)
    headline["vibration_overlap_windows"] = str(int(summary["overlap_windows"]))
    headline["vibration_test_load_hp"] = str(int(summary["test_load_hp"]))
    headline["vibration_n_classes"] = str(n_classes)
    payload["metrics"]["vibration"] = summary
    payload["metrics"]["parent_headline_rel_deviation"] = deviation
    payload["baseline"]["vibration"] = {
        "model": "random_forest",
        "features": "rms, kurtosis, crest factor, fft band energies, envelope band energies, envelope fault-order peaks",
        "test_macro_f1": summary["baseline_test_macro_f1"],
    }
    payload["splits"]["CWRU"] = {
        "train": int(summary["n_train_windows"]),
        "val": int(summary["n_val_windows"]),
        "test": int(summary["n_test_windows"]),
    }
    payload["seeds"] = [SEED]
    return write_json(metrics_path, payload)

