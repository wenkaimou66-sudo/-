"""Validate final MNI submission and its reproducibility under Python 3.7."""

import csv
import json
import os
import sys

import numpy as np

import gui  # Verify desktop software imports with this interpreter.
from mni_pipeline import VERSION
from mni_predict_one import predict_features
from run_experiment import load_manifest, metrics


def csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        names = list(reader.fieldnames)
        rows = list(reader)
    return names, rows


def main():
    assert sys.version_info[:2] == (3, 7)
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    output = "output_mni"
    train, test = load_manifest(os.path.join("..", "excise.zip"))
    fields, submission = csv_rows(os.path.join(output, "submission.csv"))
    assert fields == ["subject_id", "label"]
    assert len(submission) == 10
    assert [row["subject_id"] for row in submission] == [sid for sid, _ in test]
    assert sum(row["label"] == "1" for row in submission) == 5
    assert sum(row["label"] == "2" for row in submission) == 5
    with open(os.path.join(output, "submission.csv"), "rb") as source:
        assert source.read(3) != b"\xef\xbb\xbf", "Submission must not contain a BOM"
    _, predictions = csv_rows(os.path.join(output, "predictions.csv"))
    assert len(predictions) == 10
    for row in predictions:
        sid = row["subject_id"]
        with np.load(os.path.join(output, "features", sid + "_features.npz"),
                     allow_pickle=False) as cached:
            p, _ = predict_features(os.path.join(output, "model.npz"),
                                    cached["features"],
                                    cached["feature_names"].tolist())
        assert abs(p - float(row["p_asd"])) < 1e-10
    _, oof = csv_rows(os.path.join(output, "out_of_fold_predictions.csv"))
    assert len(oof) == len(train)
    y = np.asarray([row["true_label"] == "1" for row in oof], dtype=int)
    p = np.asarray([float(row["p_asd"]) for row in oof])
    with open(os.path.join(output, "metrics.json"), "r", encoding="utf-8") as source:
        reported = json.load(source)
    assert reported["pipeline_version"] == VERSION
    actual = metrics(y, p)
    assert all(abs(actual[key] - value) < 1e-12
               for key, value in reported["nested_cv"].items())
    with open(os.path.join(output, "quality_control.json"), "r", encoding="utf-8") as source:
        qc = json.load(source)
    assert len(qc) == 146
    assert all(row["pipeline_version"] == VERSION for row in qc)
    assert os.path.isfile(os.path.join(output, "report.html"))
    print("OK: Python 3.7, GUI import, MNI QC, model, CV metrics and exact 10-row CSV")
    print("QC review cases:", sum(bool(row.get("review_required")) for row in qc))


if __name__ == "__main__":
    main()
