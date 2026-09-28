"""Predict a single T1 image using the saved MNI model (Python 3.7)."""

import argparse
import os

import numpy as np
from scipy.special import expit

from mni_pipeline import VERSION, Reference, preprocess, read_path_image


def predict_features(model_path, features, feature_names):
    with np.load(model_path, allow_pickle=False) as model:
        if model["pipeline_version"].item() != VERSION:
            raise ValueError("Model/preprocessing version mismatch")
        if model["feature_names"].tolist() != list(feature_names):
            raise ValueError("Feature schema mismatch")
        selected = model["selected"].astype(int)
        standardized = (features - model["mean"]) / model["sd"]
        logit = float(standardized[selected].dot(model["coef"][:-1]) +
                      model["coef"][-1])
    p = float(expit(logit))
    return p, 1 if p >= 0.5 else 2


def predict_file(image_path, model_path="output_mni/model.npz"):
    reference = Reference()
    image = read_path_image(image_path, os.path.join("output_mni", "tmp"))
    result = preprocess(image, reference)
    p, label = predict_features(model_path, result["features"],
                                result["feature_names"])
    return p, label, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default=os.path.join("output_mni", "model.npz"))
    args = parser.parse_args()
    probability, label, _ = predict_file(args.image, args.model)
    print("p(ASD)=%.4f; label=%d (%s)" %
          (probability, label, "ASD" if label == 1 else "HC"))
