"""MNI152 registration, N4 correction and atlas-guided T1 tissue features.

Python 3.7 and SimpleITK 2.1.1.2. All operations are per-subject and do not
read labels. The three tissue probability maps are an atlas-guided estimate,
not a clinical segmentation or a replacement for FSL FAST/CAT12.
"""

import os
import shutil
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import numpy as np
import SimpleITK as sitk


VERSION = "mni-n4-affine-atlas-1"
# SimpleITK 2.1 on Windows may fail on non-ASCII absolute paths. Callers run
# from the project root and pass relative paths to SimpleITK file I/O.
RESOURCE_DIR = "resources"


class Reference(object):
    def __init__(self):
        def load(name):
            return sitk.ReadImage(os.path.join(RESOURCE_DIR, name))

        self.head = sitk.Cast(load("MNI152_T1_2mm.nii.gz"), sitk.sitkFloat32)
        self.brain = sitk.Cast(load("MNI152_T1_2mm_Brain.nii.gz"), sitk.sitkFloat32)
        self.mask_image = sitk.Cast(load("MNI152_T1_2mm_Brain_Mask.nii.gz") > 0.5,
                                    sitk.sitkUInt8)
        self.mask = sitk.GetArrayFromImage(self.mask_image).astype(bool)
        self.priors = np.stack([
            sitk.GetArrayFromImage(load("MNI152_T1_2mm_Brain_FAST_pve_%d.nii.gz" % i))
            for i in range(3)
        ]).astype(np.float32)
        self.priors = np.clip(self.priors, 0.0, 1.0)
        self.priors[:, ~self.mask] = 0.0
        self.voxel_mm3 = float(np.prod(self.head.GetSpacing()))


def read_zip_image(zip_path, member, temporary_dir):
    os.makedirs(temporary_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        compressed_image = archive.read(member)
    handle, path = tempfile.mkstemp(suffix=".nii.gz", dir=temporary_dir)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(compressed_image)
        image = sitk.ReadImage(os.path.relpath(path, os.getcwd()), sitk.sitkFloat32)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if image.GetDimension() != 3 or np.prod(image.GetSize()) < 1e6:
        raise ValueError("Expected complete 3D T1 image")
    return image


def read_path_image(source_path, temporary_dir):
    """Read a local NIfTI through an ASCII relative staging path."""
    os.makedirs(temporary_dir, exist_ok=True)
    handle, path = tempfile.mkstemp(suffix=".nii.gz", dir=temporary_dir)
    os.close(handle)
    try:
        shutil.copyfile(source_path, path)
        image = sitk.ReadImage(os.path.relpath(path, os.getcwd()), sitk.sitkFloat32)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return image


def n4_correction(image):
    small = sitk.Shrink(image, [2, 2, 2])
    mask = sitk.OtsuThreshold(small, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 15, 10])
    corrector.SetConvergenceThreshold(0.001)
    corrected = corrector.Execute(small, mask)
    return corrected


def affine_registration(moving, reference):
    initial = sitk.CenteredTransformInitializer(
        reference.head, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricFixedMask(reference.mask_image)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.04, seed=41)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.005, numberOfIterations=100,
        relaxationFactor=0.5)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=True)
    transform = registration.Execute(reference.head, moving)
    return transform, float(registration.GetMetricValue()), \
        int(registration.GetOptimizerIteration())


def _tissue_em(image, reference):
    mask = reference.mask
    values = image[mask].astype(np.float64)
    if values.size < 40000:
        raise ValueError("Too few MNI brain voxels")
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError("Non-finite registered image")
    lower, upper = np.percentile(values, (1, 99))
    values = np.clip(values, lower, upper)
    median = max(float(np.median(values[values > 0])), 1.0)
    values /= median
    priors = np.maximum(reference.priors[:, mask].T.astype(np.float64), 0.01)
    priors /= priors.sum(axis=1, keepdims=True)
    means = (priors * values[:, None]).sum(axis=0) / priors.sum(axis=0)
    variance = np.maximum((priors * (values[:, None] - means) ** 2).sum(axis=0) /
                          priors.sum(axis=0), 0.002)
    for _ in range(25):
        logp = np.log(priors) - 0.5 * np.log(2 * np.pi * variance) - \
            0.5 * (values[:, None] - means) ** 2 / variance
        logp -= logp.max(axis=1, keepdims=True)
        posterior = np.exp(logp)
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
        counts = np.maximum(posterior.sum(axis=0), 1.0)
        means = (posterior * values[:, None]).sum(axis=0) / counts
        variance = np.maximum((posterior * (values[:, None] - means) ** 2).sum(axis=0) /
                              counts, 0.001)
    if not (means[0] < means[1] < means[2]):
        raise ValueError("T1 tissue intensity order is implausible: " + str(means))
    probability = np.zeros((3,) + mask.shape, dtype=np.float32)
    for k in range(3):
        probability[k, mask] = posterior[:, k].astype(np.float32)
    return probability, means.tolist()


def _features(probability, reference, transform):
    mask = reference.mask
    shape = mask.shape  # SimpleITK NumPy order is z, y, x.
    center = np.asarray(reference.head.GetOrigin())
    center = center + 0.5 * np.asarray(reference.head.GetSize()) * \
        np.asarray(reference.head.GetSpacing())
    transformed_center = np.asarray(transform.TransformPoint(tuple(center)))
    basis = np.eye(3)
    matrix = np.stack([
        np.asarray(transform.TransformPoint(tuple(center + basis[i]))) - transformed_center
        for i in range(3)
    ], axis=1)
    jacobian = abs(float(np.linalg.det(matrix)))
    if jacobian < 0.5 or jacobian > 2.0:
        raise ValueError("Implausible affine volume scale %.3f" % jacobian)
    names = ["affine_jacobian", "estimated_brain_ml"]
    values = [jacobian, float(mask.sum() * reference.voxel_mm3 * jacobian / 1000.0)]
    for k, tissue in enumerate(("csf", "gm", "wm")):
        names.extend(("global_%s_fraction" % tissue, "global_%s_ml" % tissue))
        fraction = float(probability[k][mask].mean())
        values.extend((fraction, float(probability[k].sum() * reference.voxel_mm3 *
                                       jacobian / 1000.0)))
    cuts = [np.linspace(0, size, divisions + 1, dtype=int)
            for size, divisions in zip(shape, (3, 3, 2))]
    for iz in range(3):
        for iy in range(3):
            for ix in range(2):
                region = (slice(cuts[0][iz], cuts[0][iz + 1]),
                          slice(cuts[1][iy], cuts[1][iy + 1]),
                          slice(cuts[2][ix], cuts[2][ix + 1]))
                roi_mask = mask[region]
                denominator = max(int(roi_mask.sum()), 1)
                for k, tissue in enumerate(("csf", "gm", "wm")):
                    names.append("mni_%d_%d_%d_%s" % (iz, iy, ix, tissue))
                    values.append(float(probability[k][region].sum() / denominator))
    return np.asarray(values, dtype=np.float64), names, jacobian


def preprocess(image, reference):
    started = time.time()
    corrected = n4_correction(image)
    n4_seconds = round(time.time() - started, 2)
    transform, objective, iterations = affine_registration(corrected, reference)
    registered = sitk.Resample(corrected, reference.head, transform,
                               sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    registered_array = sitk.GetArrayFromImage(registered)
    brain_values = registered_array[reference.mask]
    nonzero_fraction = float(np.mean(brain_values > 0))
    if nonzero_fraction < 0.80:
        raise ValueError("MNI brain coverage too low: %.3f" % nonzero_fraction)
    probability, tissue_means = _tissue_em(registered_array, reference)
    features, names, jacobian = _features(probability, reference, transform)
    template_values = sitk.GetArrayFromImage(reference.brain)[reference.mask]
    correlation = float(np.corrcoef(brain_values, template_values)[0, 1])
    qc = {"pipeline_version": VERSION, "n4_seconds": n4_seconds,
          "registration_metric": objective, "registration_iterations": iterations,
          "brain_coverage": nonzero_fraction,
          "template_intensity_correlation": correlation,
          "affine_jacobian": jacobian, "tissue_means": tissue_means,
          "total_seconds": round(time.time() - started, 2)}
    qc["review_required"] = bool(nonzero_fraction < 0.95 or correlation < 0.30)
    return {"registered": registered, "probability": probability,
            "features": features, "feature_names": names, "qc": qc,
            "transform": transform}


def save_derivatives(result, output_dir, subject_id, reference, save_images=False):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, subject_id + "_features.npz"),
                        features=result["features"],
                        feature_names=np.asarray(result["feature_names"]))
    sitk.WriteTransform(result["transform"],
                        os.path.join(output_dir, subject_id + "_to_mni.tfm"))
    if save_images:
        sitk.WriteImage(result["registered"],
                        os.path.join(output_dir, subject_id + "_MNI_T1w.nii.gz"))
        for k, tissue in enumerate(("csf", "gm", "wm")):
            image = sitk.GetImageFromArray(result["probability"][k])
            image.CopyInformation(reference.head)
            sitk.WriteImage(image, os.path.join(output_dir,
                                               subject_id + "_MNI_" + tissue + ".nii.gz"))


def save_qc_png(result, reference, path, subject_id):
    config = os.path.join("output_mni", "matplotlib-cache")
    os.makedirs(config, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", config)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    template = sitk.GetArrayFromImage(reference.brain)
    moving = sitk.GetArrayFromImage(result["registered"])
    gm = result["probability"][1]
    mask = reference.mask
    positions = (45, 54, 45)

    def planes(data):
        return (data[:, :, positions[2]],
                data[:, positions[1], :],
                data[positions[0], :, :])

    fig = Figure(figsize=(11, 10))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 3)
    for row, data in enumerate((template, moving, gm)):
        for col, plane in enumerate(planes(data)):
            axis = axes[row, col]
            axis.imshow(np.rot90(plane), cmap="viridis" if row == 2 else "gray")
            if row == 1:
                axis.contour(np.rot90(planes(mask)[col]), levels=[0.5],
                             colors="red", linewidths=0.5)
            axis.axis("off")
            if row == 0:
                axis.set_title(("Sagittal", "Coronal", "Axial")[col])
        axes[row, 0].set_ylabel(("MNI template", "Registered T1", "GM probability")[row])
    fig.suptitle("%s | correlation %.3f | coverage %.3f" %
                 (subject_id, result["qc"]["template_intensity_correlation"],
                  result["qc"]["brain_coverage"]))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
