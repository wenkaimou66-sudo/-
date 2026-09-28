"""Python 3.7 T1 MRI preprocessing and label-free feature extraction.

The common grid is a cohort coordinate system, not an MNI atlas. Registration
uses the NIfTI affine followed by an image-mask moment affine alignment.
"""

import gzip
import json
import os
import struct
import zipfile

import numpy as np
from scipy import ndimage


GRID_SHAPE = (56, 72, 72)
GRID_SPACING = np.array((3.0, 3.0, 3.0), dtype=np.float64)
GRID_ORIGIN = np.array((-84.0, -108.0, -108.0), dtype=np.float64)
GRID_CENTER = np.array((27.5, 35.5, 39.0), dtype=np.float64)
TARGET_SD = np.array((10.5, 12.5, 11.5), dtype=np.float64)
PIPELINE_VERSION = "1.2"


def _decode_nifti(raw):
    if len(raw) < 352 or struct.unpack_from("<i", raw, 0)[0] != 348:
        raise ValueError("Expected little-endian NIfTI-1 header")
    dims = struct.unpack_from("<8h", raw, 40)
    datatype = struct.unpack_from("<h", raw, 70)[0]
    bitpix = struct.unpack_from("<h", raw, 72)[0]
    offset = int(struct.unpack_from("<f", raw, 108)[0])
    supported = {(2, 8): ("u1", 1), (4, 16): ("<i2", 2),
                 (16, 32): ("<f4", 4)}
    if dims[0] != 3 or (datatype, bitpix) not in supported:
        raise ValueError("Expected a supported 3D NIfTI image")
    shape = tuple(int(v) for v in dims[1:4])
    if any(v <= 0 for v in shape):
        raise ValueError("Invalid image dimensions")
    count = int(np.prod(shape))
    dtype, bytes_per_voxel = supported[(datatype, bitpix)]
    if offset < 352 or len(raw) < offset + bytes_per_voxel * count:
        raise ValueError("Truncated NIfTI image")
    if struct.unpack_from("<h", raw, 254)[0] <= 0:
        raise ValueError("NIfTI sform is required for spatial alignment")
    affine = np.eye(4, dtype=np.float64)
    for row in range(3):
        affine[row, :] = struct.unpack_from("<4f", raw, 280 + 16 * row)
    if abs(np.linalg.det(affine[:3, :3])) < 1e-6:
        raise ValueError("Singular NIfTI affine")
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    return data.reshape(shape, order="F"), affine


def read_image_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, "r") as archive:
        raw_gzip = archive.read(member)
    return _decode_nifti(gzip.decompress(raw_gzip))


def read_image_file(path):
    with gzip.open(path, "rb") as source:
        return _decode_nifti(source.read())


def _resample_physical(image, affine):
    inverse = np.linalg.inv(affine)
    matrix = inverse[:3, :3].dot(np.diag(GRID_SPACING))
    offset = inverse[:3, :3].dot(GRID_ORIGIN - affine[:3, 3])
    return ndimage.affine_transform(
        image, matrix, offset=offset, output_shape=GRID_SHAPE,
        order=1, mode="constant", cval=0.0, prefilter=False
    ).astype(np.float32)


def _largest_component(mask):
    labeled, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("No foreground component found")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _anatomical_prior():
    x, y, z = np.ogrid[:GRID_SHAPE[0], :GRID_SHAPE[1], :GRID_SHAPE[2]]
    px = GRID_ORIGIN[0] + GRID_SPACING[0] * x
    py = GRID_ORIGIN[1] + GRID_SPACING[1] * y
    pz = GRID_ORIGIN[2] + GRID_SPACING[2] * z
    return ((px / 68.0) ** 2 + ((py + 11.0) / 82.0) ** 2 +
            ((pz - 17.0) / 72.0) ** 2) <= 1.0


def _initial_mask(image):
    positive = image[image > 0]
    if positive.size < 20000:
        raise ValueError("Image has too little nonzero data")
    threshold = max(3.0, float(np.percentile(positive, 17)))
    mask = (ndimage.gaussian_filter(image, 0.8) > threshold) & _anatomical_prior()
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = _largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    if mask.sum() < 15000 or mask.sum() > 210000:
        raise ValueError("Implausible brain-region mask volume")
    return mask


def _correct_bias(image, mask):
    weighted = ndimage.gaussian_filter(image * mask, sigma=15.0)
    weight = ndimage.gaussian_filter(mask.astype(np.float32), sigma=15.0)
    field = weighted / np.maximum(weight, 0.05)
    reference = float(np.median(field[mask]))
    corrected = image / np.clip(field / max(reference, 1.0), 0.4, 2.5)
    corrected[~mask] = 0.0
    upper = float(np.percentile(corrected[mask], 99.5))
    corrected = np.clip(corrected / max(upper, 1.0), 0.0, 1.5)
    return corrected.astype(np.float32)


def _register_by_moments(image, mask):
    center = np.array(ndimage.center_of_mass(mask), dtype=np.float64)
    coords = np.argwhere(mask)
    sd = np.maximum(coords.std(axis=0), 1.0)
    scale = np.clip(sd / TARGET_SD, 0.70, 1.30)
    matrix = np.diag(scale)
    offset = center - scale * GRID_CENTER
    aligned = ndimage.affine_transform(
        image, matrix, offset=offset, output_shape=GRID_SHAPE,
        order=1, mode="constant", cval=0.0, prefilter=False
    ).astype(np.float32)
    aligned_mask = ndimage.affine_transform(
        mask.astype(np.uint8), matrix, offset=offset,
        output_shape=GRID_SHAPE, order=0, mode="constant", cval=0
    ).astype(bool)
    return aligned, aligned_mask, center, scale


def _tissue_posteriors(image, mask):
    """Three-intensity Gaussian mixture; component order CSF, GM, WM.

    These are approximate intensity classes, not validated anatomical tissue
    maps. The QC and report deliberately use that wording.
    """
    values = image[mask].astype(np.float64)
    if values.size < 10000:
        raise ValueError("Insufficient brain-region voxels")
    values = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
    sample = values[::max(1, values.size // 40000)]
    means = np.percentile(sample, (18, 50, 82)).astype(np.float64)
    variances = np.full(3, max(float(np.var(sample)) / 7.0, 0.001))
    weights = np.full(3, 1.0 / 3.0)
    for _ in range(40):
        logp = []
        for k in range(3):
            logp.append(np.log(max(weights[k], 1e-5)) -
                        0.5 * np.log(2 * np.pi * variances[k]) -
                        0.5 * (sample - means[k]) ** 2 / variances[k])
        logp = np.stack(logp, axis=1)
        logp -= logp.max(axis=1, keepdims=True)
        post = np.exp(logp)
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
        count = np.maximum(post.sum(axis=0), 1e-5)
        means = (post * sample[:, None]).sum(axis=0) / count
        variances = np.maximum((post * (sample[:, None] - means) ** 2).sum(axis=0) /
                               count, 0.0005)
        weights = count / count.sum()
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    x = image[mask].astype(np.float64)
    logp = np.stack([
        np.log(max(weights[k], 1e-5)) - 0.5 * np.log(2 * np.pi * variances[k]) -
        0.5 * (x - means[k]) ** 2 / variances[k] for k in range(3)
    ], axis=1)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    result = np.zeros((3,) + GRID_SHAPE, dtype=np.float32)
    for k in range(3):
        result[k][mask] = p[:, k].astype(np.float32)
    return result, means


def _features(probabilities, mask, source_volume_ml):
    names = ["region_volume_ml"]
    values = [float(source_volume_ml)]
    count = max(int(mask.sum()), 1)
    for k, name in enumerate(("csf_like", "gm_like", "wm_like")):
        names.append("global_" + name)
        values.append(float(probabilities[k].sum() / count))
    cuts = [np.linspace(0, size, n + 1, dtype=int)
            for size, n in zip(GRID_SHAPE, (2, 3, 3))]
    for ix in range(2):
        for iy in range(3):
            for iz in range(3):
                sl = (slice(cuts[0][ix], cuts[0][ix + 1]),
                      slice(cuts[1][iy], cuts[1][iy + 1]),
                      slice(cuts[2][iz], cuts[2][iz + 1]))
                region = mask[sl]
                n = max(int(region.sum()), 1)
                prefix = "roi_%d_%d_%d_" % (ix, iy, iz)
                names.append(prefix + "occupancy")
                values.append(float(region.mean()))
                for k, tissue in enumerate(("csf", "gm", "wm")):
                    names.append(prefix + tissue)
                    values.append(float(probabilities[k][sl].sum() / n))
    return np.asarray(values, dtype=np.float64), names


def preprocess(image, affine):
    if image.ndim != 3 or image.size < 100000:
        raise ValueError("Expected a full 3D T1 image")
    physical = _resample_physical(image.astype(np.float32), affine)
    rough_mask = _initial_mask(physical)
    corrected = _correct_bias(physical, rough_mask)
    aligned, mask, center, scale = _register_by_moments(corrected, rough_mask)
    probabilities, means = _tissue_posteriors(aligned, mask)
    source_volume_ml = float(rough_mask.sum() * np.prod(GRID_SPACING) / 1000.0)
    features, names = _features(probabilities, mask, source_volume_ml)
    qc = {
        "pipeline_version": PIPELINE_VERSION,
        "source_mask_ml": source_volume_ml,
        "registered_mask_voxels": int(mask.sum()),
        "source_center_grid": center.tolist(),
        "source_spread_grid": np.argwhere(rough_mask).std(axis=0).tolist(),
        "registration_scale": scale.tolist(),
        "tissue_means": means.tolist(),
        "max_input_intensity": int(np.max(image)),
        "warning": "Brain mask and tissue classes are approximate; inspect QC overlays."
    }
    return {"image": aligned, "mask": mask, "probabilities": probabilities,
            "features": features, "feature_names": names, "qc": qc}


def save_nifti(path, data):
    arr = np.asarray(data)
    if tuple(arr.shape) != GRID_SHAPE:
        raise ValueError("Output image has wrong grid dimensions")
    if arr.dtype == np.uint8:
        datatype, bitpix = 2, 8
    else:
        arr = arr.astype("<f4")
        datatype, bitpix = 16, 32
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *GRID_SHAPE, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, 1.0, *GRID_SPACING, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[123] = 2  # millimetres
    struct.pack_into("<h", header, 254, 1)
    for row in range(3):
        axis = [0.0, 0.0, 0.0, float(GRID_ORIGIN[row])]
        axis[row] = float(GRID_SPACING[row])
        struct.pack_into("<4f", header, 280 + row * 16, *axis)
    header[344:348] = b"n+1\x00"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as output:
        output.write(header)
        output.write(np.asfortranarray(arr).tobytes(order="F"))


def save_qc_png(path, result, title):
    mpl_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "matplotlib-cache")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = result["image"]
    mask = result["mask"]
    gm = result["probabilities"][1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    slices = [(image[28, :, :], mask[28, :, :], gm[28, :, :]),
              (image[:, 36, :], mask[:, 36, :], gm[:, 36, :]),
              (image[:, :, 39], mask[:, :, 39], gm[:, :, 39])]
    for col, (base, region, tissue) in enumerate(slices):
        axes[0, col].imshow(np.rot90(base), cmap="gray", vmin=0, vmax=1.1)
        axes[0, col].contour(np.rot90(region), levels=[0.5], colors="r", linewidths=0.5)
        axes[1, col].imshow(np.rot90(tissue), cmap="viridis", vmin=0, vmax=1)
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(title + " | approximate mask and GM-like probability")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_result(output_dir, subject_id, result, save_images=False):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, subject_id + "_features.npz"),
                        features=result["features"],
                        feature_names=np.asarray(result["feature_names"]))
    with open(os.path.join(output_dir, subject_id + "_qc.json"), "w", encoding="utf-8") as out:
        json.dump(result["qc"], out, ensure_ascii=False, indent=2)
    if save_images:
        save_nifti(os.path.join(output_dir, subject_id + "_corrected.nii.gz"), result["image"])
        save_nifti(os.path.join(output_dir, subject_id + "_mask.nii.gz"),
                   result["mask"].astype(np.uint8))
        classes = np.argmax(result["probabilities"], axis=0).astype(np.uint8) + 1
        classes[~result["mask"]] = 0
        save_nifti(os.path.join(output_dir, subject_id + "_tissue.nii.gz"), classes)
