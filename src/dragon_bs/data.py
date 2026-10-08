"""MATLAB dataset layouts, including the original MATLAB pixel ordering."""

from pathlib import Path

import numpy as np
import scipy.io as sio


DATASETS = {
    "indian_pines": ("data_indian_pines_drl.mat", "x", "flat_F", 200,
                     "indian_pines_GT.mat", "indian_pines_GT"),
    "pavia": ("paviaU.mat", "paviaU", "cube", 103, "paviaU_gt.mat", "paviaU_gt"),
    "gulfport": ("muufl_gulfport_campus_1_hsi_220_label.mat", "hsi", "muufl", 64, None, None),
    "houston": ("houston.mat", "houston", "cube", 144, "houston_gt_sum.mat", "houston_gt_sum"),
    "hanchuan": ("HanChuan/WHU_Hi_HanChuan.mat", "WHU_Hi_HanChuan", "cube", 274,
                 "HanChuan/WHU_Hi_HanChuan_gt.mat", "WHU_Hi_HanChuan_gt"),
    "honghu": ("HongHu/WHU_Hi_HongHu.mat", "WHU_Hi_HongHu", "cube", 270,
               "HongHu/WHU_Hi_HongHu_gt.mat", "WHU_Hi_HongHu_gt"),
    "botswana": ("data_botswana_drl.mat", "x", "flat_F", 145, "Botswana_gt.mat", "Botswana_gt"),
    "salinas": ("Salinas_corrected.mat", "salinas_corrected", "cube", 204,
                "Salinas_gt.mat", "salinas_gt"),
}


def load_dataset(name, data_dir="data4drl", gt_dir="data4classification", *, with_labels=False):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    filename, key, layout, expected, gt_filename, gt_key = DATASETS[name]
    path = Path(data_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}. See docs/DATASETS.md.")
    raw = sio.loadmat(path)[key]
    gt = None
    if layout == "muufl":
        struct = raw[0, 0]
        cube = np.asarray(struct["Data"], dtype=np.float32)
        x = cube.reshape(-1, cube.shape[-1])
        if with_labels:
            gt = np.asarray(struct["sceneLabels"][0, 0]["labels"])
            gt = np.where(gt < 0, 0, gt)
    else:
        raw = np.asarray(raw, dtype=np.float32)
        if layout == "cube" and raw.ndim != 3:
            raise ValueError(f"Expected a (rows, columns, bands) cube in {path}.")
        if layout == "flat_F" and raw.ndim != 2:
            raise ValueError(f"Expected a MATLAB-flattened (pixels, bands) array in {path}.")
        x = raw if layout == "flat_F" else raw.reshape(-1, raw.shape[-1])
        if with_labels:
            gt_base = Path(data_dir) if name in ("hanchuan", "honghu") else Path(gt_dir)
            gt_path = gt_base / gt_filename
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing ground truth: {gt_path}. See docs/DATASETS.md.")
            gt = np.asarray(sio.loadmat(gt_path)[gt_key])
    if x.shape[1] != expected:
        raise ValueError(f"{name} expects {expected} bands in this input layout; got {x.shape[1]}.")
    result = {"x": x, "name": name, "nb_bands": x.shape[1], "source": str(path.resolve())}
    if with_labels:
        y = gt.ravel(order="F" if layout == "flat_F" else "C").astype(np.int64)
        if len(y) != len(x):
            raise ValueError(f"{name}: {len(x)} spectra but {len(y)} labels.")
        result["y"] = y
    return result
