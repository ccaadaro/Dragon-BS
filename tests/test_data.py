import numpy as np
import scipy.io as sio

from dragon_bs.data import load_dataset


def test_matlab_flattened_spectra_align_with_fortran_labels(tmp_path):
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    x = np.repeat(gt.ravel(order="F")[:, None], 200, axis=1)
    sio.savemat(tmp_path / "data_indian_pines_drl.mat", {"x": x})
    sio.savemat(tmp_path / "indian_pines_GT.mat", {"indian_pines_GT": gt})
    d = load_dataset("indian_pines", tmp_path, tmp_path, with_labels=True)
    np.testing.assert_array_equal(d["x"][:, 0], d["y"])


def test_cube_labels_align_in_c_order(tmp_path):
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    cube = np.repeat(gt[:, :, None], 103, axis=2)
    sio.savemat(tmp_path / "paviaU.mat", {"paviaU": cube})
    sio.savemat(tmp_path / "paviaU_gt.mat", {"paviaU_gt": gt})
    d = load_dataset("pavia", tmp_path, tmp_path, with_labels=True)
    np.testing.assert_array_equal(d["x"][:, 0], d["y"])


def test_selection_does_not_require_ground_truth(tmp_path):
    sio.savemat(tmp_path / "data_indian_pines_drl.mat", {"x": np.zeros((6, 200))})
    assert "y" not in load_dataset("indian_pines", tmp_path)
