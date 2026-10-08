import json

import numpy as np
import pytest
import scipy.io as sio

from dragon_bs.evaluate import evaluate, read_bands
from dragon_bs.explain import calibrate_shap


def test_zero_based_selection_without_band_zero_is_not_shifted(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"bands": [5, 9], "band_indexing": "0-based"}))
    np.testing.assert_array_equal(read_bands(p), [5, 9])


def test_matlab_selection_requires_indexing(tmp_path):
    p = tmp_path / "bands.mat"
    sio.savemat(p, {"selected_bands": [5, 9]})
    with pytest.raises(ValueError, match="indexing"):
        read_bands(p)
    np.testing.assert_array_equal(read_bands(p, "1-based"), [4, 8])


def test_duplicate_selection_is_rejected(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"bands": [5, 5], "band_indexing": "0-based"}))
    with pytest.raises(ValueError, match="duplicates"):
        read_bands(p)


def test_classification_excludes_unlabelled_pixels():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-5, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2)),
              np.full((10, 2), 1000)]
    y = np.r_[np.ones(50), np.full(50, 2), np.zeros(10)].astype(int)
    result = evaluate(x, y, [0], train_size=0.2, split_seeds=[0, 1])
    assert result["mean"] == {"OA": 1.0, "AA": 1.0, "Kappa": 1.0}


def test_shap_recovers_sign_scale_on_held_out_states():
    shap = np.random.default_rng(0).normal(size=(20, 4, 3))
    predictions = shap.sum(axis=1) * np.array([-0.1, 0.2, -0.3]) + [1, 2, 3]
    contributions, report = calibrate_shap(shap, predictions)
    assert report["complete"]
    np.testing.assert_allclose(contributions.sum(axis=1) + report["bases"], predictions)


def test_shap_checks_every_output_including_constant_sum():
    shap = np.random.default_rng(0).normal(size=(20, 4, 3))
    predictions = -0.1 * shap.sum(axis=1)
    shap[:, :, 2] = 0
    predictions[:, 2] = np.arange(20)
    _, report = calibrate_shap(shap, predictions)
    assert not report["complete"]
    assert report["n_outputs_passed"] == 2
