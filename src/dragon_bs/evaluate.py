"""Evaluate exported selections with fixed, shared classification splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from .data import DATASETS, load_dataset


def read_bands(path, indexing=None):
    """Read an explicit index convention; never infer it from the minimum index."""
    path = Path(path)
    if path.suffix == ".json":
        report = json.loads(path.read_text())
        raw = np.asarray(report["bands"]).reshape(-1)
        indexing = indexing or report.get("band_indexing")
    elif path.suffix == ".mat":
        report = sio.loadmat(path)
        raw = np.asarray(report["selected_bands"]).reshape(-1)
        if indexing is None and "band_indexing" in report:
            indexing = str(np.asarray(report["band_indexing"]).reshape(-1)[0]).strip()
    else:
        raise ValueError("Use a run.json or bands.mat file.")
    if indexing not in ("0-based", "1-based"):
        raise ValueError("Band indexing is missing; specify --indexing 0-based or 1-based.")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("Band indices must be finite integers.")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Band indices must be integers.")
    bands = raw.astype(np.int64) - (indexing == "1-based")
    if not len(bands) or len(np.unique(bands)) != len(bands):
        raise ValueError("Band selection must be nonempty and contain no duplicates.")
    return bands


def evaluate(x, y, bands, *, train_size=0.1, split_seeds=(0, 1, 2, 3, 4), classifier="knn"):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    bands = np.asarray(bands, dtype=np.int64)
    if (bands < 0).any() or (bands >= x.shape[1]).any():
        raise ValueError(f"Band indices must be in [0, {x.shape[1] - 1}].")
    labelled = y > 0
    features, labels = x[labelled][:, bands], y[labelled]
    scores = []
    for seed in split_seeds:
        xt, xv, yt, yv = train_test_split(features, labels, train_size=train_size,
                                         stratify=labels, random_state=seed)
        if classifier == "knn":
            if len(yt) < 5:
                raise ValueError("5-NN evaluation needs at least 5 training pixels.")
            estimator = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        elif classifier == "svm":
            if np.unique(yt, return_counts=True)[1].min() < 5:
                raise ValueError("SVM grid search needs at least 5 training samples per class.")
            estimator = GridSearchCV(
                make_pipeline(StandardScaler(), SVC()),
                {"svc__C": [1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1]},
                cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=1)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        estimator.fit(xt, yt)
        prediction = estimator.predict(xv)
        scores.append({"split_seed": int(seed), "OA": float(accuracy_score(yv, prediction)),
                       "AA": float(recall_score(yv, prediction, labels=np.unique(labels),
                                                average="macro", zero_division=0)),
                       "Kappa": float(cohen_kappa_score(yv, prediction))})
    return {"runs": scores, "mean": {k: float(np.mean([s[k] for s in scores]))
                                      for k in ("OA", "AA", "Kappa")},
            "std": {k: float(np.std([s[k] for s in scores])) for k in ("OA", "AA", "Kappa")}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", type=Path, help="run.json or bands.mat")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data-dir", type=Path, default=Path("data4drl"))
    p.add_argument("--gt-dir", type=Path, default=Path("data4classification"))
    p.add_argument("--indexing", choices=["0-based", "1-based"])
    p.add_argument("--classifier", choices=["knn", "svm"], default="knn")
    p.add_argument("--train-size", type=float, default=0.1)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = p.parse_args(argv)
    if not 0 < args.train_size < 1:
        p.error("--train-size must be between 0 and 1")
    try:
        bands = read_bands(args.selection, args.indexing)
        data = load_dataset(args.dataset, args.data_dir, args.gt_dir, with_labels=True)
        result = evaluate(data["x"], data["y"], bands, train_size=args.train_size,
                          split_seeds=args.split_seeds, classifier=args.classifier)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        p.error(str(exc))
    report = {"dataset": args.dataset, "bands": bands.tolist(), "band_indexing": "0-based",
              "classifier": args.classifier, "train_size": args.train_size,
              "split_seeds": args.split_seeds, "standardize": "training split only",
              "source_selection": str(args.selection), **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["mean"], indent=2))
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
