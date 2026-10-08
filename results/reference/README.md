# Historical selection artifacts

These six CSV files were copied unchanged from the pre-cleanup working tree.
`experimentos.csv` refers to Indian Pines; the other names identify the scene.
Each file has an algorithm label, an experiment identifier, and 30 band indices.

They contain mixed historical runs: different numbers of repetitions, exploratory
algorithm labels, hyperparameter sweeps, and baseline selections. An `exp_id` is
not a verified training seed or classification split. Index conventions and
selection/ranking order have not been independently certified for every method.
No automatic one-based/zero-based conversion is applied.

These artifacts support inspection of the research history. They are not a
machine-readable reproduction of Table II or a source of certified accuracy
numbers. New runs use `run.json`/`bands.mat` with explicit 0-based indexing.

The raw files retain their original names and bytes for provenance. The later
Fisher-reward experiments, checkpoints, intermediate figures and logs remain in
the author's ignored local archive, separate from these artifacts.
