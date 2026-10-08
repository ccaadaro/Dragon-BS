# Dataset preparation

Datasets are external inputs and are not redistributed here. Use the original
providers and their accompanying documentation:

- [Indian Pines, Pavia, Botswana and Salinas: UPV/EHU scene collection](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
- [MUUFL Gulfport: GatorSense data and scene labels](https://github.com/GatorSense/MUUFLGulfport).
- [WHU-Hi: Wuhan University resource sharing](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm).
- [Houston: University of Houston hyperspectral resources](https://hyperspectral.ee.uh.edu/).

The following paths and keys describe the **local input layouts used by the
original scripts**, which may differ from provider filenames. Renaming a MATLAB
file does not rename its internal key.

| CLI dataset | Spectral path relative to `data4drl/` | MATLAB key | Bands | Layout |
| --- | --- | --- | ---: | --- |
| `indian_pines` | `data_indian_pines_drl.mat` | `x` | 200 | MATLAB-flattened pixels × bands |
| `pavia` | `paviaU.mat` | `paviaU` | 103 | rows × columns × bands |
| `gulfport` | `muufl_gulfport_campus_1_hsi_220_label.mat` | `hsi[0,0].Data` | 64 | nested MUUFL cube |
| `houston` | `houston.mat` | `houston` | 144 | rows × columns × bands |
| `hanchuan` | `HanChuan/WHU_Hi_HanChuan.mat` | `WHU_Hi_HanChuan` | 274 | rows × columns × bands |
| `honghu` | `HongHu/WHU_Hi_HongHu.mat` | `WHU_Hi_HongHu` | 270 | rows × columns × bands |
| `botswana` | `data_botswana_drl.mat` | `x` | 145 | MATLAB-flattened pixels × bands |
| `salinas` | `Salinas_corrected.mat` | `salinas_corrected` | 204 | rows × columns × bands |

Botswana and Salinas are supported auxiliary datasets; the paper evaluates the
other six scenes.

For Indian Pines, convert a corrected 200-band cube to the expected layout using
MATLAB ordering:

```python
import numpy as np
import scipy.io as sio
cube = sio.loadmat("Indian_pines_corrected.mat")["indian_pines_corrected"]
assert cube.shape[-1] == 200
sio.savemat("data4drl/data_indian_pines_drl.mat", {
    "x": np.asarray(cube).reshape(-1, 200, order="F")
})
```

Keep spectral values from the chosen provider input; the runner converts them to
float32. Entropy uses a 256-bin histogram independently for every band and then
min/max normalization across bands. The loader does not silently remove,
interpolate or reorder spectral bands.

Ground truth is needed only for evaluation:

| Dataset | Ground-truth path | MATLAB key |
| --- | --- | --- |
| Indian Pines | `data4classification/indian_pines_GT.mat` | `indian_pines_GT` |
| Pavia | `data4classification/paviaU_gt.mat` | `paviaU_gt` |
| Houston | `data4classification/houston_gt_sum.mat` | `houston_gt_sum` |
| HanChuan | `data4drl/HanChuan/WHU_Hi_HanChuan_gt.mat` | `WHU_Hi_HanChuan_gt` |
| HongHu | `data4drl/HongHu/WHU_Hi_HongHu_gt.mat` | `WHU_Hi_HongHu_gt` |
| Botswana | `data4classification/Botswana_gt.mat` | `Botswana_gt` |
| Salinas | `data4classification/Salinas_gt.mat` | `salinas_gt` |

MUUFL labels are read from `hsi[0,0].sceneLabels[0,0].labels` in the spectral file.
Negative labels become 0 (unlabelled). Other scene labels use 0 for unlabelled
pixels. Pre-flattened Indian Pines/Botswana spectra align with **Fortran-order**
labels; 3D cubes and their labels are flattened together in **C order**.

Houston's `houston_gt_sum.mat` is the local workflow's combined label map.
Obtaining and combining provider train/test maps must preserve their pixel
alignment and non-overlapping labels. The evaluator creates new stratified
splits; it does not claim to reproduce the provider's official split.

The paper reports 93 Pavia bands and 72 Gulfport bands, while the retained local
scripts and CSV indices use 103 and 64 respectively. No mapping resolving that
difference was available during repository preparation. These layouts are
therefore stated explicitly rather than presented as equivalent. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
