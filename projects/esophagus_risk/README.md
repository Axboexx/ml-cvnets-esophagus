# Esophagus Risk on ml-cvnets

This directory contains the esophagus-risk classification experiments on top of `ml-cvnets`.

## Recommended path: native 3D ml-cvnets integration

The current recommended experiment is a native `ml-cvnets` classification pipeline:

- input: full `3D CT`
- supervision: `HR / no-HR`
- ROI extraction: online esophagus crop using `mask == 15`
- model: native registry-based 3D classifier
- training entry: `cvnets-train`

### Expected dataset layout

```text
dataset_root/
  HR/
    case_001/
      *.nii.gz
      *mask*.nii.gz
    case_002/
      ...
  no-HR/
    case_101/
      *.nii.gz
      *mask*.nii.gz
```

Each case directory should contain one CT volume and one corresponding mask volume. The dataset code searches recursively inside each case directory and identifies the mask file using keywords such as `mask`, `seg`, or `label`.

### Native 3D training

Run from the `ml-cvnets` root:

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/esophagus_roi_3d.yaml \
  --common.results-loc results/esophagus_roi_3d \
  --common.override-kwargs \
    dataset.root_train=/path/to/dataset_root
```

If you already have a dedicated validation split, point `dataset.root_val` to that root as well:

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/esophagus_roi_3d.yaml \
  --common.results-loc results/esophagus_roi_3d \
  --common.override-kwargs \
    dataset.root_train=/path/to/train_root \
    dataset.root_val=/path/to/val_root
```

If `dataset.root_val` is empty, the dataset performs an internal case-level split using:

- `dataset.esophagus_roi_3d.val_split_ratio`
- `dataset.esophagus_roi_3d.split_seed`

### Main components

- dataset: `data/datasets/classification/esophagus_risk_roi_3d.py`
- model: `cvnets/models/classification/esophagus_roi_3d.py`
- config: `config/classification/esophagus_risk/esophagus_roi_3d.yaml`

### Key assumptions

- esophagus label id is `15`
- input is resized to a fixed `3D` ROI size
- training uses CT windowing by default
- `common.channels_last` must remain `false` for this 3D path

## Legacy 2D path

Older scripts in this directory support exporting `2D PNG` slices and running custom case-level baselines. They are retained only for comparison and debugging, not as the main training path.

- `prepare_imagefolder.py`
- `case_dataset.py`
- `case_model.py`
- `train_roi_crop.py`
