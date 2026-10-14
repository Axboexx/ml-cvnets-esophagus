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

### SuPreM U-Net ROI fine-tuning

The SuPreM experiment uses the same online `mask == 15` ROI crop dataset as the
native 3D baseline, but replaces the compact 3D CNN with a SuPreM U-Net backbone
and a classification head.

Download the SuPreM U-Net checkpoint separately:

```text
supervised_suprem_unet_2100.pth
```

The SuPreM source repository is discovered from one of these locations:

- `SUPREM_ROOT` environment variable
- `../SuPreM/supervised_pretraining`
- `~/work/SuPreM/supervised_pretraining`
- `~/project/SuPreM/supervised_pretraining`

Then run from the `ml-cvnets` root:

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/suprem_unet_roi.yaml \
  --common.results-loc results/suprem_unet_roi \
  --common.override-kwargs \
    dataset.root_train=/path/to/dataset_root \
    model.classification.suprem_unet_roi.pretrained_path=/path/to/supervised_suprem_unet_2100.pth
```

If `dataset.root_val` is empty, the dataset performs an internal case-level split using:

- `dataset.esophagus_roi_3d.val_split_ratio`
- `dataset.esophagus_roi_3d.split_seed`

### Youyi external test dataset

The Youyi dataset uses separated CT and mask directories:

```text
youyi/
  youyi-nii/
    nhr/
      1/
      2/
    hr/
      1/
      2/
  youyi-mask/
    nhr/
      1/
      2/
    hr/
      1/
      2/
```

Use `esophagus_risk_youyi_roi_3d` for this layout. It pairs cases by the class
folder (`nhr` or `hr`) and case folder name, then reuses the same ROI extraction
and preprocessing settings from `dataset.esophagus_roi_3d`.

The current protocol uses a deterministic, class-stratified Tongji 80/20 split
(`val_split_ratio: 0.2`, `split_seed: 42`). The held-out 20% is reused by the
framework as validation during training and as the internal test set during
evaluation. Youyi remains a full, independent external test set.

For multi-organ F3 experiments, every configured organ uses zero X/Y/Z margins.
The esophagus keeps its inferior-to-superior 40 mm crop with tight X/Y bounds;
liver, spleen, and portal venous system use their full tight 3D mask bounding boxes.

Example evaluation command:

```bash
cvnets-eval \
  --common.config-file config/classification/esophagus_risk/suprem_unet_roi.yaml \
  --common.results-loc results/suprem_unet_roi \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_roi_3d \
    dataset.root_test=/path/to/youyi \
    model.classification.suprem_unet_roi.pretrained_path=/path/to/supervised_suprem_unet_2100.pth
```

### SuPreM Swin UNETR ROI fine-tuning

This variant keeps the same ROI crop, labels, optimizer, and metrics as the
SuPreM U-Net experiment, but uses the SuPreM Swin UNETR encoder feature before
the decoder.

Download the SuPreM Swin UNETR checkpoint separately:

```text
supervised_suprem_swinunetr_2100.pth
```

Then run from the `ml-cvnets` root:

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/suprem_swinunetr_roi.yaml \
  --common.results-loc results/suprem_swinunetr_roi \
  --common.override-kwargs \
    dataset.root_train=/path/to/dataset_root \
    model.classification.suprem_swinunetr_roi.pretrained_path=/path/to/supervised_suprem_swinunetr_2100.pth
```

### F4 image-clinical fusion under the 4 cm protocol

F4 keeps the new F-series image protocol (eligible `mask == 15` height at least
40 mm, inferior-to-superior 40 mm crop, zero XYZ margin, and four MIL depth
segments) and adds all 15 workbook predictors. The fixed clinical feature order
is:

```text
sex, age, hct, hb, plt, pt, inr, ast, alt, alb, tb, tp, cr, child, hepatitis
```

`id` is used only to join image and clinical cases. `hrev` is used only for
label consistency checks; the image dataset folder remains the training target.
Missing cases, duplicate normalized IDs, non-finite values, and label conflicts
fail before training. The clinical branch applies source-training BatchNorm,
then a `15 -> 32 -> 16` MLP. Its embedding is concatenated with the F2b image
representation immediately before the binary classifier.

Both supplied filenames indicate that missing values were already imputed.
Before an official F4 run, retain the imputation method and fitted statistics as
provenance and confirm that validation/Youyi values were not used to fit them.
If cohort-level statistics were fitted before splitting, regenerate the tables
with imputation fitted on the Tongji training subset only; do not treat the
current files as leakage-safe merely because they contain no empty cells.

Run Tongji development training from the `ml-cvnets` root:

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/suprem_unet_f4_clinical_roi.yaml \
  --common.results-loc results/F4 \
  --common.override-kwargs \
    dataset.root_train=/path/to/tongji \
    dataset.esophagus_f4.clinical_workbook=/path/to/同济临床填补指标-4cm.xlsx \
    model.classification.suprem_unet_dual_mil_roi.pretrained_path=/path/to/supervised_suprem_unet_2100.pth
```

For independent Youyi evaluation, load the selected Tongji checkpoint through
the normal evaluation checkpoint option and override the dataset and workbook:

```bash
cvnets-eval \
  --common.config-file config/classification/esophagus_risk/suprem_unet_f4_clinical_roi.yaml \
  --common.results-loc results/F4 \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_f4_roi_3d \
    dataset.root_test=/path/to/youyi \
    dataset.esophagus_f4.clinical_workbook=/path/to/友谊临床指标填补-4cm.xlsx \
    model.classification.suprem_unet_dual_mil_roi.pretrained_path=/path/to/supervised_suprem_unet_2100.pth \
    model.classification.pretrained=/path/to/tongji_f4/checkpoint_best.pt
```

Youyi is evaluation-only: its values do not update the clinical BatchNorm
statistics or participate in checkpoint/threshold selection. The Youyi F4
dataset registration rejects train/validation construction at runtime.

### F5 esophagus-liver-clinical fusion from SuPreM

F5 extends the F3 esophagus+liver residual-context model with the same 15-item
clinical encoder used by F4. The shared SuPreM image branch produces a 512-D
representation (`esophagus + projected liver residual`); the clinical branch
produces 16 dimensions, and the binary classifier receives their 528-D
concatenation. Both esophagus and liver are required. The experiment starts
from the SuPreM backbone rather than an F3 checkpoint, and auto-resume is
disabled in the F5 config.

Tongji training from the server `ml-cvnets` root:

```bash
python3 -c 'from main_train import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_f5_clinical_context.yaml \
  --common.results-loc ~/work/results/F5 \
  --common.run-label f5_esophagus_liver_clinical_tongji82_4cm_seed42 \
  --common.seed 42 \
  --common.override-kwargs \
    dataset.root_train=/home/mw/work/dataset/egc_dataset \
    dataset.root_val="" \
    dataset.root_test="" \
    dataset.esophagus_f5.clinical_workbook=/home/mw/work/dataset/同济临床填补指标-4cm.xlsx \
    model.classification.suprem_unet_e3a_residual_context.pretrained_path=/home/mw/work/SuPreM/supervised_suprem_unet_2100.pth
```

Do not add `model.classification.pretrained` to this training command: that
option loads a complete prior model and would no longer be a SuPreM-only start.

### Saved training metrics

Each run writes lightweight analysis files under:

```text
<common.results-loc>/<common.run-label>/
```

Important files:

- `training_metrics.jsonl`: one JSON record per epoch, including train/val loss, checkpoint metric, best metric, learning rate, iteration, and elapsed time.
- `training_metrics.csv`: the same epoch-level metrics in spreadsheet-friendly format.
- `validation_confusion_matrix.jsonl` / `validation_confusion_matrix.csv`: binary validation confusion matrix (`tn`, `fp`, `fn`, `tp`) and derived metrics per epoch, including Recall, F1-score, and AUC when model outputs are binary classification logits.
- `training_run_summary.json`: run status, final training time, last epoch metrics, best metric, and error traceback if training failed.

### Main components

- dataset: `data/datasets/classification/esophagus_risk_roi_3d.py`
- Youyi test dataset: `data/datasets/classification/esophagus_risk_youyi_roi_3d.py`
- model: `cvnets/models/classification/esophagus_roi_3d.py`
- SuPreM model: `cvnets/models/classification/suprem_unet_roi.py`
- SuPreM Swin UNETR model: `cvnets/models/classification/suprem_swinunetr_roi.py`
- F4 clinical data contract: `projects/esophagus_risk/clinical_data.py`
- F4 Tongji/Youyi datasets: `data/datasets/classification/esophagus_risk_f4_roi_3d.py`
- F4 fusion model: `cvnets/models/classification/suprem_unet_f4_clinical_roi.py`
- config: `config/classification/esophagus_risk/esophagus_roi_3d.yaml`
- SuPreM config: `config/classification/esophagus_risk/suprem_unet_roi.yaml`
- SuPreM Swin UNETR config: `config/classification/esophagus_risk/suprem_swinunetr_roi.yaml`
- F4 config: `config/classification/esophagus_risk/suprem_unet_f4_clinical_roi.yaml`
- F5 Tongji/Youyi datasets: `data/datasets/classification/esophagus_risk_f5_clinical_roi_3d.py`
- F5 fusion model: `cvnets/models/classification/suprem_unet_f5_clinical_context.py`
- F5 config: `config/classification/esophagus_risk/suprem_unet_f5_clinical_context.yaml`

### Key assumptions

- esophagus label id is `15`
- the crop starts at the inferior end and extends 40 mm toward the superior end
- the X/Y bounds are the tight mask bounds within that 40 mm segment, with no margin
- MIL local branches use 4 depth segments
- input is resized to a fixed `3D` ROI size
- training uses CT windowing by default
- `common.channels_last` must remain `false` for this 3D path

## Legacy 2D path

Older scripts in this directory support exporting `2D PNG` slices and running custom case-level baselines. They are retained only for comparison and debugging, not as the main training path.

- `prepare_imagefolder.py`
- `case_dataset.py`
- `case_model.py`
- `train_roi_crop.py`
