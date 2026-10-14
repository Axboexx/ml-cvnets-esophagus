#!/usr/bin/env bash
# Run all 12 F-series Youyi external evaluations in order.
# Edit the paths and individual commands below as needed.
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: bash scripts/run_f_series_auc_external_eval.sh' >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_ROOT=/home/mw/work/results
YOUYI_ROOT=/home/mw/work/dataset/youyi
YOUYI_CLINICAL_WORKBOOK=/home/mw/work/dataset/友谊临床指标填补-4cm.xlsx

# Choose the highest validation-AUC checkpoint; ties use the earlier epoch.
# Fail if the run has no AUC checkpoint (never fall back to an accuracy checkpoint).
best_auc_checkpoint() {
    python3 -c '
from pathlib import Path
import re
import sys

run_dir = Path(sys.argv[1])
pattern = re.compile(r"checkpoint_auc_epoch_(\d+)_score_([0-9]+(?:\.[0-9]+)?)\.pt")
matches = []
for path in run_dir.glob("checkpoint_auc_epoch_*_score_*.pt"):
    match = pattern.fullmatch(path.name)
    if match:
        matches.append((float(match.group(2)), -int(match.group(1)), path))
if not matches:
    sys.exit(f"No validation-AUC checkpoint in {run_dir}")
print(max(matches)[2])
' "$1"
}

# F1
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F1/f1_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_roi.yaml \
  --common.results-loc "$RESULTS_ROOT/F1" \
  --common.run-label f1_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
    dataset.esophagus_roi_3d.roi_margin_xy=0 \
    dataset.esophagus_roi_3d.roi_margin_z=0 \
    model.classification.suprem_unet_roi.require_pretrained=false

# F2a
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_mil_roi.yaml \
  --common.results-loc "$RESULTS_ROOT/F2a" \
  --common.run-label f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
    dataset.esophagus_roi_3d.roi_margin_xy=0 \
    dataset.esophagus_roi_3d.roi_margin_z=0 \
    model.classification.suprem_unet_mil_roi.require_pretrained=false \
    model.classification.suprem_unet_mil_roi.num_segments=4

# F2b
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42_auc_ckpt_3")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_dual_mil_roi.yaml \
  --common.results-loc "$RESULTS_ROOT/F2b" \
  --common.run-label f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42_auc_ckpt_3_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
    dataset.esophagus_roi_3d.roi_margin_xy=0 \
    dataset.esophagus_roi_3d.roi_margin_z=0 \
    model.classification.suprem_unet_dual_mil_roi.require_pretrained=false \
    model.classification.suprem_unet_dual_mil_roi.num_segments=4

# F3: esophagus + liver
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_liver/f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_liver" \
  --common.run-label f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=2

# F3: esophagus + liver + portal venous system
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_liver_portal/f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_liver_portal" \
  --common.run-label f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver,portal_venous_system \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5,64 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=3

# F3: esophagus + liver + spleen
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_liver_spleen/f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_liver_spleen" \
  --common.run-label f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver,spleen \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5,1 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=3

# F3: esophagus + liver + spleen + portal venous system
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_liver_spleen_portal/f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_liver_spleen_portal" \
  --common.run-label f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver,spleen,portal_venous_system \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5,1,64 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0,0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0,0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=4

# F3: esophagus + portal venous system
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_portal_venous_system/f3_esophagus_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_portal_venous_system" \
  --common.run-label f3_esophagus_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,portal_venous_system \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,64 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=2

# F3: esophagus + spleen
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_spleen/f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_spleen" \
  --common.run-label f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,spleen \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,1 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=2

# F3: esophagus + spleen + portal venous system
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F3_esophagus_spleen_portal_venous_system/f3_esophagus_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F3_esophagus_spleen_portal_venous_system" \
  --common.run-label f3_esophagus_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42_auc_ckpt_rerun_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_e3a_multi_organ_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,spleen,portal_venous_system \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,1,64 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0,0 \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=3

# F4
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F4/f4_tongji82_4cm_4seg_15clinical_seed42")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_f4_clinical_roi.yaml \
  --common.results-loc "$RESULTS_ROOT/F4" \
  --common.run-label f4_tongji82_4cm_4seg_15clinical_seed42_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_f4_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
    dataset.esophagus_roi_3d.roi_margin_xy=0 \
    dataset.esophagus_roi_3d.roi_margin_z=0 \
    dataset.esophagus_f4.clinical_workbook="$YOUYI_CLINICAL_WORKBOOK" \
    model.classification.suprem_unet_dual_mil_roi.require_pretrained=false \
    model.classification.suprem_unet_dual_mil_roi.num_segments=4

# F5
CHECKPOINT=$(best_auc_checkpoint "$RESULTS_ROOT/F5/f5_esophagus_liver_clinical_tongji82_4cm_seed42")
python3 -c 'from main_eval import main_worker; main_worker()' \
  --common.config-file config/classification/esophagus_risk/suprem_unet_f5_clinical_context.yaml \
  --common.results-loc "$RESULTS_ROOT/F5" \
  --common.run-label f5_esophagus_liver_clinical_tongji82_4cm_seed42_youyi_external_auc_best \
  --common.eval-stage-name youyi_external \
  --common.seed 42 \
  --model.classification.pretrained "$CHECKPOINT" \
  --common.override-kwargs \
    dataset.name=esophagus_risk_youyi_f5_clinical_roi_3d \
    dataset.root_test="$YOUYI_ROOT" \
    dataset.eval_batch_size0=4 \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5 \
    dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
    dataset.esophagus_f5.clinical_workbook="$YOUYI_CLINICAL_WORKBOOK" \
    model.classification.suprem_unet_e3a_residual_context.require_pretrained=false \
    model.classification.suprem_unet_e3a_residual_context.num_organs=2
