#!/usr/bin/env bash

set -e

python3 -c 'from main_eval import main_worker; main_worker()' \
    --common.config-file config/classification/esophagus_risk/suprem_unet_mil_roi.yaml \
    --common.results-loc ~/work/results/F2a \
    --common.run-label f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_external \
    --common.eval-stage-name youyi_external \
    --common.seed 42 \
    --model.classification.pretrained ~/work/results/F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42/checkpoint_best.pt \
    --common.override-kwargs \
        dataset.name=esophagus_risk_youyi_roi_3d \
        dataset.root_test=~/work/dataset/youyi \
        dataset.eval_batch_size0=4 \
        dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
        dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
        dataset.esophagus_roi_3d.roi_margin_xy=0 \
        dataset.esophagus_roi_3d.roi_margin_z=0 \
        model.classification.suprem_unet_mil_roi.pretrained_path=false \
        model.classification.suprem_unet_mil_roi.num_segments=4

python3 -c 'from main_eval import main_worker; main_worker()' \
    --common.config-file config/classification/esophagus_risk/suprem_unet_dual_mil_roi.yaml \
    --common.results-loc ~/work/results/F2b \
    --common.run-label f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_exterl \
    --common.eval-stage-name youyi_external \
    --common.seed 42 \
    --model.classification.pretrained ~/work/results/F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42/checkpoint_best.pt \
    --common.override-kwargs \
        dataset.name=esophagus_risk_youyi_roi_3d \
        dataset.root_test=~/work/dataset/youyi \
        dataset.eval_batch_size0=1 \
        dataset.esophagus_roi_3d.min_esophagus_height_mm=40.0 \
        dataset.esophagus_roi_3d.inferior_to_superior_height_mm=40.0 \
        dataset.esophagus_roi_3d.roi_margin_xy=0 \
        dataset.esophagus_roi_3d.roi_margin_z=0 \
        model.classification.suprem_unet_dual_mil_roi.pretrained_path=false \
        model.classification.suprem_unet_dual_mil_roi.num_segments=4

python3 -c 'from main_train import main_worker; main_worker()' \
    --common.config-file config/classification/esophagus_risk/suprem_unet_e3a_residual_context.yaml \
    --common.results-loc ~/work/results/F3_esophagus_spleen \
    --common.run-label f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42 \
    --common.seed 42 \
    --common.override-kwargs \
        dataset.root_train=/home/mw/work/dataset/egc_dataset \
        dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,spleen \
        dataset.e3a_multi_organ_roi_3d.label_ids=15,1 \
        dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
        dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
        dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm=40.0 \
        dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm=40.0 \
        dataset.e3a_multi_organ_roi_3d.val_split_ratio=0.2 \
        dataset.e3a_multi_organ_roi_3d.split_seed=42 \
        model.classification.suprem_unet_e3a_residual_context.pretrained_path=~/work/SuPreM/supervised_suprem_unet_2100.pth \
        model.classification.suprem_unet_e3a_residual_context.num_organs=2
