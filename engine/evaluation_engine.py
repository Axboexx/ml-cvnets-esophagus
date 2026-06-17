#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from common import DEFAULT_LOG_FREQ, SUPPORTED_VIDEO_CLIP_VOTING_FN
from engine.utils import autocast_fn, get_batch_size, get_log_writers
from metrics.stats import Statistics
from options.parse_args import parse_validation_metric_names
from utils import logger
from utils.common_utils import move_to_device
from utils.ddp_utils import is_master
from utils.tensor_utils import all_gather_list, reduce_tensor_sum


class Evaluator(object):
    # Note: "test_loader" used to be named "eval_loader". We recently renamed data-related "eval_*" names to "test_*"
    #   to follow the standard train/val/test terminology. Engine-related names (eval_engine, is_evaluation, evaluator,
    #   etc.) remained unchanged. One of the reasons was to prevent "eval_engine.py"->"test_engine.py" being
    #   recognized a test suite by pytest.
    def __init__(self, opts, model, test_loader):
        super(Evaluator, self).__init__()

        self.opts = opts

        self.model = model

        self.test_loader = test_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.is_master_node = is_master(opts)
        self.stage_name = getattr(opts, "common.eval_stage_name", "evaluation")
        self.save_location = getattr(opts, "common.exp_loc", "results/eval")
        self.metrics_jsonl_path = Path(self.save_location) / "external_eval_metrics.jsonl"
        self.metrics_csv_path = Path(self.save_location) / "external_eval_metrics.csv"
        self.confusion_jsonl_path = (
            Path(self.save_location) / "external_eval_confusion_matrix.jsonl"
        )
        self.confusion_csv_path = (
            Path(self.save_location) / "external_eval_confusion_matrix.csv"
        )
        self.misclassified_jsonl_path = (
            Path(self.save_location) / "external_eval_misclassified_samples.jsonl"
        )
        self.misclassified_csv_path = (
            Path(self.save_location) / "external_eval_misclassified_samples.csv"
        )
        self.metrics_csv_fields = [
            "stage",
            "total_samples",
            "ckpt_metric_name",
            "ckpt_submetric_name",
            "ckpt_metric",
            "elapsed_seconds",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision",
            "sensitivity",
            "recall",
            "specificity",
            "f1",
            "f1_score",
            "auc",
        ]
        self.confusion_csv_fields = [
            "stage",
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision",
            "sensitivity",
            "recall",
            "specificity",
            "f1",
            "f1_score",
            "auc",
        ]
        self.misclassified_csv_fields = [
            "stage",
            "sample_index",
            "case_id",
            "target",
            "prediction",
            "score",
        ]

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        self.mixed_precision_dtype = getattr(
            opts, "common.mixed_precision_dtype", "float16"
        )

        (
            self.metric_names,
            self.ckpt_metric,
            self.ckpt_submetric,
        ) = parse_validation_metric_names(self.opts)

        self.log_writers = get_log_writers(self.opts, save_location=None)

        # inference modality based eval function
        self.eval_fn = self.eval_fn_image
        inference_modality = getattr(opts, "common.inference_modality", "image")
        if inference_modality is not None and inference_modality.lower() == "video":
            self.eval_fn = self.eval_fn_video

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): Evaluator._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Evaluator._json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _binary_probabilities_from_logits(
        prediction: Tensor, target: Tensor
    ) -> Optional[Dict[str, Tensor]]:
        if target.ndim != 1 or prediction.ndim != 2:
            return None
        if prediction.shape[0] != target.shape[0]:
            return None
        if prediction.shape[1] == 2:
            positive_probability = torch.softmax(prediction.float(), dim=1)[:, 1]
        elif prediction.shape[1] == 1:
            positive_probability = torch.sigmoid(prediction.float()[:, 0])
        else:
            return None

        target_label = target.long()
        if not torch.all((target_label == 0) | (target_label == 1)):
            return None

        return {"labels": target_label, "probabilities": positive_probability}

    @staticmethod
    def _binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
        labels = labels.astype(np.int64)
        probabilities = probabilities.astype(np.float64)

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_count = int(pos_mask.sum())
        neg_count = int(neg_mask.sum())
        if pos_count == 0 or neg_count == 0:
            return 0.5

        order = np.argsort(probabilities, kind="mergesort")
        sorted_probs = probabilities[order]
        ranks = np.empty_like(sorted_probs, dtype=np.float64)

        start = 0
        while start < len(sorted_probs):
            end = start + 1
            while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
                end += 1
            ranks[start:end] = (start + end - 1) / 2.0 + 1.0
            start = end

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        original_ranks = ranks[inv_order]
        rank_sum = float(original_ranks[pos_mask].sum())
        return (rank_sum - pos_count * (pos_count + 1) / 2.0) / (
            pos_count * neg_count
        )

    @classmethod
    def _binary_confusion_from_logits(
        cls, prediction: Tensor, target: Tensor
    ) -> Optional[Tensor]:
        outputs = cls._binary_probabilities_from_logits(
            prediction=prediction, target=target
        )
        if outputs is None:
            return None

        pred_label = (outputs["probabilities"] >= 0.5).long()
        target_label = outputs["labels"]
        tn = ((pred_label == 0) & (target_label == 0)).sum()
        fp = ((pred_label == 1) & (target_label == 0)).sum()
        fn = ((pred_label == 0) & (target_label == 1)).sum()
        tp = ((pred_label == 1) & (target_label == 1)).sum()
        return torch.stack([tn, fp, fn, tp]).to(dtype=torch.float64)

    @classmethod
    def _classification_predictions_from_logits(
        cls, prediction: Tensor, target: Tensor
    ) -> Optional[Dict[str, Tensor]]:
        if target.ndim != 1 or prediction.ndim != 2:
            return None
        if prediction.shape[0] != target.shape[0]:
            return None

        target_label = target.long()
        if prediction.shape[1] == 1:
            score = torch.sigmoid(prediction.float()[:, 0])
            pred_label = (score >= 0.5).long()
        elif prediction.shape[1] == 2:
            score = torch.softmax(prediction.float(), dim=1)[:, 1]
            pred_label = (score >= 0.5).long()
        else:
            probabilities = torch.softmax(prediction.float(), dim=1)
            score, pred_label = torch.max(probabilities, dim=1)

        return {
            "labels": target_label,
            "predictions": pred_label,
            "scores": score,
        }

    def _case_id_from_sample_index(self, sample_index: int) -> str:
        dataset = getattr(self.test_loader, "dataset", None)
        records = getattr(dataset, "records", None)
        if records is None or sample_index < 0 or sample_index >= len(records):
            return str(sample_index)

        record = records[sample_index]
        return str(getattr(record, "case_id", sample_index))

    def _collect_misclassified_samples(
        self,
        batch: Dict,
        prediction: Tensor,
        target: Tensor,
        stage: str,
    ) -> list:
        outputs = self._classification_predictions_from_logits(
            prediction=prediction, target=target
        )
        if outputs is None or "sample_id" not in batch:
            return []

        sample_ids = torch.as_tensor(batch["sample_id"]).detach().cpu().tolist()
        labels = outputs["labels"].detach().cpu().tolist()
        predictions = outputs["predictions"].detach().cpu().tolist()
        scores = outputs["scores"].detach().cpu().tolist()

        misclassified_samples = []
        for sample_index, label, predicted_label, score in zip(
            sample_ids, labels, predictions, scores
        ):
            if int(label) == int(predicted_label):
                continue
            sample_index = int(sample_index)
            misclassified_samples.append(
                {
                    "stage": stage,
                    "sample_index": sample_index,
                    "case_id": self._case_id_from_sample_index(sample_index),
                    "target": int(label),
                    "prediction": int(predicted_label),
                    "score": float(score),
                }
            )
        return misclassified_samples

    def _summarize_binary_confusion(
        self,
        confusion: Optional[Tensor],
        labels: Optional[list],
        probabilities: Optional[list],
        stage: str,
    ) -> Optional[Dict]:
        if confusion is None:
            return None

        if self.use_distributed:
            confusion = reduce_tensor_sum(confusion)
            gathered_labels = all_gather_list(labels or [])
            gathered_probabilities = all_gather_list(probabilities or [])
            labels = [
                item for worker_labels in gathered_labels for item in worker_labels
            ]
            probabilities = [
                item
                for worker_probabilities in gathered_probabilities
                for item in worker_probabilities
            ]

        tn, fp, fn, tp = [int(v) for v in confusion.detach().cpu().tolist()]
        total = tn + fp + fn + tp
        precision = self._safe_divide(tp, tp + fp)
        recall = self._safe_divide(tp, tp + fn)
        specificity = self._safe_divide(tn, tn + fp)
        f1 = self._safe_divide(2.0 * precision * recall, precision + recall)
        auc = None
        if labels and probabilities:
            auc = self._binary_auc(
                labels=np.asarray(labels), probabilities=np.asarray(probabilities)
            )

        return {
            "stage": stage,
            "threshold": 0.5,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": self._safe_divide(tp + tn, total),
            "precision": precision,
            "sensitivity": recall,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "f1_score": f1,
            "auc": auc,
        }

    def _append_eval_metrics(self, metrics: Dict) -> None:
        if not self.is_master_node:
            return

        safe_metrics = self._json_safe(metrics)
        self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_metrics, ensure_ascii=False, sort_keys=True) + "\n")

        write_header = not self.metrics_csv_path.exists()
        row = {field: safe_metrics.get(field) for field in self.metrics_csv_fields}
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_confusion_metrics(self, metrics: Optional[Dict]) -> None:
        if not self.is_master_node or metrics is None:
            return

        safe_metrics = self._json_safe(metrics)
        self.confusion_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.confusion_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_metrics, ensure_ascii=False, sort_keys=True) + "\n")

        write_header = not self.confusion_csv_path.exists()
        row = {field: safe_metrics.get(field) for field in self.confusion_csv_fields}
        with self.confusion_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.confusion_csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_misclassified_samples(self, samples: list) -> None:
        if not self.is_master_node or not samples:
            return

        safe_samples = [self._json_safe(sample) for sample in samples]
        self.misclassified_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.misclassified_jsonl_path.open("a", encoding="utf-8") as f:
            for sample in safe_samples:
                f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")

        write_header = not self.misclassified_csv_path.exists()
        with self.misclassified_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.misclassified_csv_fields)
            if write_header:
                writer.writeheader()
            for sample in safe_samples:
                row = {
                    field: sample.get(field)
                    for field in self.misclassified_csv_fields
                }
                writer.writerow(row)

    def _record_classification_eval_outputs(
        self,
        confusion: Optional[Tensor],
        labels: list,
        probabilities: list,
        misclassified_samples: list,
        total_samples: int,
        elapsed_seconds: float,
        ckpt_metric: Optional[float],
    ) -> None:
        confusion_metrics = self._summarize_binary_confusion(
            confusion=confusion,
            labels=labels,
            probabilities=probabilities,
            stage=self.stage_name,
        )
        self._append_confusion_metrics(confusion_metrics)

        if self.use_distributed:
            gathered_misclassified_samples = all_gather_list(misclassified_samples)
            misclassified_samples = [
                sample
                for worker_samples in gathered_misclassified_samples
                for sample in worker_samples
            ]
        self._append_misclassified_samples(misclassified_samples)

        metrics = {
            "stage": self.stage_name,
            "total_samples": total_samples,
            "ckpt_metric_name": self.ckpt_metric,
            "ckpt_submetric_name": self.ckpt_submetric,
            "ckpt_metric": ckpt_metric,
            "elapsed_seconds": elapsed_seconds,
        }
        if confusion_metrics is not None:
            metrics.update(
                {
                    "tn": confusion_metrics["tn"],
                    "fp": confusion_metrics["fp"],
                    "fn": confusion_metrics["fn"],
                    "tp": confusion_metrics["tp"],
                    "accuracy": confusion_metrics["accuracy"],
                    "precision": confusion_metrics["precision"],
                    "sensitivity": confusion_metrics["sensitivity"],
                    "recall": confusion_metrics["recall"],
                    "specificity": confusion_metrics["specificity"],
                    "f1": confusion_metrics["f1"],
                    "f1_score": confusion_metrics["f1_score"],
                    "auc": confusion_metrics["auc"],
                }
            )
        self._append_eval_metrics(metrics)

    def eval_fn_image(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            opts=self.opts,
            metric_names=self.metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.test_loader)
            processed_samples = 0
            confusion = None
            binary_labels = []
            binary_probabilities = []
            misclassified_samples = []

            for batch_id, batch in enumerate(self.test_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]

                batch_size = get_batch_size(samples)

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)
                    batch_confusion = self._binary_confusion_from_logits(
                        prediction=pred_label, target=targets
                    )
                    if batch_confusion is not None:
                        if confusion is None:
                            confusion = torch.zeros_like(batch_confusion)
                        confusion += batch_confusion
                    batch_binary_outputs = self._binary_probabilities_from_logits(
                        prediction=pred_label, target=targets
                    )
                    if batch_binary_outputs is not None:
                        binary_labels.extend(
                            batch_binary_outputs["labels"].detach().cpu().tolist()
                        )
                        binary_probabilities.extend(
                            batch_binary_outputs["probabilities"]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    misclassified_samples.extend(
                        self._collect_misclassified_samples(
                            batch=batch,
                            prediction=pred_label,
                            target=targets,
                            stage=self.stage_name,
                        )
                    )

                processed_samples += batch_size

                evaluation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={
                        "loss": torch.tensor(0.0, dtype=torch.float, device=self.device)
                    },
                    batch_time=0.0,
                    batch_size=batch_size,
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=0,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=0, stage=self.stage_name)
        ckpt_metric = evaluation_stats.avg_statistics(
            metric_name=self.ckpt_metric, sub_metric_name=self.ckpt_submetric
        )
        self._record_classification_eval_outputs(
            confusion=confusion,
            labels=binary_labels,
            probabilities=binary_probabilities,
            misclassified_samples=misclassified_samples,
            total_samples=processed_samples,
            elapsed_seconds=time.time() - epoch_start_time,
            ckpt_metric=ckpt_metric,
        )

    def eval_fn_video(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            opts=self.opts,
            metric_names=self.metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        num_clips_per_video = getattr(self.opts, "sampler.bs.clips_per_video", 1)
        voting_fn = getattr(
            self.opts, "model.video_classification.clip_out_voting_fn", "sum"
        )
        if voting_fn is None:
            voting_fn = "sum"
        voting_fn = voting_fn.lower()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.test_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.test_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]
                # target_label is Batch*Num_clips
                batch_size_ = get_batch_size(samples)
                batch_size = batch_size_ // num_clips_per_video
                if batch_size_ != (batch_size * num_clips_per_video):
                    logger.log(
                        "Skipping batch. Expected batch size= {}. Got: (bxc:{}x{})".format(
                            batch_size_, batch_size, num_clips_per_video
                        )
                    )
                    continue

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)

                targets = targets.reshape(batch_size, num_clips_per_video)
                # label is the same for all clips in the video
                targets = targets[:, 0]
                pred_label = pred_label.reshape(batch_size, num_clips_per_video, -1)

                if voting_fn == "sum":
                    pred_label = torch.sum(pred_label, dim=1)
                elif voting_fn == "max":
                    pred_label = torch.max(pred_label, dim=1)
                else:
                    logger.error(
                        "--model.video-classification.clip-out-fusion-fn can be {}. Got: {}".format(
                            SUPPORTED_VIDEO_CLIP_VOTING_FN, voting_fn
                        )
                    )

                processed_samples += batch_size

                evaluation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={
                        "loss": torch.tensor(0.0, dtype=torch.float, device=self.device)
                    },
                    batch_time=0.0,
                    batch_size=batch_size,
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=0,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=0, stage=self.stage_name)

    def run(self):
        eval_start_time = time.time()
        self.eval_fn(model=self.model)
        eval_end_time = time.time() - eval_start_time
        logger.log("Evaluation took {} seconds".format(eval_end_time))
