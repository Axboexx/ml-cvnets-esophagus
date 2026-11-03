#!/usr/bin/env python3
"""Plot G-series ROC, calibration and decision curves in the F-series style.

Example (run from the ml-cvnets repository on the server):

    python3 scripts/plot_g_series_figures.py \
        --results-root /home/mw/work/results \
        --output-dir /home/mw/work/results/G_series_figures

Figures (internal = Tongji 60-case validation, external = Youyi):

- ``roc_esophagus_{internal,external}``: G1, G2a, G2b.
- ``roc_ablation_{internal,external}``: G1 and the seven G3 organ combinations.
- ``calibration_<model>_{internal,external}`` and ``dca_<model>_{...}`` for
  G3 esophagus + liver and G3 esophagus + liver + spleen, one model per figure.

Every model uses the epoch with the highest validation AUC (earliest on ties),
which must match the top ``checkpoint_auc_epoch_*`` file and the
``checkpoint_auc_best.pt`` copy evaluated on Youyi. ROC and DCA follow the
2026-10-08 F-series figures (empirical ROC; DCA thresholds 0.01-0.80 with a
0-0.70 net-benefit axis). Calibration follows 交流文件/校准曲线代码2.R: LOESS
(span 0.8) with a 95% confidence band, plus the Brier score.
"""

import argparse
import csv
import filecmp
import hashlib
import json
import math
import platform
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    from scripts.compare_f_series_validation_auc_delong import select_auc_best_epoch
    from scripts.statistical_tests.validation_auc import load_prediction_partition
    from scripts.summarize_g_series_results import (
        EXTERNAL_STAGE,
        EXTERNAL_SUFFIX,
        G_SERIES_RUNS,
        VALIDATION_STAGE,
        top_auc_checkpoint,
    )
except ModuleNotFoundError:
    from compare_f_series_validation_auc_delong import select_auc_best_epoch
    from statistical_tests.validation_auc import load_prediction_partition
    from summarize_g_series_results import (
        EXTERNAL_STAGE,
        EXTERNAL_SUFFIX,
        G_SERIES_RUNS,
        VALIDATION_STAGE,
        top_auc_checkpoint,
    )


@dataclass(frozen=True)
class Model:
    name: str  # key in G_SERIES_RUNS
    label: str
    stem: str
    color: str


MODELS: tuple[Model, ...] = (
    Model("G1", "G1", "g1", "#333333"),
    Model("G2a", "G2a", "g2a", "#E69F00"),
    Model("G2b", "G2b", "g2b", "#0072B2"),
    Model("G3 E+L", "Esophagus + liver", "esophagus_liver", "#D55E00"),
    Model("G3 E+S", "Esophagus + spleen", "esophagus_spleen", "#0072B2"),
    Model("G3 E+PV", "Esophagus + PVS", "esophagus_pvs", "#009E73"),
    Model(
        "G3 E+L+S",
        "Esophagus + liver + spleen",
        "esophagus_liver_spleen",
        "#CC79A7",
    ),
    Model("G3 E+L+PV", "Esophagus + liver + PVS", "esophagus_liver_pvs", "#E69F00"),
    Model("G3 E+S+PV", "Esophagus + spleen + PVS", "esophagus_spleen_pvs", "#56B4E9"),
    Model(
        "G3 E+L+S+PV",
        "Esophagus + liver + spleen + PVS",
        "esophagus_liver_spleen_pvs",
        "#7D5A99",
    ),
)
ROC_GROUPS = (
    ("roc_esophagus", ("G1", "G2a", "G2b")),
    (
        "roc_ablation",
        (
            "G1",
            "G3 E+L",
            "G3 E+S",
            "G3 E+PV",
            "G3 E+L+S",
            "G3 E+L+PV",
            "G3 E+S+PV",
            "G3 E+L+S+PV",
        ),
    ),
)
CLINICAL_CURVE_MODELS = ("G3 E+L", "G3 E+L+S")
CURVE_COLOR = "#D55E00"
LOESS_COLOR = "#2C3E50"
DATASETS = ("internal", "external")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_sources(results_root: Path, name: str) -> tuple[int, float, Path, Path]:
    """Return AUC-best epoch, its logged AUC and both prediction files."""
    runs = {run_name: (subdir, label) for run_name, subdir, label in G_SERIES_RUNS}
    subdir, label = runs[name]
    run_dir = results_root / subdir / label
    metrics_path = run_dir / "validation_confusion_matrix.jsonl"
    validation_path = run_dir / "validation_predictions.csv"
    external_path = (
        results_root
        / subdir
        / f"{label}{EXTERNAL_SUFFIX}"
        / "external_eval_predictions.csv"
    )
    for path in (metrics_path, validation_path, external_path):
        if not path.is_file():
            raise FileNotFoundError(f"{name}: file not found: {path}")
    epoch, recorded_auc, _ = select_auc_best_epoch(metrics_path)
    checkpoint_epoch, checkpoint_path = top_auc_checkpoint(run_dir)
    if checkpoint_epoch != epoch:
        raise ValueError(
            f"{name}: AUC-best epoch {epoch} differs from {checkpoint_path.name}"
        )
    best_copy = run_dir / "checkpoint_auc_best.pt"
    if not best_copy.is_file() or not filecmp.cmp(
        best_copy, checkpoint_path, shallow=False
    ):
        raise ValueError(f"{name}: {best_copy} is not a copy of {checkpoint_path}")
    return epoch, recorded_auc, validation_path, external_path


def load_predictions(
    path: Path, epoch: int | None, stage: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = load_prediction_partition(path, epoch, stage)
    case_ids = np.array(sorted(predictions))
    labels = np.array([predictions[case].target for case in case_ids], dtype=int)
    scores = np.array([predictions[case].score for case in case_ids], dtype=float)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{path}: both classes are required")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{path}: invalid probabilities")
    return case_ids, labels, scores


def roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positive = np.r_[0, np.cumsum(sorted_labels)[ends]]
    false_positive = np.r_[0, (ends + 1) - np.cumsum(sorted_labels)[ends]]
    tpr = true_positive / labels.sum()
    fpr = false_positive / (len(labels) - labels.sum())
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    positives, negatives = scores[labels == 1], scores[labels == 0]
    paired_auc = float(
        np.mean(
            (positives[:, None] > negatives) + 0.5 * (positives[:, None] == negatives)
        )
    )
    if not np.isclose(auc, paired_auc, atol=1e-12):
        raise ValueError("AUC integration and pairwise calculation disagree")
    return fpr, tpr, auc


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def axis_style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#DDDDDD", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def dataset_title(dataset: str, n: int) -> str:
    title = "Internal validation" if dataset == "internal" else "External validation"
    return f"{title} (n = {n})"


def plot_roc(
    group: Sequence[Model], data: dict, dataset: str, output: Path, name: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.5))
    for model in group:
        labels, scores = data[(model.name, dataset)]
        fpr, tpr, auc = roc(labels, scores)
        ax.plot(
            fpr,
            tpr,
            color=model.color,
            linewidth=1.9,
            label=f"{model.label} (AUC = {auc:.3f})",
        )
        write_csv(
            output / f"{name}_{model.stem}_coordinates.csv",
            [
                {"false_positive_rate": float(x), "true_positive_rate": float(y)}
                for x, y in zip(fpr, tpr)
            ],
        )
    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=":", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - Specificity", ylabel="Sensitivity")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(dataset_title(dataset, len(labels)), fontsize=13, pad=12)
    axis_style(ax)
    many = len(group) > 3
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=9,
        ncol=1 if many else 3,
        handlelength=3,
    )
    fig.subplots_adjust(bottom=0.33 if many else 0.19)
    save_figure(fig, output, name)


def plot_dca(
    y: np.ndarray, p: np.ndarray, model: Model, dataset: str, output: Path, name: str
) -> None:
    thresholds = np.linspace(0.01, 0.80, 160)
    net_benefit = np.array(
        [
            (
                (y[p >= threshold] == 1).sum()
                - (y[p >= threshold] == 0).sum() * threshold / (1 - threshold)
            )
            / len(y)
            for threshold in thresholds
        ]
    )
    treat_all = y.mean() - (1 - y.mean()) * thresholds / (1 - thresholds)
    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    ax.plot(
        thresholds, net_benefit, color=CURVE_COLOR, linewidth=2.2, label=model.label
    )
    ax.plot(
        thresholds,
        treat_all,
        color="#777777",
        linestyle="--",
        linewidth=1.6,
        label="Treat all",
    )
    ax.axhline(0, color="#222222", linestyle=":", linewidth=1.5, label="Treat none")
    ax.set(
        xlim=(0, 0.8),
        ylim=(0, 0.70),
        xlabel="Threshold probability",
        ylabel="Net benefit",
    )
    ax.set_title(f"{dataset_title(dataset, len(y))}\n{model.label}", fontsize=13)
    axis_style(ax)
    ax.legend(
        frameon=False, loc="lower left", fontsize=10, bbox_to_anchor=(0.02, 0.025)
    )
    fig.tight_layout()
    save_figure(fig, output, name)
    write_csv(
        output / f"{name}_coordinates.csv",
        [
            {
                "threshold_probability": float(t),
                "model_net_benefit": float(nb),
                "treat_all_net_benefit": float(a),
                "treat_none_net_benefit": 0.0,
            }
            for t, nb, a in zip(thresholds, net_benefit, treat_all)
        ],
    )


def calibration_bins(y: np.ndarray, p: np.ndarray, count: int) -> list[dict[str, Any]]:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, count + 1)))
    memberships = np.searchsorted(edges[1:-1], p, side="right")
    bins = []
    for index in np.unique(memberships):
        selected = memberships == index
        n = int(selected.sum())
        events = int(y[selected].sum())
        rate = events / n
        z = 1.959963984540054
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = (
            z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        )
        bins.append(
            {
                "bin": int(index + 1),
                "n": n,
                "events": events,
                "mean_predicted_probability": float(p[selected].mean()),
                "observed_event_rate": rate,
                "wilson_95ci_lower": float(max(0.0, min(rate, center - half))),
                "wilson_95ci_upper": float(min(1.0, max(rate, center + half))),
            }
        )
    return bins


def _t_quantile(probability: float, df: float) -> float:
    """Student-t quantile by bisection on Simpson-integrated density (no SciPy)."""
    log_norm = (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    )

    def cdf(x: float) -> float:
        grid = np.linspace(0.0, x, 4001)
        density = np.exp(log_norm - (df + 1) / 2 * np.log1p(grid**2 / df))
        step = x / 4000
        area = (
            step
            / 3
            * (
                density[0]
                + density[-1]
                + 4 * density[1:-1:2].sum()
                + 2 * density[2:-1:2].sum()
            )
        )
        return 0.5 + float(area)

    low, high = 0.0, 1.0
    while cdf(high) < probability:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if cdf(middle) < probability else (low, middle)
    return (low + high) / 2


def _local_fit_rows(
    p: np.ndarray, point: float, q: int, degree: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Rows giving the local tricube polynomial's value and slope at point."""
    distance = np.abs(p - point)
    radius = np.sort(distance)[q - 1]
    if radius <= 0:
        raise ValueError("LOESS neighbourhood has zero width")
    weights = np.where(distance < radius, (1 - (distance / radius) ** 3) ** 3, 0.0)
    design = np.vander(p - point, degree + 1, increasing=True)
    weighted = design * weights[:, None]
    solution = np.linalg.pinv(design.T @ weighted) @ weighted.T
    return solution[0], solution[1]


def _loess_vertices(p: np.ndarray, span: float, cell: float = 0.2) -> np.ndarray:
    """k-d tree cell bounds of R loess (ehg126/ehg129) for one predictor."""
    capacity = math.floor(len(p) * span * cell)
    low, high = float(p.min()), float(p.max())
    pad = 0.005 * max(high - low, 1e-10 * max(abs(low), abs(high)) + 1e-30)
    vertices = {low - pad, high + pad}
    stack = [(low - pad, high + pad, np.sort(p))]
    while stack:
        lower, upper, points = stack.pop()
        if len(points) <= capacity:
            continue
        # 1-based median m = (1 + count) / 2, moved off ties as in ehg129.
        m, offset = (1 + len(points)) // 2, 0
        while 1 <= m + offset < len(points):
            if points[m + offset - 1] == points[m + offset]:
                offset = -offset + (1 if -offset >= 0 else 0)
            else:
                m += offset
                break
        split = float(points[m - 1])
        if split in (lower, upper):
            continue
        vertices.add(split)
        stack.append((lower, split, points[:m]))
        stack.append((split, upper, points[m:]))
    return np.array(sorted(vertices))


def _loess_operator(p: np.ndarray, points: np.ndarray, span: float) -> np.ndarray:
    """Rows map outcomes to R loess(degree=2) fits at points.

    R's default surface="interpolate": exact local fits (value and slope) at
    the k-d tree vertices, cubic Hermite interpolation between them.
    """
    q = math.floor(len(p) * span)
    if not 1 <= q <= len(p):
        raise ValueError("LOESS span leaves no neighbourhood")
    vertices = _loess_vertices(p, span)
    rows = [_local_fit_rows(p, vertex, q) for vertex in vertices]
    values = np.array([row[0] for row in rows])
    slopes = np.array([row[1] for row in rows])
    operator = np.empty((len(points), len(p)))
    for index, point in enumerate(points):
        cell = np.searchsorted(vertices, point, side="right") - 1
        cell = min(max(cell, 0), len(vertices) - 2)
        width = vertices[cell + 1] - vertices[cell]
        t = (point - vertices[cell]) / width
        operator[index] = (
            (1 - t) ** 2 * (1 + 2 * t) * values[cell]
            + t**2 * (3 - 2 * t) * values[cell + 1]
            + width * t * (1 - t) ** 2 * slopes[cell]
            - width * t**2 * (1 - t) * slopes[cell + 1]
        )
    return operator


def loess_calibration(
    y: np.ndarray, p: np.ndarray, span: float, points: int = 80
) -> dict[str, np.ndarray]:
    """Match ggplot2 geom_smooth(method="loess", se=TRUE) on (pred, obs).

    R loess defaults (degree 2, tricube, gaussian, surface="interpolate") at
    ggplot's 80 evenly spaced points. The 95% band is fit +/- t(0.975, df) * se
    with R's residual scale and look-up df = delta1^2 / delta2; R approximates
    delta1 and delta2 (ehg141) where this computes them exactly, so the band
    can differ from R by about 1e-4.
    """
    hat = _loess_operator(p, p, span)
    residual_maker = np.eye(len(p)) - hat
    delta = residual_maker.T @ residual_maker
    delta1 = float(np.trace(delta))
    delta2 = float(np.trace(delta @ delta))
    residual_scale = math.sqrt(float(np.sum((y - hat @ y) ** 2)) / delta1)
    grid = np.linspace(float(p.min()), float(p.max()), points)
    operator = _loess_operator(p, grid, span)
    fit = operator @ y
    se = residual_scale * np.sqrt(np.sum(operator**2, axis=1))
    margin = _t_quantile(0.975, delta1**2 / delta2) * se
    return {
        "grid": grid,
        "fit": fit,
        "se": se,
        "lower": fit - margin,
        "upper": fit + margin,
    }


def plot_calibration(
    y: np.ndarray,
    p: np.ndarray,
    model: Model,
    dataset: str,
    output: Path,
    name: str,
    span: float,
) -> None:
    """Calibration curve in the style of 交流文件/校准曲线代码2.R."""
    bins = calibration_bins(y, p, 6)
    curve = loess_calibration(y, p, span)
    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    ax.fill_between(
        curve["grid"],
        curve["lower"],
        curve["upper"],
        color="skyblue",
        alpha=0.4,
        linewidth=0,
        label="95% CI",
    )
    ax.plot(
        curve["grid"],
        curve["fit"],
        color=LOESS_COLOR,
        linewidth=2.2,
        label=f"LOESS (span = {span:g})",
    )
    ax.plot(
        [0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.5, label="Ideal"
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted Probability",
        ylabel="Observed Proportion",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{dataset_title(dataset, len(y))}\n{model.label}", fontsize=13)
    ax.text(
        0.04,
        0.96,
        f"Brier score = {np.mean((p - y) ** 2):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    # theme_minimal: light grid, no axis lines.
    ax.spines[:].set_visible(False)
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=9,
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.21)
    save_figure(fig, output, name)
    write_csv(output / f"{name}_bins.csv", bins)
    write_csv(
        output / f"{name}_loess_coordinates.csv",
        [
            {
                "predicted_probability": float(x),
                "loess_fit": float(f),
                "standard_error": float(e),
                "ci95_lower": float(lo),
                "ci95_upper": float(hi),
            }
            for x, f, e, lo, hi in zip(
                curve["grid"], curve["fit"], curve["se"], curve["lower"], curve["upper"]
            )
        ],
    )


def overview(figures_dir: Path, names: list[str], path: Path) -> None:
    rows = (len(names) + 1) // 2
    canvas = Image.new("RGB", (1800, 600 * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        with Image.open(figures_dir / f"{name}.png") as image:
            thumb = ImageOps.contain(image.convert("RGB"), (875, 555))
        x, y = (index % 2) * 900, (index // 2) * 600
        canvas.paste(thumb, (x + (900 - thumb.width) // 2, y + 30))
        draw.text((x + 25, y + 8), name, fill="#333333")
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Must not already exist"
    )
    parser.add_argument("--loess-span", type=float, default=0.8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"error: output directory already exists: {output_dir}")
    by_name = {model.name: model for model in MODELS}

    data, cohorts, sources, summary = {}, {}, [], []
    for model in MODELS:
        epoch, recorded_auc, validation_path, external_path = resolve_sources(
            results_root, model.name
        )
        for dataset, path, stage, part_epoch in (
            ("internal", validation_path, VALIDATION_STAGE, epoch),
            ("external", external_path, EXTERNAL_STAGE, None),
        ):
            digest = _sha256(path)
            case_ids, labels, scores = load_predictions(path, part_epoch, stage)
            cohort = np.column_stack([case_ids, labels.astype(str)])
            if dataset in cohorts and not np.array_equal(cohorts[dataset], cohort):
                raise ValueError(f"{model.name}: {dataset} cases/labels differ")
            cohorts[dataset] = cohort
            auc = roc(labels, scores)[2]
            if dataset == "internal" and not np.isclose(auc, recorded_auc, atol=1e-8):
                raise ValueError(
                    f"{model.name}: recorded AUC {recorded_auc} != recomputed {auc}"
                )
            data[(model.name, dataset)] = (labels, scores)
            sources.append(
                {
                    "model": model.name,
                    "dataset": dataset,
                    "path": str(path),
                    "sha256": digest,
                    "auc_best_epoch": epoch,
                    "n": len(labels),
                    "events": int(labels.sum()),
                }
            )
            summary.append(
                {
                    "model": model.name,
                    "label": model.label,
                    "dataset": dataset,
                    "auc_best_epoch": epoch,
                    "n": len(labels),
                    "events": int(labels.sum()),
                    "auc": auc,
                    "brier_score": float(np.mean((scores - labels) ** 2)),
                }
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.labelsize": 12,
            "lines.linewidth": 1.8,
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temp_name:
        temp_dir = Path(temp_name)
        figures_dir = temp_dir / "figures"
        figures_dir.mkdir()
        names = []
        for prefix, group in ROC_GROUPS:
            for dataset in DATASETS:
                name = f"{prefix}_{dataset}"
                plot_roc(
                    [by_name[key] for key in group], data, dataset, figures_dir, name
                )
                names.append(name)
        for key in CLINICAL_CURVE_MODELS:
            model = by_name[key]
            for dataset in DATASETS:
                labels, scores = data[(key, dataset)]
                name = f"calibration_{model.stem}_{dataset}"
                plot_calibration(
                    labels,
                    scores,
                    model,
                    dataset,
                    figures_dir,
                    name,
                    args.loess_span,
                )
                names.append(name)
            for dataset in DATASETS:
                labels, scores = data[(key, dataset)]
                name = f"dca_{model.stem}_{dataset}"
                plot_dca(labels, scores, model, dataset, figures_dir, name)
                names.append(name)

        for source in sources:
            if _sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"input changed during plotting: {source['path']}")
        write_csv(temp_dir / "metrics_summary.csv", summary)
        manifest = {
            "sources": sources,
            "positive_class": 1,
            "figures": names,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "selection": (
                "AUC-best validation epoch (earliest on ties); external = "
                "checkpoint_auc_best.pt of that epoch; no selection on Youyi"
            ),
            "calibration": (
                "ggplot2 geom_smooth(method='loess', se=TRUE) equivalent on "
                "(predicted probability, 0/1 outcome): local quadratic, tricube, "
                "gaussian family; 95% band = fit +/- t * se; apparent, no "
                "recalibration; display clipped to [0, 1] like coord_cartesian"
            ),
            "loess_span": args.loess_span,
            "dca": (
                "Thresholds 0.01-0.80; p >= threshold; net benefit "
                "TP/n - FP/n * t/(1-t); y axis 0-0.70"
            ),
        }
        (temp_dir / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        overview(figures_dir, names, temp_dir / "overview.png")
        temp_dir.rename(output_dir)

    for row in summary:
        print(
            f"{row['model']:<12} {row['dataset']:<8} epoch={row['auc_best_epoch']:<4} "
            f"n={row['n']:<4} AUC={row['auc']:.3f} Brier={row['brier_score']:.3f}"
        )
    print(f"{len(names)} figures written to: {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
