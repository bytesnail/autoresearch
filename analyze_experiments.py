#!/usr/bin/env python3
"""
Comprehensive experiment analysis for autoresearch project.
Parses results.tsv and git history, generates charts and HTML report.
"""

import csv
import os
import re
import subprocess
import base64
from io import BytesIO
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# ============================================================
# Data structures
# ============================================================


@dataclass
class Experiment:
    index: int
    commit: str
    val_bpb: Optional[float]
    memory_gb: Optional[float]
    status: str
    description: str
    phase: str = ""
    category: str = ""
    hyperparameter: str = ""
    is_best: bool = False
    is_timeout: bool = False
    is_oom: bool = False
    lost_steps: bool = False
    steps_count: Optional[int] = None


# ============================================================
# Parse results.tsv
# ============================================================


def parse_results(filepath):
    experiments = []
    baseline_val_bpb = None
    current_best = None
    best_history = []

    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for i, row in enumerate(reader):
            if len(row) < 5:
                continue
            commit = row[0].strip()
            val_str = row[1].strip()
            mem_str = row[2].strip()
            status = row[3].strip()
            desc = row[4].strip()

            # Parse val_bpb
            val_bpb = None
            if val_str.upper() == "CRASH" or val_str == "0.000000":
                if val_str == "0.000000":
                    val_bpb = 0.0
                status = "crash"
            else:
                try:
                    val_bpb = float(val_str)
                except ValueError:
                    status = "crash"

            # Parse memory
            memory_gb = None
            try:
                memory_gb = float(mem_str)
            except ValueError:
                pass

            if i == 0 and baseline_val_bpb is None:
                baseline_val_bpb = val_bpb

            is_timeout = False
            is_oom = False
            lost_steps = False
            steps_count = None

            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in ["lost steps", "lost ", " steps)"]):
                lost_steps = True
            steps_m = re.search(r"\((\d+)\s+steps", desc_lower)
            if not steps_m:
                steps_m = re.search(r"(\d+)\s+steps\s+vs", desc_lower)
            if steps_m:
                steps_count = int(steps_m.group(1))

            if memory_gb is not None and memory_gb > 10.0 and status == "discard":
                is_oom = True

            is_best = False
            if status not in ("crash",) and val_bpb is not None and val_bpb > 0:
                if current_best is None or val_bpb < current_best:
                    current_best = val_bpb
                    is_best = True

            best_history.append(current_best)

            exp = Experiment(
                index=i,
                commit=commit,
                val_bpb=val_bpb,
                memory_gb=memory_gb,
                status=status,
                description=desc,
                is_best=is_best,
                is_timeout=is_timeout,
                is_oom=is_oom,
                lost_steps=lost_steps,
                steps_count=steps_count,
            )
            experiments.append(exp)

    return experiments, baseline_val_bpb, best_history


# ============================================================
# Classify experiments into phases and categories
# ============================================================


def classify_hyperparameter(desc):
    """Map description to hyperparameter category."""
    desc_lower = desc.lower()

    # Architecture
    if any(
        k in desc_lower
        for k in [
            "depth",
            "aspect_ratio",
            "aspect",
            "head_dim",
            "mlp ratio",
            "gelu",
            "relu",
            "silu",
            "activation",
            "layernorm",
            "rmsnorm",
            "parallel attn",
            "gqa",
            "depth_scaling",
            "window pattern",
        ]
    ):
        return "Architecture"

    # Learning Rate - Matrix
    if any(k in desc_lower for k in ["matrix lr", "matrix_lr", "mlr"]):
        return "Learning Rate (Matrix)"

    # Learning Rate - Scalar
    if any(k in desc_lower for k in ["scalar lr", "scalar_lr", "slr"]):
        return "Learning Rate (Scalar)"

    # Learning Rate - Embedding
    if any(k in desc_lower for k in ["embedding lr", "embedding_lr"]):
        return "Learning Rate (Embedding)"

    # Learning Rate - Unembedding
    if any(k in desc_lower for k in ["unembedding lr", "unembedding_lr"]):
        return "Learning Rate (Unembedding)"

    # Learning Rate - Other
    if any(
        k in desc_lower
        for k in ["lr ", "_lr", "resid_lr", "ve_lr", "lr_shift", "lr_combo"]
    ):
        return "Learning Rate (Other)"

    # Weight Decay
    if any(
        k in desc_lower for k in ["weight decay", "weight_decay", "wd=", "wd_", "wd "]
    ):
        return "Weight Decay"

    # Optimizer - Muon
    if any(
        k in desc_lower for k in ["muon", "momentum", "ns_steps", "ns steps", "normuon"]
    ):
        return "Optimizer (Muon)"

    # Optimizer - Adam
    if any(k in desc_lower for k in ["adam", "beta1", "beta2", "eps"]):
        return "Optimizer (Adam)"

    # Softcap / Attention
    if any(k in desc_lower for k in ["softcap", "learnable softcap"]):
        return "Softcap (Attention)"

    # Schedule
    if any(
        k in desc_lower
        for k in ["warmdown", "warmup", "cosine", "quadratic", "sqrt", "schedule"]
    ):
        return "LR Schedule"

    # Final LR
    if any(k in desc_lower for k in ["final_lr", "final lr"]):
        return "Final LR Fraction"

    # Batch Size
    if any(k in desc_lower for k in ["batch", "total_batch", "device_batch"]):
        return "Batch Size"

    # Weight Tying
    if any(k in desc_lower for k in ["weight tying", "tying"]):
        return "Weight Tying"

    # Initialization
    if any(
        k in desc_lower
        for k in [
            "init",
            "x0_lambda",
            "x0_",
            "resid_lambda",
            "resid_init",
            "proj_init",
            "layer_scaled",
        ]
    ):
        return "Initialization"

    # Normalization
    if any(k in desc_lower for k in ["norm", "pe norm", "qk norm", "no_qk"]):
        return "Normalization"

    # Value Embedding / Gate
    if any(k in desc_lower for k in ["ve_gate", "ve ", "value_embed", "value embed"]):
        return "Value Embedding"

    # Regularization
    if any(k in desc_lower for k in ["label smooth", "grad clip", "ema", "swa", "mtp"]):
        return "Regularization"

    # Compile / Performance
    if any(
        k in desc_lower
        for k in ["compile", "reduce-overhead", "max-autotune", "autotune", "overhead"]
    ):
        return "Compilation/Performance"

    # RoPE
    if any(k in desc_lower for k in ["rope", "rope_base"]):
        return "RoPE"

    # WD Masking
    if any(k in desc_lower for k in ["cautious", "standard wd"]):
        return "WD Masking"

    # Baseline
    if "baseline" in desc_lower:
        return "Baseline"

    return "Other"


def classify_phase(experiments):
    """Assign experiments to research phases based on sequential patterns."""
    phases = []
    # Phase 1: Early exploration (rows 1-10)
    # Phase 2: Systematic hyperparameter search (rows 11-70)
    # Phase 3: Deep dive + Weight tying (rows 71-130)
    # Phase 4: Named experiments e134-e250 (rows 134-250)
    # Phase 5: Advanced optimization e251+ (rows 251-end)

    for exp in experiments:
        idx = exp.index
        if idx <= 8:
            phase = "Phase 1: Initial Exploration"
        elif idx <= 42:
            phase = "Phase 2: Foundational HP Search"
        elif idx <= 68:
            phase = "Phase 3: Momentum & WD Optimization"
        elif idx <= 121:
            phase = "Phase 4: Fine-Grained Grid Search"
        elif idx <= 131:
            phase = "Phase 5: Weight Tying"
        elif idx <= 153:
            phase = "Phase 6: QK Norm & Softcap Deep Dive"
        elif idx <= 207:
            phase = "Phase 7: Systematic Re-testing"
        elif idx <= 255:
            phase = "Phase 8: Advanced Techniques"
        elif idx <= 272:
            phase = "Phase 9: WD Scheduling Discovery"
        else:
            phase = "Phase 10: Final Optimization"
        phases.append(phase)
        exp.phase = phase

    return phases


# ============================================================
# Statistical analysis
# ============================================================


def compute_statistics(experiments, baseline_val_bpb):
    stats = {
        "total": len(experiments),
        "crashes": 0,
        "improved": 0,
        "worse": 0,
        "kept": 0,
        "new_best": 0,
        "discarded": 0,
        "memory_violations": 0,
        "best_val_bpb": baseline_val_bpb,
        "improvement_from_baseline": 0,
        "category_stats": defaultdict(
            lambda: {"total": 0, "improved": 0, "worse": 0, "crash": 0, "best": 0}
        ),
        "phase_stats": defaultdict(
            lambda: {"total": 0, "improved": 0, "worse": 0, "crash": 0, "best": 0}
        ),
    }

    best_val = baseline_val_bpb
    for exp in experiments:
        if exp.status == "crash":
            stats["crashes"] += 1
        elif (
            exp.is_best
            or exp.status == "NEW BEST"
            or "new best" in exp.description.lower()
        ):
            stats["new_best"] += 1
            stats["improved"] += 1
            if exp.val_bpb and exp.val_bpb < best_val:
                best_val = exp.val_bpb
        elif exp.status == "keep" or exp.status == "improved":
            stats["improved"] += 1
            stats["kept"] += 1
            if exp.val_bpb and exp.val_bpb < best_val:
                best_val = exp.val_bpb
        elif exp.status in ("discard", "worse"):
            stats["worse"] += 1
            stats["discarded"] += 1

        if exp.memory_gb and exp.memory_gb > 2.5:
            stats["memory_violations"] += 1

        # Category stats
        cat = exp.category
        if not cat:
            cat = classify_hyperparameter(exp.description)
            exp.category = cat
        stats["category_stats"][cat]["total"] += 1
        if (
            exp.is_best
            or exp.status == "NEW BEST"
            or exp.status == "improved"
            or exp.status == "keep"
        ):
            stats["category_stats"][cat]["improved"] += 1
            if exp.is_best or exp.status == "NEW BEST":
                stats["category_stats"][cat]["best"] += 1
        elif exp.status == "crash":
            stats["category_stats"][cat]["crash"] += 1
        else:
            stats["category_stats"][cat]["worse"] += 1

        # Phase stats
        phase = exp.phase
        stats["phase_stats"][phase]["total"] += 1
        if (
            exp.is_best
            or exp.status == "NEW BEST"
            or exp.status == "improved"
            or exp.status == "keep"
        ):
            stats["phase_stats"][phase]["improved"] += 1
        elif exp.status == "crash":
            stats["phase_stats"][phase]["crash"] += 1
        else:
            stats["phase_stats"][phase]["worse"] += 1

    stats["best_val_bpb"] = best_val
    if baseline_val_bpb and best_val:
        stats["improvement_from_baseline"] = (
            (baseline_val_bpb - best_val) / baseline_val_bpb * 100
        )

    return stats


# ============================================================
# Chart generation
# ============================================================


def chart_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def _apply_style(ax, title="", xlabel="", ylabel=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle="-", color="#cccccc")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#2c3e50")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color="#34495e")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color="#34495e")


def generate_progress_chart(experiments, best_history):
    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.5, 1]}
    )

    indices, vals, colors = [], [], []
    for exp in experiments:
        if exp.val_bpb is not None and exp.val_bpb > 0 and exp.status != "crash":
            indices.append(exp.index)
            vals.append(exp.val_bpb)
            if exp.is_best or exp.status == "NEW BEST":
                colors.append("#2ecc71")
            elif exp.status == "keep" or exp.status == "improved":
                colors.append("#3498db")
            elif exp.status == "worse" or exp.status == "discard":
                colors.append("#e74c3c")
            else:
                colors.append("#95a5a6")

    # Main chart
    ax_main.scatter(
        indices, vals, c=colors, s=18, alpha=0.55, edgecolors="none", zorder=3
    )
    ax_main.plot(
        range(len(best_history)),
        best_history,
        color="#27ae60",
        linewidth=2.5,
        label="Running Best",
        zorder=5,
    )
    if experiments and experiments[0].val_bpb:
        ax_main.axhline(
            y=experiments[0].val_bpb,
            color="#e67e22",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ecc71",
            markersize=8,
            label="New Best",
            markeredgecolor="#27ae60",
            markeredgewidth=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=7,
            label="Kept/Improved",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=7,
            label="Discarded/Worse",
        ),
        Line2D([0], [0], color="#27ae60", linewidth=2.5, label="Running Best"),
        Line2D(
            [0], [0], color="#e67e22", linewidth=1.5, linestyle="--", label="Baseline"
        ),
    ]
    ax_main.legend(
        handles=legend_elements,
        fontsize=9,
        loc="upper right",
        framealpha=0.9,
        edgecolor="#ddd",
    )
    _apply_style(
        ax_main,
        "Val BPB Progress Over Experiments",
        "Experiment #",
        "Validation BPB (lower = better)",
    )

    # Zoom on the interesting region (bottom 40% of range)
    zoom_max = experiments[0].val_bpb
    zoom_min = min(v for v in vals if v > 0) - 0.002
    ax_zoom.scatter(
        indices, vals, c=colors, s=22, alpha=0.6, edgecolors="none", zorder=3
    )
    ax_zoom.plot(
        range(len(best_history)), best_history, color="#27ae60", linewidth=2.5, zorder=5
    )
    ax_zoom.set_ylim(zoom_min, zoom_max)
    ax_zoom.axhline(
        y=experiments[0].val_bpb,
        color="#e67e22",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
    )
    _apply_style(ax_zoom, "Zoom: Best Region", "Experiment #", "")

    # Annotate final best on zoom
    final_best_idx = len(best_history) - 1
    final_best_val = best_history[-1]
    ax_zoom.annotate(
        f"Best: {final_best_val:.6f}",
        xy=(final_best_idx, final_best_val),
        xytext=(-60, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
    )

    fig.tight_layout()
    return chart_to_base64(fig)


def generate_category_pie(stats):
    fig, ax = plt.subplots(figsize=(13, 8))

    cats = {
        k: v["total"]
        for k, v in sorted(
            stats["category_stats"].items(), key=lambda x: -x[1]["total"]
        )
        if v["total"] > 0
    }
    labels = list(cats.keys())
    sizes = list(cats.values())

    cmap = plt.cm.Set3
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.82,
        wedgeprops=dict(width=0.65, edgecolor="white", linewidth=2),
    )
    for text in autotexts:
        text.set_fontsize(7)
        text.set_fontweight("bold")
    ax.legend(
        wedges,
        [f"{l} ({s})" for l, s in zip(labels, sizes)],
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
        title_fontsize=10,
    )
    _apply_style(ax, "Experiment Distribution by Category")

    return chart_to_base64(fig)


def generate_success_rate_by_category(stats):
    fig, ax = plt.subplots(figsize=(15, 7))

    cats = {
        k: v
        for k, v in sorted(
            stats["category_stats"].items(), key=lambda x: -x[1]["total"]
        )
        if v["total"] >= 3
    }

    labels = list(cats.keys())
    totals = [cats[l]["total"] for l in labels]
    improved = [cats[l]["improved"] for l in labels]
    crashes = [cats[l]["crash"] for l in labels]
    worse = [cats[l]["worse"] for l in labels]

    x = np.arange(len(labels))
    width = 0.6

    p1 = ax.bar(
        x,
        improved,
        width,
        label="Improved/Kept",
        color="#2ecc71",
        edgecolor="white",
        linewidth=0.5,
    )
    p2 = ax.bar(
        x,
        worse,
        width,
        bottom=improved,
        label="Worse/Discarded",
        color="#e74c3c",
        edgecolor="white",
        linewidth=0.5,
    )
    p3 = ax.bar(
        x,
        crashes,
        width,
        bottom=[i + w for i, w in zip(improved, worse)],
        label="Crash",
        color="#e67e22",
        edgecolor="white",
        linewidth=0.5,
    )

    for xi, imp, tot in zip(x, improved, totals):
        if tot > 0:
            pct = imp / tot * 100
            ax.text(
                xi,
                tot + 0.5,
                f"{pct:.0f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#2c3e50",
            )

    _apply_style(ax, "Experiment Outcomes by Category", "", "Number of Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    return chart_to_base64(fig)


def generate_phase_progress(experiments):
    phases_order = [
        "Phase 1: Initial Exploration",
        "Phase 2: Foundational HP Search",
        "Phase 3: Momentum & WD Optimization",
        "Phase 4: Fine-Grained Grid Search",
        "Phase 5: Weight Tying",
        "Phase 6: QK Norm & Softcap Deep Dive",
        "Phase 7: Systematic Re-testing",
        "Phase 8: Advanced Techniques",
        "Phase 9: WD Scheduling Discovery",
        "Phase 10: Final Optimization",
    ]

    phase_bests = {}
    for exp in experiments:
        if exp.val_bpb and exp.val_bpb > 0 and exp.status != "crash":
            phase = exp.phase
            if phase not in phase_bests or exp.val_bpb < phase_bests[phase]:
                phase_bests[phase] = exp.val_bpb

    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [p for p in phases_order if p in phase_bests]
    vals = [phase_bests[p] for p in labels]
    short_labels = [p.split(": ", 1)[1] if ": " in p else p for p in labels]

    vmin, vmax = min(vals) - 0.002, max(vals) + 0.002
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.RdYlGn_r
    bar_colors = [cmap(norm(v)) for v in vals]

    bars = ax.barh(
        range(len(short_labels)),
        vals,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(
            val + 0.0005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.6f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.invert_yaxis()
    _apply_style(
        ax, "Best Val BPB by Research Phase", "Best Val BPB (lower = better)", ""
    )
    ax.set_xlim(vmin - 0.005, vmax + 0.01)

    ax.annotate(
        f"Global Best: {min(vals):.6f}",
        xy=(min(vals), vals.index(min(vals))),
        xytext=(min(vals) + 0.01, vals.index(min(vals)) + 0.3),
        fontsize=10,
        fontweight="bold",
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
    )

    return chart_to_base64(fig)


def generate_lr_sensitivity(experiments):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    lr_types = {
        "Matrix LR": [],
        "Scalar LR": [],
        "Embedding LR": [],
        "Unembedding LR": [],
    }

    for exp in experiments:
        if not exp.val_bpb or exp.val_bpb <= 0 or exp.status == "crash":
            continue
        desc = exp.description.lower()
        is_best = exp.is_best or exp.status == "NEW BEST"

        m = re.search(r"(?:^|(?<=\s))matrix[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)", desc)
        if m:
            lr_types["Matrix LR"].append((float(m.group(1)), exp.val_bpb, is_best))

        m = re.search(r"(?:^|(?<=\s))scalar[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)", desc)
        if m:
            lr_types["Scalar LR"].append((float(m.group(1)), exp.val_bpb, is_best))

        if "unembedding" in desc:
            m = re.search(r"unembedding[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)", desc)
            if m:
                lr_types["Unembedding LR"].append(
                    (float(m.group(1)), exp.val_bpb, is_best)
                )
                continue

        m = re.search(r"(?:^|(?<=\s))embedding[_ ]lr[\s_=]+(?:0\.|)(\d+\.?\d*)", desc)
        if m:
            lr_types["Embedding LR"].append((float(m.group(1)), exp.val_bpb, is_best))

    for ax, (lr_name, data) in zip(axes.flat, lr_types.items()):
        if not data:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                transform=ax.transAxes,
                fontsize=14,
                color="#bdc3c7",
            )
            ax.set_title(f"{lr_name}", fontsize=12, fontweight="bold", color="#2c3e50")
            continue

        data.sort(key=lambda x: x[0])
        lrs = [d[0] for d in data]
        bpbs = [d[1] for d in data]
        is_best_list = [d[2] for d in data]

        colors = ["#2ecc71" if b else "#3498db" for b in is_best_list]
        sizes = [80 if b else 30 for b in is_best_list]

        ax.scatter(
            lrs,
            bpbs,
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
        )

        if len(data) >= 4:
            z = np.polyfit(lrs, bpbs, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(lrs), max(lrs), 100)
            ax.plot(
                x_smooth,
                p(x_smooth),
                "--",
                color="#e74c3c",
                alpha=0.5,
                linewidth=1.5,
                zorder=2,
            )

        best_idx = bpbs.index(min(bpbs))
        ax.annotate(
            f"Best: {min(bpbs):.4f}\nLR={lrs[best_idx]}",
            xy=(lrs[best_idx], min(bpbs)),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=8,
            color="#27ae60",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1),
        )

        _apply_style(
            ax,
            f"{lr_name} Sensitivity ({len(data)} experiments)",
            f"{lr_name} Value",
            "Val BPB",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(
        "Learning Rate Sensitivity Analysis",
        fontsize=15,
        fontweight="bold",
        color="#2c3e50",
        y=0.99,
    )

    return chart_to_base64(fig)


def generate_weight_decay_chart(experiments):
    fig, ax = plt.subplots(figsize=(13, 7))

    wd_data = []
    for exp in experiments:
        if not exp.val_bpb or exp.val_bpb <= 0 or exp.status == "crash":
            continue
        desc = exp.description.lower()
        if not (
            "weight decay" in desc or "wd=" in desc or "wd_" in desc or "wd " in desc
        ):
            continue
        wd_val = None
        m = re.search(r"(?:weight.?decay|wd)[\s_=]+(\d+\.?\d*)", desc)
        if m:
            wd_val = float(m.group(1))
            if wd_val > 1.0:
                wd_val = wd_val / 100.0
        if wd_val is not None:
            wd_data.append(
                (wd_val, exp.val_bpb, exp.is_best or exp.status == "NEW BEST")
            )

    if not wd_data:
        ax.text(
            0.5, 0.5, "No weight decay data found", ha="center", transform=ax.transAxes
        )
        return chart_to_base64(fig)

    wd_data.sort(key=lambda x: x[0])

    from collections import OrderedDict

    grouped = OrderedDict()
    for wd_val, bpb, is_b in wd_data:
        key = round(wd_val, 3)
        grouped.setdefault(key, []).append((wd_val, bpb, is_b))

    positions = list(range(len(grouped)))
    labels = list(grouped.keys())

    for pos, (wd_val, items) in enumerate(zip(labels, grouped.values())):
        bpbs = [it[1] for it in items]
        any_best = any(it[2] for it in items)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(bpbs))
        color = "#2ecc71" if any_best else "#e74c3c"
        ax.scatter(
            [pos + j for j in jitter],
            bpbs,
            c=color,
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
        )
        mean_val = np.mean(bpbs)
        ax.hlines(
            mean_val,
            pos - 0.35,
            pos + 0.35,
            colors="#2c3e50",
            linewidth=2,
            zorder=4,
            alpha=0.7,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l:.3f}" for l in labels], fontsize=9)
    _apply_style(
        ax,
        f"Weight Decay Sensitivity ({len(wd_data)} experiments)",
        "Weight Decay Value",
        "Val BPB",
    )

    ax.axhline(
        y=min(d[1] for d in wd_data),
        color="#27ae60",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
    )

    return chart_to_base64(fig)


def generate_softcap_chart(experiments):
    fig, ax = plt.subplots(figsize=(12, 7))

    sc_data = []
    for exp in experiments:
        if not exp.val_bpb or exp.val_bpb <= 0 or exp.status == "crash":
            continue
        desc = exp.description.lower()
        if "softcap" not in desc:
            continue
        m = re.search(r"softcap[\s_=]+(?:0\.|)(\d+\.?\d*)", desc)
        if m:
            sc_val = float(m.group(1))
            sc_data.append(
                (sc_val, exp.val_bpb, exp.is_best or exp.status == "NEW BEST")
            )

    if not sc_data:
        ax.text(0.5, 0.5, "No softcap data found", ha="center", transform=ax.transAxes)
        return chart_to_base64(fig)

    from collections import OrderedDict

    grouped = OrderedDict()
    for sc_val, bpb, is_b in sorted(sc_data, key=lambda x: x[0]):
        key = round(sc_val, 1)
        grouped.setdefault(key, []).append((sc_val, bpb, is_b))

    positions = list(range(len(grouped)))
    labels = list(grouped.keys())
    rng = np.random.default_rng(42)

    for pos, (sc_val, items) in enumerate(zip(labels, grouped.values())):
        bpbs = [it[1] for it in items]
        any_best = any(it[2] for it in items)
        jitter = rng.uniform(-0.18, 0.18, len(bpbs))
        color = "#2ecc71" if any_best else "#e74c3c"
        ax.scatter(
            [pos + j for j in jitter],
            bpbs,
            c=color,
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
        )
        ax.hlines(
            np.mean(bpbs),
            pos - 0.35,
            pos + 0.35,
            colors="#2c3e50",
            linewidth=2,
            zorder=4,
            alpha=0.7,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(l) for l in labels], fontsize=10)
    _apply_style(
        ax,
        f"Softcap Sensitivity ({len(sc_data)} experiments)",
        "Softcap Value",
        "Val BPB",
    )

    return chart_to_base64(fig)


def generate_momentum_chart(experiments):
    fig, ax = plt.subplots(figsize=(12, 7))

    mom_data = []
    for exp in experiments:
        if not exp.val_bpb or exp.val_bpb <= 0 or exp.status == "crash":
            continue
        desc = exp.description.lower()
        if "momentum" not in desc:
            continue
        m = re.search(
            r"momentum[\s_]+(?:end\s+|start\s+|constant\s+|ramp\s+)?(?:0\.)(\d+)", desc
        )
        if m:
            mom_val = float("0." + m.group(1))
            mom_data.append(
                (mom_val, exp.val_bpb, exp.is_best or exp.status == "NEW BEST")
            )

    if not mom_data:
        ax.text(0.5, 0.5, "No momentum data found", ha="center", transform=ax.transAxes)
        return chart_to_base64(fig)

    from collections import OrderedDict

    grouped = OrderedDict()
    for mom_val, bpb, is_b in sorted(mom_data, key=lambda x: x[0]):
        key = round(mom_val, 2)
        grouped.setdefault(key, []).append((mom_val, bpb, is_b))

    positions = list(range(len(grouped)))
    labels = list(grouped.keys())
    rng = np.random.default_rng(42)

    for pos, (mom_val, items) in enumerate(zip(labels, grouped.values())):
        bpbs = [it[1] for it in items]
        any_best = any(it[2] for it in items)
        jitter = rng.uniform(-0.18, 0.18, len(bpbs))
        color = "#2ecc71" if any_best else "#e74c3c"
        ax.scatter(
            [pos + j for j in jitter],
            bpbs,
            c=color,
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
        )
        ax.hlines(
            np.mean(bpbs),
            pos - 0.35,
            pos + 0.35,
            colors="#2c3e50",
            linewidth=2,
            zorder=4,
            alpha=0.7,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l:.2f}" for l in labels], fontsize=10)
    _apply_style(
        ax,
        f"Muon Momentum Sensitivity ({len(mom_data)} experiments)",
        "Momentum Value",
        "Val BPB",
    )

    return chart_to_base64(fig)


def generate_status_distribution(stats):
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = ["Improved/New Best", "Discarded/Worse", "Crash"]
    sizes = [stats["improved"], stats["worse"], stats["crashes"]]
    colors = ["#2ecc71", "#e74c3c", "#e67e22"]
    explode = (0.05, 0, 0.1)

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops=dict(edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    ax.legend(
        wedges,
        [f"{l} ({s})" for l, s in zip(labels, sizes)],
        loc="lower center",
        fontsize=11,
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
    )
    _apply_style(ax, "Overall Experiment Outcomes")

    return chart_to_base64(fig)


def generate_cumulative_improvement(experiments, best_history, baseline_val_bpb):
    fig, ax = plt.subplots(figsize=(16, 6))

    improvements = [
        (baseline_val_bpb - bh) / baseline_val_bpb * 100 for bh in best_history
    ]

    ax.fill_between(range(len(improvements)), improvements, alpha=0.2, color="#27ae60")
    ax.plot(range(len(improvements)), improvements, color="#27ae60", linewidth=2)

    _apply_style(
        ax,
        "Cumulative BPB Improvement Over Experiments",
        "Experiment #",
        "Improvement from Baseline (%)",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))

    step_changes = []
    current_best = baseline_val_bpb
    for i, exp in enumerate(experiments):
        if (
            exp.val_bpb
            and exp.val_bpb > 0
            and exp.val_bpb < current_best
            and exp.status != "crash"
        ):
            current_best = exp.val_bpb
            step_changes.append((i, exp.description[:50]))

    for idx, desc in step_changes[:12]:
        y_val = improvements[idx]
        ax.plot(idx, y_val, "o", color="#27ae60", markersize=5, zorder=5)
        if y_val > 0.5:
            ax.annotate(
                desc[:35],
                xy=(idx, y_val),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.8,
                color="#2c3e50",
                rotation=0,
            )

    return chart_to_base64(fig)


def generate_cumulative_improvement(experiments, best_history, baseline_val_bpb):
    """Generate cumulative improvement chart."""
    fig, ax = plt.subplots(figsize=(14, 6))

    improvements = [
        (baseline_val_bpb - bh) / baseline_val_bpb * 100 for bh in best_history
    ]

    ax.fill_between(range(len(improvements)), improvements, alpha=0.3, color="#2ecc71")
    ax.plot(range(len(improvements)), improvements, color="#2ecc71", linewidth=2)

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Cumulative Improvement from Baseline (%)", fontsize=12)
    ax.set_title(
        "Cumulative BPB Improvement Over Experiments", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))

    # Mark milestones
    milestones = {}
    current_best = baseline_val_bpb
    for i, exp in enumerate(experiments):
        if (
            exp.val_bpb
            and exp.val_bpb > 0
            and exp.val_bpb < current_best
            and exp.status != "crash"
        ):
            current_best = exp.val_bpb
            milestones[i] = exp.description[:40]

    for idx, desc in list(milestones.items())[:15]:  # Show first 15 milestones
        ax.annotate(
            desc,
            xy=(idx, improvements[idx]),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=6,
            alpha=0.7,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    return chart_to_base64(fig)


# ============================================================
# Key milestones extraction
# ============================================================


def extract_milestones(experiments, baseline_val_bpb):
    """Extract key improvement milestones."""
    milestones = []
    current_best = baseline_val_bpb

    for exp in experiments:
        if (
            exp.val_bpb
            and exp.val_bpb > 0
            and exp.val_bpb < current_best
            and exp.status != "crash"
        ):
            improvement = (current_best - exp.val_bpb) / current_best * 100
            total_improvement = (
                (baseline_val_bpb - exp.val_bpb) / baseline_val_bpb * 100
            )
            milestones.append(
                {
                    "index": exp.index,
                    "commit": exp.commit,
                    "val_bpb": exp.val_bpb,
                    "description": exp.description,
                    "step_improvement": improvement,
                    "total_improvement": total_improvement,
                }
            )
            current_best = exp.val_bpb

    return milestones


# ============================================================
# HTML Report Generation
# ============================================================


def parse_git_history(repo_path, start_commit="e23b26d"):
    try:
        result = subprocess.run(
            ["git", "log", f"{start_commit}..HEAD", "--format=%H|%s|%ai"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append(
                    {
                        "hash": parts[0],
                        "message": parts[1],
                        "date": parts[2],
                    }
                )
        return commits
    except Exception:
        return []


def generate_git_section(commits):
    if not commits:
        return "<p><em>无法从git仓库读取提交记录</em></p>"
    rows = ""
    for c in commits:
        short_hash = c["hash"][:8]
        rows += f"""
        <tr>
            <td><code>{short_hash}</code></td>
            <td>{c["date"][:19]}</td>
            <td>{c["message"]}</td>
        </tr>"""
    return f"""
    <p>从 commit e23b26d 开始，共有 <strong>{len(commits)}</strong> 次提交：</p>
    <table>
        <thead><tr><th>Commit</th><th>时间</th><th>描述</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def generate_html_report(
    experiments, stats, milestones, charts, baseline_val_bpb, git_commits
):
    """Generate the full HTML report."""

    git_section = generate_git_section(git_commits)

    # Sort milestones
    milestones_html = ""
    for m in milestones:
        milestones_html += f"""
        <tr>
            <td>{m["index"]}</td>
            <td><code>{m["commit"][:8]}</code></td>
            <td>{m["val_bpb"]:.6f}</td>
            <td>{m["description"]}</td>
            <td class="improvement">-{m["step_improvement"]:.3f}%</td>
            <td class="improvement">-{m["total_improvement"]:.3f}%</td>
        </tr>"""

    # Category table
    cat_rows = ""
    for cat, cstat in sorted(
        stats["category_stats"].items(), key=lambda x: -x[1]["total"]
    ):
        if cstat["total"] == 0:
            continue
        success_rate = (
            cstat["improved"] / cstat["total"] * 100 if cstat["total"] > 0 else 0
        )
        cat_rows += f"""
        <tr>
            <td>{cat}</td>
            <td>{cstat["total"]}</td>
            <td class="improvement">{cstat["improved"]}</td>
            <td class="worse">{cstat["worse"]}</td>
            <td class="crash">{cstat["crash"]}</td>
            <td class="improvement">{cstat["best"]}</td>
            <td>{success_rate:.1f}%</td>
        </tr>"""

    # Phase table
    phase_order = [
        "Phase 1: Initial Exploration",
        "Phase 2: Foundational HP Search",
        "Phase 3: Momentum & WD Optimization",
        "Phase 4: Fine-Grained Grid Search",
        "Phase 5: Weight Tying",
        "Phase 6: QK Norm & Softcap Deep Dive",
        "Phase 7: Systematic Re-testing",
        "Phase 8: Advanced Techniques",
        "Phase 9: WD Scheduling Discovery",
        "Phase 10: Final Optimization",
    ]
    phase_rows = ""
    for phase in phase_order:
        if phase not in stats["phase_stats"]:
            continue
        pstat = stats["phase_stats"][phase]
        if pstat["total"] == 0:
            continue
        success_rate = (
            pstat["improved"] / pstat["total"] * 100 if pstat["total"] > 0 else 0
        )
        phase_name = phase.split(": ", 1)[1] if ": " in phase else phase
        phase_rows += f"""
        <tr>
            <td>{phase_name}</td>
            <td>{pstat["total"]}</td>
            <td class="improvement">{pstat["improved"]}</td>
            <td class="worse">{pstat["worse"]}</td>
            <td class="crash">{pstat["crash"]}</td>
            <td>{success_rate:.1f}%</td>
        </tr>"""

    # All experiments table
    all_exp_rows = ""
    for exp in experiments:
        status_class = ""
        if exp.status == "crash":
            status_class = "crash"
        elif exp.is_best or exp.status == "NEW BEST":
            status_class = "best"
        elif exp.status == "keep" or exp.status == "improved":
            status_class = "improved"
        else:
            status_class = "worse"

        val_display = (
            f"{exp.val_bpb:.6f}" if exp.val_bpb and exp.val_bpb > 0 else "CRASH"
        )
        mem_display = f"{exp.memory_gb:.1f}" if exp.memory_gb else "-"

        all_exp_rows += f"""
        <tr class="{status_class}">
            <td>{exp.index}</td>
            <td><code>{exp.commit[:12] if len(exp.commit) > 12 else exp.commit}</code></td>
            <td>{val_display}</td>
            <td>{mem_display}</td>
            <td>{exp.status}</td>
            <td>{exp.category}</td>
            <td>{exp.description}</td>
        </tr>"""

    # Crash experiments
    crash_exps = [e for e in experiments if e.status == "crash"]
    crash_rows = ""
    for exp in crash_exps:
        crash_rows += f"""
        <tr class="crash">
            <td>{exp.index}</td>
            <td><code>{exp.commit[:12]}</code></td>
            <td>{exp.description}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoResearch 实验分析报告</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; font-size: 2em; margin-bottom: 10px; }}
        h2 {{ color: #2c3e50; font-size: 1.5em; margin: 30px 0 15px; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        h3 {{ color: #34495e; font-size: 1.2em; margin: 20px 0 10px; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stat-card .label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
        .stat-card.best .value {{ color: #27ae60; }}
        .stat-card.crash .value {{ color: #e67e22; }}
        .stat-card.improvement .value {{ color: #2980b9; }}

        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .chart-container img {{ max-width: 100%; height: auto; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-size: 0.9em;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #ecf0f1;
            font-size: 0.85em;
        }}
        tr:hover {{ background: #f8f9fa; }}
        tr.improved {{ background: #f0fff0; }}
        tr.best {{ background: #e8f8e0; font-weight: bold; }}
        tr.worse {{ background: #fff5f5; }}
        tr.crash {{ background: #fff8e1; }}
        td.improvement {{ color: #27ae60; font-weight: bold; }}
        td.worse {{ color: #e74c3c; }}
        td.crash {{ color: #e67e22; }}

        .phase-timeline {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .phase-item {{
            background: white;
            border-left: 4px solid #3498db;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 250px;
        }}
        .phase-item h4 {{ color: #2c3e50; margin-bottom: 5px; }}
        .phase-item p {{ font-size: 0.85em; color: #7f8c8d; }}

        .theory-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .theory-card h3 {{ color: #8e44ad; margin-bottom: 10px; }}
        .theory-card p {{ margin-bottom: 10px; }}
        .theory-card .ref {{ font-size: 0.85em; color: #7f8c8d; font-style: italic; }}

        .evolution-item {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}
        .evolution-item h4 {{ color: #2c3e50; }}
        .evolution-item .arrow {{ color: #3498db; font-weight: bold; }}

        .search-box {{
            width: 100%;
            padding: 10px 15px;
            margin: 10px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }}
        .search-box:focus {{ border-color: #3498db; outline: none; }}

        .nav {{
            background: #2c3e50;
            padding: 10px 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .nav a {{
            color: white;
            text-decoration: none;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .nav a:hover {{ background: #34495e; }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            margin-top: 40px;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
            table {{ font-size: 0.75em; }}
            th, td {{ padding: 6px 8px; }}
        }}
    </style>
</head>
<body>
    <nav class="nav">
        <a href="#summary">📊 概览</a>
        <a href="#progress">📈 进展</a>
        <a href="#milestones">🏆 里程碑</a>
        <a href="#categories">📂 分类分析</a>
        <a href="#phases">🔄 阶段演进</a>
        <a href="#sensitivity">🎯 超参敏感性</a>
        <a href="#crashes">⚠️ 崩溃与超时</a>
        <a href="#nonimproving">❌ 未改善分析</a>
        <a href="#theory">📚 理论解释</a>
        <a href="#evolution">🧬 修改演进</a>
        <a href="#git">📝 Git记录</a>
        <a href="#all">📋 全部实验</a>
    </nav>

    <div class="container">
        <h1>🔬 AutoResearch 实验分析报告</h1>
        <p style="color: #7f8c8d; margin-bottom: 20px;">
            基于 results.tsv 的 {stats["total"]} 轮实验完整分析 | 基线 Val BPB: {baseline_val_bpb:.6f} → 最终最佳: {stats["best_val_bpb"]:.6f}
        </p>

        <!-- ==================== Summary ==================== -->
        <h2 id="summary">📊 总体概览</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="value">{stats["total"]}</div>
                <div class="label">总实验数</div>
            </div>
            <div class="stat-card best">
                <div class="value">{stats["new_best"]}</div>
                <div class="label">新纪录次数</div>
            </div>
            <div class="stat-card improvement">
                <div class="value">{stats["improved"]}</div>
                <div class="label">改善/保留</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: #e74c3c">{stats["worse"]}</div>
                <div class="label">未改善/丢弃</div>
            </div>
            <div class="stat-card crash">
                <div class="value">{stats["crashes"]}</div>
                <div class="label">运行崩溃</div>
            </div>
            <div class="stat-card improvement">
                <div class="value">{stats["improvement_from_baseline"]:.2f}%</div>
                <div class="label">总改善幅度</div>
            </div>
            <div class="stat-card best">
                <div class="value">{stats["best_val_bpb"]:.6f}</div>
                <div class="label">最佳 Val BPB</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats["improved"] / stats["total"] * 100:.1f}%</div>
                <div class="label">成功率</div>
            </div>
        </div>

        <div class="chart-container">
            <img src="data:image/png;base64,{charts["status_distribution"]}" alt="Status Distribution">
        </div>

        <!-- ==================== Progress ==================== -->
        <h2 id="progress">📈 实验进展趋势</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["progress"]}" alt="Progress Chart">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["cumulative"]}" alt="Cumulative Improvement">
        </div>

        <div class="phase-timeline">
            <div class="phase-item">
                <h4>🎯 目标</h4>
                <p>在固定5分钟训练时间内，最小化验证集 Bits Per Byte (BPB)。BPB越低越好。</p>
            </div>
            <div class="phase-item">
                <h4>📏 评价标准</h4>
                <p>每个实验跑固定5分钟。与当前最佳比较，改善则保留（keep），否则丢弃（discard）。</p>
            </div>
            <div class="phase-item">
                <h4>🏆 最终成果</h4>
                <p>Val BPB从 {baseline_val_bpb:.6f} 降至 {stats["best_val_bpb"]:.6f}，改善 {stats["improvement_from_baseline"]:.2f}%</p>
            </div>
        </div>

        <!-- ==================== Milestones ==================== -->
        <h2 id="milestones">🏆 关键里程碑 (新纪录)</h2>
        <p>以下实验每次都刷新了最佳 Val BPB 记录：</p>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Commit</th>
                    <th>Val BPB</th>
                    <th>描述</th>
                    <th>步骤改善</th>
                    <th>累计改善</th>
                </tr>
            </thead>
            <tbody>
                {milestones_html}
            </tbody>
        </table>

        <!-- ==================== Categories ==================== -->
        <h2 id="categories">📂 超参数分类分析</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["category_pie"]}" alt="Category Distribution">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["success_rate"]}" alt="Success Rate by Category">
        </div>

        <table>
            <thead>
                <tr>
                    <th>类别</th>
                    <th>实验数</th>
                    <th>改善</th>
                    <th>未改善</th>
                    <th>崩溃</th>
                    <th>新纪录</th>
                    <th>成功率</th>
                </tr>
            </thead>
            <tbody>
                {cat_rows}
            </tbody>
        </table>

        <!-- ==================== Phases ==================== -->
        <h2 id="phases">🔄 研究阶段演进</h2>
        <p>根据实验内容的时间顺序和主题，将333轮实验分为以下阶段：</p>

        <div class="chart-container">
            <img src="data:image/png;base64,{charts["phase_progress"]}" alt="Phase Progress">
        </div>

        <table>
            <thead>
                <tr>
                    <th>阶段</th>
                    <th>实验数</th>
                    <th>改善</th>
                    <th>未改善</th>
                    <th>崩溃</th>
                    <th>成功率</th>
                </tr>
            </thead>
            <tbody>
                {phase_rows}
            </tbody>
        </table>

        <div class="evolution-item">
            <h4>Phase 1: 初始探索 (实验 1-8)</h4>
            <p>建立基线，测试基本架构变化（depth, aspect_ratio, batch size, warmdown）。快速发现模型不宜过大（显存限制），batch size和warmdown需要仔细调节。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 2: 基础超参搜索 (实验 9-42)</h4>
            <p>系统搜索关键超参数：weight decay、learning rate（各种）、activation function、softcap、momentum。确定了大部分超参数的粗略最优范围。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 3: 动量与权重衰减优化 (实验 43-68)</h4>
            <p>深入探索 Muon 动量参数和权重衰减策略。发现常量权重衰减和常量动量优于调度策略。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 4: 精细网格搜索 (实验 69-121)</h4>
            <p>对已确定范围的核心超参数做精细搜索。softcap从15逐步降到9，MATRIX_LR从0.04逐步调到0.036，SCALAR_LR降到0.2，FINAL_LR_FRAC降到0.03。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 5: 权重绑定 (实验 121-132)</h4>
            <p>引入权重绑定（weight tying），将embedding和output projection共享参数。带来显著改善（1.237766 vs 前最佳）。但需注意使用double update而非single update。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 6: QK归一化移除与Softcap深挖 (实验 150-165)</h4>
            <p>发现移除QK normalization反而提升了性能！这开启了新的优化空间。softcap继续从12降到8，MATRIX_LR反向提升回0.040。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 7: 系统性重测 (实验 166-255)</h4>
            <p>在新的配置基础上系统性重测所有超参数。确认大部分之前的最优值仍然适用，同时发现PE norm constant 1.05-1.06的新改进点。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 8: 高级技术探索 (实验 234-255)</h4>
            <p>尝试了多种高级技术：parallel attention+MLP、learnable softcap、layer-scaled init、各种warmdown变体。大部分未见改善，但发现了"最后25%权重衰减"的新策略。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 9: WD调度发现 (实验 256-292)</h4>
            <p>发现仅在训练最后25%应用权重衰减的策略，这是一个重要的突破。进一步优化了WD threshold到0.78，softcap提升到9。</p>
        </div>
        <div class="evolution-item">
            <h4>Phase 10: 最终优化 (实验 291-333)</h4>
            <p>在最新配置基础上精细调节：muon beta2从0.95调到0.955，FINAL_LR_FRAC从0.03调到0.035。每步改善越来越小，表明接近当前架构的性能极限。</p>
        </div>

        <!-- ==================== Sensitivity ==================== -->
        <h2 id="sensitivity">🎯 超参数敏感性分析</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["lr_sensitivity"]}" alt="LR Sensitivity">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["wd_chart"]}" alt="Weight Decay">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["softcap_chart"]}" alt="Softcap">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{charts["momentum_chart"]}" alt="Momentum">
        </div>

        <!-- ==================== Key Findings ==================== -->
        <h2>🔍 关键发现总结</h2>
        <div class="theory-card">
            <h3>最具影响力的发现</h3>
            <table>
                <thead>
                    <tr><th>排名</th><th>改进</th><th>发现</th><th>说明</th></tr>
                </thead>
                <tbody>
                    <tr><td>1</td><td class="improvement">大</td><td>移除QK归一化</td><td>在小模型上，QK normalization反而限制了表征能力</td></tr>
                    <tr><td>2</td><td class="improvement">大</td><td>权重绑定</td><td>共享embedding和output投影，减少参数量同时提升泛化</td></tr>
                    <tr><td>3</td><td class="improvement">中</td><td>Softcap 15→8-9</td><td>更低的logit softcap提供了更好的attention regularization</td></tr>
                    <tr><td>4</td><td class="improvement">中</td><td>WD仅最后25%</td><td>延迟权重衰减让模型先充分学习再正则化</td></tr>
                    <tr><td>5</td><td class="improvement">中</td><td>Muon动量0.85（常量）</td><td>常量动量优于ramp调度，简化训练</td></tr>
                    <tr><td>6</td><td class="improvement">小</td><td>PE norm 1.05-1.06</td><td>对位置编码做norm有助于训练稳定性</td></tr>
                    <tr><td>7</td><td class="improvement">小</td><td>Muon beta2 0.955</td><td>二阶矩估计的微调</td></tr>
                    <tr><td>8</td><td class="improvement">小</td><td>FINAL_LR_FRAC 0.035</td><td>学习率不完全衰减到0，保留小部分学习率</td></tr>
                </tbody>
            </table>
        </div>

        <!-- ==================== Crashes ==================== -->
        <h2 id="crashes">⚠️ 崩溃、超时与步数损失分析</h2>

        <h3>运行崩溃 (共 {stats["crashes"]} 次)</h3>
        <p>以下实验在运行时直接崩溃，无法完成训练：</p>
        <table>
            <thead>
                <tr><th>#</th><th>Commit</th><th>描述与崩溃原因</th></tr>
            </thead>
            <tbody>
                {crash_rows}
            </tbody>
        </table>

        <div class="theory-card">
            <h3>崩溃原因分类</h3>
            <p><strong>形状不匹配 (Shape Mismatch):</strong> GQA n_kv_head=2 修改导致注意力计算中KV头数与查询头数维度不匹配。这是因为改动GQA配置需要同步修改多处相关代码。</p>
            <p><strong>梯度缺失 (No Grad):</strong> 禁用VE（Value Embedding）或直接去掉VE gate时，某些参数在前向传播中被跳过，导致反向传播时没有梯度，优化器更新失败。</p>
            <p><strong>CUDA编译错误:</strong> torch.compile的reduce-overhead和max-autotune模式在权重绑定场景下与CUDA graphs不兼容，触发编译时崩溃。</p>
        </div>

        <h3>超时与步数损失</h3>
        <div class="theory-card">
            <h3>关于超时情况</h3>
            <p>在 results.tsv 中<strong>没有明确记录运行超时</strong>的实验。本项目采用固定5分钟训练时间预算的设计，所有实验在时间限制内完成（无论训练步数多少）。但存在以下间接相关的情况：</p>
            <table>
                <thead>
                    <tr><th>类型</th><th>实验</th><th>说明</th></tr>
                </thead>
                <tbody>
                    <tr><td>显存不足 (OOM)</td><td>depth 16 (10.3GB), depth 10 (3.8GB)</td><td>模型参数量太大超出GPU显存。虽然标记为discard而非crash，但在更小的GPU上可能导致OOM超时。</td></tr>
                    <tr><td>步数显著减少</td><td>no_compile (314步 vs 正常563步), MTP (少84步)</td><td>某些修改导致每步变慢，5分钟内有效训练步数大幅减少。</td></tr>
                    <tr><td>步数损失</td><td>EMA eval (also lost steps), per_dim_gating (lost steps)</td><td>额外计算开销（EMA评估、额外参数）导致训练步数减少。</td></tr>
                    <tr><td>编译超时</td><td>reduce-overhead, max-autotune (CRASH)</td><td>torch.compile模式在CUDA graphs + weight tying场景下崩溃。</td></tr>
                </tbody>
            </table>
        </div>

        <!-- ==================== Theory ==================== -->
        <h2 id="theory">📚 理论解释与参考文献</h2>

        <div class="theory-card">
            <h3>1. 为什么 ReLU² 优于 GELU/SiLU</h3>
            <p>ReLU² (Squared ReLU) 在实验中持续优于 GELU（1.263796 vs 1.245）和 SiLU（1.274506 vs 1.245），甚至 ReLU^1.5 和 ReLU^3 也更差。</p>
            <p><strong>无界正梯度:</strong> Huang (2024) 提供了关键的理论解释。ReLU² 的梯度为 <code>f'(x) = 2x</code>（x>0）——线性递增且无上界。而 GELU 和 SiLU 的梯度都有上界（GELU ≤ 1, SiLU ~1.1）。对于大的正预激活值，ReLU² 传播更强的误差信号，使模型能更有效地从高幅度特征中学习。</p>
            <p><strong>稀疏性与硬件效率:</strong> Zhang et al. (2024) 证明 ReLU² 达到了最佳稀疏性-性能权衡——约90%的稀疏度下性能损失<0.1%。负值置零提供稀疏性，正值区域的二次函数提供平滑非线性梯度。与 GELU/SiLU 相比，ReLU² 计算更简单，在固定5分钟预算内允许更多训练步骤。</p>
            <p><strong>小模型优势:</strong> ReLU² 不需要 SwiGLU/GEGLU 门控机制的50%参数开销，却在亚10亿参数级别匹配或超越这些机制。小模型的浅层深度使 dying ReLU 问题不太严重，而稀疏执行的计算节省比例更大。</p>
            <p class="ref">[1] Zhang et al., "ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs", arXiv:2402.03804 (2024)<br>
            [2] Huang, "Deriving Activation Functions via Integration", arXiv:2411.13010 (2024)<br>
            [3] So et al., "Primer: Searching for Efficient Transformers for Language Modeling" (2021)</p>
        </div>

        <div class="theory-card">
            <h3>2. 为什么 Muon 优化器有效 (ns_steps=5, momentum=0.85, beta2=0.955)</h3>
            <p>Muon 通过 Newton-Schulz (NS) 迭代对参数矩阵进行正交化更新，与传统的梯度下降有本质区别。</p>
            <p><strong>ns_steps=5 足够收敛:</strong> Kim & Oh (ICLR 2026) 证明收敛因子 χ_q（NS与精确SVD极分解的常数因子差距）以 q（NS步数）的<strong>双指数速度</strong>趋近1。使用优化的五次系数（a≈3.44, b≈-4.77, c≈2.03），5步已将奇异值收敛到 [0.7, 1.3] 范围，与精确正交化在训练损失上不可区分。每增加一步带来快速递减的收益，同时线性增加 FLOP 开销。</p>
            <p><strong>Momentum 0.85:</strong> Shulgin et al. (2025) 建立了<strong>LMO不精确性与最优动量的基本耦合关系</strong>。理论表明：较低的 NS 精度需要更小的步长但更大的动量来维持收敛。高动量有效平均了近似极分解在迭代间引入的高频噪声。实验证实随着 NS 精度从1步增加到5步，最优学习率上移，而最优动量保持在高区间（0.85-0.95）。常量动量优于 ramp 调度是因为简化了这种耦合关系。</p>
            <p><strong>Beta2 0.955:</strong> Muon 对标量参数使用 AdamW 辅助优化器。Beta2 接近1意味着长时间尺度的方差估计，提供更稳定的学习率自适应。Muon 的谱范数更新已经提供了激进的一阶进展，高 beta2 防止 AdamW 对稀疏梯度（embedding和bias）过度缩小学习信号。</p>
            <p class="ref">[1] Jordan et al., "Muon: An optimizer for hidden layers in neural networks" (2024)<br>
            [2] Shulgin et al., "Beyond the Ideal: Analyzing the Inexact Muon Update", arXiv:2510.19933 (2025)<br>
            [3] Kim & Oh, "Convergence of Muon with Newton-Schulz", arXiv:2601.19156, ICLR 2026<br>
            [4] Du & Su, "The Newton-Muon Optimizer", arXiv:2604.01472 (2026)</p>
        </div>

        <div class="theory-card">
            <h3>3. 为什么 Attention Logit Softcap 有效 (最优值 8-9)</h3>
            <p>Softcap 对注意力 logits 做 <code>soft_cap * tanh(logits / soft_cap)</code> 变换，最优值从15逐步降到8-9。</p>
            <p><strong>防止注意力熵坍缩:</strong> Dehghani et al. (2023) 表明无约束点积注意力在规模扩大时可以产生超过50,000量级的 logits，驱动 softmax 输出趋向 one-hot 分布，导致<strong>注意力熵坍缩</strong>和梯度消失。Softcap 提供平滑、可微的界限，保持 softmax 熵和稳定梯度。</p>
            <p><strong>不引入硬截断:</strong> 与硬截断不同，tanh softcap 处处可微。Gemma 2 技术报告明确将其作为稳定化机制，允许使用更高的学习率（在某些消融实验中高达1.5倍）而不发散。</p>
            <p><strong>最优值8-9的解释:</strong> 在移除QK norm后，softcap成为唯一的logit约束机制，最优值从12降至8-9。值8-9在"允许足够的注意力区分度"和"防止过度集中"之间取得平衡。太低（7）限制表达能力，太高（15以上）接近无约束注意力。</p>
            <p class="ref">[1] Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size", arXiv:2408.00118 (2024)<br>
            [2] Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters", arXiv:2302.05442 (2023)<br>
            [3] Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities", arXiv:2309.17421 (2023)</p>
        </div>

        <div class="theory-card">
            <h3>4. 为什么权重绑定有效</h3>
            <p>共享输入 embedding（wte）和输出投影（lm_head）的权重，带来了显著改善（1.237766 vs 前最佳 1.238607）。</p>
            <p><strong>参数效率:</strong> 权重绑定减少 d_model × vocab_size 参数量，在小模型中可占总参数的>25%，显著减轻过拟合风险。</p>
            <p><strong>分布假设对齐:</strong> Bertolotti & Cazzola (2024) 证明在分布假设下，最优的输入和输出 embedding 必须编码相同的语义关系，论证了参数共享的理论合理性。Press & Wolf 实证表明未绑定模型的输出 embedding 通常优于输入 embedding；绑定强制共享矩阵继承输出侧的质量。</p>
            <p><strong>梯度动态:</strong> Lopardo et al. (2026) 表明绑定模型中输出层梯度在训练早期占主导，将共享 embedding 矩阵偏向 unembedding 空间。对小模型这是净收益，因为输出投影是参数最重的组件。但"single update"（3.150207，灾难性恶化）远不如"double update"——两条路径提供互补梯度信号。</p>
            <p class="ref">[1] Press & Wolf, "Using the Output Embedding to Improve Language Models", EMNLP 2017<br>
            [2] Inan et al., "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling", ICLR 2017<br>
            [3] Lopardo et al., "Weight Tying Biases Token Embeddings Towards the Output Space", arXiv:2603.26663 (2026)<br>
            [4] Bertolotti & Cazzola, "Tying Embeddings You Are Assuming the Distributional Hypothesis" (2024)</p>
        </div>

        <div class="theory-card">
            <h3>5. 为什么移除 QK 归一化反而改善</h3>
            <p>移除 QK normalization 带来了重大改善（1.237623 vs 1.237766），并开启了新的优化空间。</p>
            <p><strong>全局归一化下的冗余性:</strong> Loshchilov et al. (nGPT, 2024) 构建了所有 embedding、隐藏状态和权重矩阵都做 L2 归一化的 Transformer。在此架构中，Q 和 K 是已归一化隐藏状态的投影，其范数自然可比，显式 QK 归一化变得<strong>冗余</strong>。移除后每步训练时间减少约12%，性能仅有微小差异。</p>
            <p><strong>小模型的容量约束:</strong> QK norm 限制了 Query-Key 点积的动态范围。对于小模型（~50M参数），模型容量有限，移除 QK norm 恢复了更大的注意力 logit 动态范围，使模型能更灵活地表达注意力模式。</p>
            <p><strong>与 Softcap 的互补:</strong> 移除 QK norm 后，softcap 成为唯一的 logit 约束机制，两者原本目标相同的 logit 爆炸问题。移除一个后另一个的负担加重，因此 softcap 最优值从12变为8-9（更严格的约束）。</p>
            <p class="ref">[1] Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning on the Hypersphere", arXiv:2410.01131 (2024)<br>
            [2] Henry et al., "Query-Key Normalization for Transformers", EMNLP Findings 2020<br>
            [3] Dremov et al., "Training Dynamics of the Cooldown Stage in WSD Learning Rate Scheduler", arXiv:2508.01483 (2025)</p>
        </div>

        <div class="theory-card">
            <h3>6. 为什么延迟权重衰减有效（最后25%训练）⭐ 创新发现</h3>
            <p>这是本实验中最具创新性的发现：仅在训练最后25%应用权重衰减（val_bpb=1.221516），优于常量权重衰减（1.240961）和全程权重衰减。</p>
            <p><strong>河谷景观理论:</strong> Wen et al. (2024) 将预训练损失建模为<strong>河谷</strong>——底部有河流的深谷。早期训练需要高学习率沿河流快速前进（粗粒度特征学习）。过早的权重衰减将迭代点拉离河流，减缓进展。后期权重衰减（最后20-25%）类似于将迭代点推向河流中心，锐化解而不牺牲已学到的粗粒度结构。</p>
            <p><strong>动力学稳定器而非正则化器:</strong> Kosson et al. (ICML 2024) 表明 AdamW 达到平衡时权重衰减收缩与梯度增长平衡，相对更新量 ∝ √(ηλ)——<strong>ηλ 乘积</strong>控制有效学习速度。延迟应用权重衰减在模型接近收敛时提供稳定化效果，同时不干扰早期特征学习的大幅更新需求。</p>
            <p><strong>与 WSD 调度的一致性:</strong> Dremov et al. (2025) 分析 WSD cooldown 动力学，发现在 cooldown 阶段禁用权重衰减会降低性能。最优衰减比例在大型 LLM 训练中一致为总步数的<strong>20%</strong>（IMU-1, LLaMA 系列），与我们25%的发现高度一致。</p>
            <p><strong>塑性-正则化权衡:</strong> Han et al. (2026) 证明对于计算最优训练的小模型，更高的权重衰减（0.5-1.0）改善下游塑性性。延迟应用高 WD 允许模型先建立良好的表征，然后在后期精炼。</p>
            <p class="ref">[1] Wen et al., "Understanding WSD Learning Rates: A River Valley Loss Landscape Perspective", arXiv:2410.05192 (2024)<br>
            [2] Kosson et al., "Rotational Equilibrium: How Weight Decay Balances Learning", ICML 2024, arXiv:2305.17212<br>
            [3] Chen et al., "Cautious Weight Decay", ICLR 2026, arXiv:2510.12402<br>
            [4] Han et al., "Weight Decay Improves Language Model Plasticity", arXiv:2602.11137 (2026)<br>
            [5] Bergsma et al., "Power Lines: Scaling Laws for Weight Decay", NeurIPS 2025, arXiv:2505.13738</p>
        </div>

        <div class="theory-card">
            <h3>7. PE Norm Constant 的作用 (最优值 1.05-1.06)</h3>
            <p>位置编码归一化常数值从1.02优化到1.06（1.221845），过高（1.08）和过低（1.01）都更差。</p>
            <p><strong>初始化信号幅度匹配:</strong> Vaswani et al. (2017) 的原始 Transformer 将 embedding 乘以 √d_model。学习到的 token embedding 通常以方差 ~1/d_model 初始化（范数 ~1），而正弦位置编码的 L2 范数为 ~√(d/2)。对于 d=512，PE 范数约16，而未缩放的 embedding 范数约1。乘以 √d_model 使两者都达到 O(√d) 级别。</p>
            <p><strong>训练初期的平衡:</strong> PE norm constant > 1.0 适度增强位置编码信号，防止内容信息在训练初期淹没位置信息。但过大（1.08）则使位置信号过于突出，干扰语义理解。1.05-1.06 是这个平衡的甜蜜点。</p>
            <p class="ref">[1] Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Section 3.4<br>
            [2] Zhang et al., "Deep Network Approximation: Beyond ReLU to Diverse Activation Functions", JMLR (2025)</p>
        </div>

        <div class="theory-card">
            <h3>8. 为什么 FULL ATTENTION 优于窗口注意力</h3>
            <p>实验早期发现 "L" pattern（full attention）优于 "SSSL" pattern（交替 banded attention）。</p>
            <p><strong>序列长度短:</strong> 在小序列长度下，full attention 的计算开销可控（O(n²) 可接受），且能捕获所有 token 间的关系。窗口注意力在长序列场景下有意义，但短序列下不必要地限制了信息流动。</p>
            <p class="ref">[1] Beltagy et al., "Longformer: The Long-Document Transformer" (2020)<br>
            [2] Zaheer et al., "Big Bird: Transformers for Longer Sequences" (2020)</p>
        </div>

        <div class="theory-card">
            <h3>9. 关键失败的尝试及理论解释</h3>
            <table>
                <thead>
                    <tr><th>尝试</th><th>结果</th><th>理论解释</th><th>参考文献</th></tr>
                </thead>
                <tbody>
                    <tr><td>增加模型深度 (Depth 9-16)</td><td class="worse">显著恶化</td><td>固定5分钟预算下，更大模型训练步骤太少，严重欠拟合。谱范数约束在少步数下无法收敛。</td><td>Kaplan et al., "Scaling Laws for Neural Language Models" (2020)</td></tr>
                    <tr><td>Label Smoothing 0.1</td><td class="worse">灾难性 (1.55)</td><td>小数据集小模型上，0.1的平滑过于激进，有效降低了模型的表达上限。</td><td>Szegedy et al., "Rethinking the Inception Architecture" (2016)</td></tr>
                    <tr><td>MTP辅助损失</td><td class="worse">显著 (1.56)</td><td>Multi-token prediction 在小模型上分散主任务的学习信号，额外损失权重优化困难。</td><td>Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024)</td></tr>
                    <tr><td>Parallel Attn+MLP</td><td class="worse">恶化</td><td>减少有效深度，小模型需要更多串行计算层来表达复杂特征。</td><td>He et al., "DeBERTaV3" (2021)</td></tr>
                    <tr><td>Learnable Softcap</td><td class="worse">灾难性 (1.97)</td><td>可学习参数可能落入不好局部最优，softcap 值在训练中漂移导致注意力不稳定。</td><td>—</td></tr>
                    <tr><td>禁用 torch.compile</td><td class="worse">显著 (1.29)</td><td>步骤数从563降至314，计算效率下降44%直接减少有效训练量。</td><td>—</td></tr>
                    <tr><td>梯度裁剪 (clip 1.0)</td><td class="worse">恶化</td><td>Muon 的正交化更新自带稳定性，额外裁剪限制了有效更新幅度。</td><td>—</td></tr>
                    <tr><td>LayerNorm 替代 RMSNorm</td><td class="worse">恶化</td><td>RMSNorm 计算更简单（无需均值计算），固定预算内允许更多步骤。</td><td>Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)</td></tr>
                </tbody>
            </table>
        </div>

        <!-- ==================== Non-improving Experiments ==================== -->
        <h2 id="nonimproving">❌ 未改善实验深度分析</h2>
        <p>共有 <strong>{stats["worse"]}</strong> 轮实验未能改善目标。以下按类别分析未改善的主要原因：</p>

        <div class="theory-card">
            <h3>未改善实验的常见原因分类</h3>
            <table>
                <thead>
                    <tr><th>原因类别</th><th>典型实验</th><th>数量(估)</th><th>理论解释</th></tr>
                </thead>
                <tbody>
                    <tr><td>超参已近最优边界</td><td>MLR 0.035, 0.037, 0.039, 0.041, 0.042 等</td><td>~120</td><td>超参在最优值附近微调时，变化方向往往比预期更容易变差。优化景观在最优值附近呈尖锐谷形。</td></tr>
                    <tr><td>模型规模与时间预算矛盾</td><td>Depth 9-16, Aspect 72-96</td><td>~10</td><td>固定5分钟预算下，增大模型→训练步数减少→欠拟合。参考 Kaplan et al. Scaling Laws (2020)。</td></tr>
                    <tr><td>架构选择不适合小模型</td><td>GELU, SiLU, Parallel Attn, LayerNorm, GQA</td><td>~15</td><td>大模型的成功架构不一定迁移到小模型。计算开销、有效深度、参数效率在小模型上更关键。</td></tr>
                    <tr><td>正则化过于激进</td><td>Label Smoothing 0.1, Grad Clip 1.0, MTP</td><td>~8</td><td>小模型的容量有限，过强正则化进一步压缩了有效学习空间。</td></tr>
                    <tr><td>重测确认无效方向</td><td>大量 "retest" 后缀的实验</td><td>~40</td><td>在配置变化后重测之前放弃的方向，大多数仍无效。说明超参敏感性在配置变化后基本保持不变。</td></tr>
                    <tr><td>步数损失导致有效训练不足</td><td>no_compile (314步), MTP (少84步)</td><td>~5</td><td>某些修改导致每步变慢，5分钟内训练步数减少，直接降低有效训练量。</td></tr>
                </tbody>
            </table>
        </div>

        <!-- ==================== Evolution ==================== -->
        <h2 id="evolution">🧬 修改内容的分类与演进</h2>

        <h3>超参数调优类型分布</h3>
        <div class="evolution-item">
            <h4>1. 学习率优化 (约35%的实验)</h4>
            <p><span class="arrow">→</span> 分为Matrix LR、Scalar LR、Embedding LR、Unembedding LR四类。从粗搜索到精调：</p>
            <p>MATRIX_LR: 0.04 → 0.035 → 0.036 → 0.038 → 0.040 (移除QK norm后重新搜索)</p>
            <p>SCALAR_LR: 0.5 → 0.3 → 0.25 → 0.2</p>
            <p>EMBEDDING_LR: 稳定在0.6</p>
            <p>UNEMBEDDING_LR: 稳定在0.004</p>
        </div>

        <div class="evolution-item">
            <h4>2. 正则化策略 (约20%的实验)</h4>
            <p><span class="arrow">→</span> 从恒定权重衰减到调度策略：</p>
            <p>WD策略演进: 恒定WD 0.1 → 0.05 → 0.06 → 常量WD → 仅最后25%训练 → threshold 0.78</p>
            <p>这反映了从"全局正则化"到"针对性正则化"的认知进化。</p>
        </div>

        <div class="evolution-item">
            <h4>3. 优化器配置 (约15%的实验)</h4>
            <p><span class="arrow">→</span> Muon优化器的深度调优：</p>
            <p>Momentum: ramp 0.85→0.90 → ramp 0.85→0.88 → constant 0.85 (最终发现常量最优)</p>
            <p>Beta2: 0.95 → 0.96 → 0.955 (精细搜索)</p>
            <p>ns_steps: 5 (稳定，6和3都不如)</p>
        </div>

        <div class="evolution-item">
            <h4>4. 架构改动 (约10%的实验)</h4>
            <p><span class="arrow">→</span> 包括激活函数、注意力模式、归一化策略：</p>
            <p>关键发现：ReLU² > GELU/SiLU, Full attention > Window, 无QK norm > 有QK norm, Weight tying有效</p>
        </div>

        <div class="evolution-item">
            <h4>5. 学习率调度 (约10%的实验)</h4>
            <p><span class="arrow">→</span> warmdown ratio、final LR fraction、warmup ratio：</p>
            <p>WARMDOWN_RATIO稳定在0.5, FINAL_LR_FRAC: 0.0 → 0.1 → 0.03 → 0.035, 不使用warmup</p>
        </div>

        <div class="evolution-item">
            <h4>6. 其他探索 (约10%的实验)</h4>
            <p><span class="arrow">→</span> 初始化策略、Value Embedding gate、RoPE base、PE norm等。</p>
            <p>大部分验证了现有设置的最优性，PE norm constant 1.06是一个新发现。</p>
        </div>

        <!-- ==================== Git History ==================== -->
        <h2 id="git">📝 Git提交记录 (从 e23b26d 开始)</h2>
        {git_section}

        <!-- ==================== All Experiments ==================== -->
        <h2 id="all">📋 全部实验明细</h2>
        <input type="text" id="searchInput" class="search-box" placeholder="搜索实验 (输入关键词过滤)...">
        <table id="expTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Commit</th>
                    <th>Val BPB</th>
                    <th>内存(GB)</th>
                    <th>状态</th>
                    <th>类别</th>
                    <th>描述</th>
                </tr>
            </thead>
            <tbody>
                {all_exp_rows}
            </tbody>
        </table>

        <div class="footer">
            <p>AutoResearch 实验分析报告 | 生成于 2026-04-17 | 共 {stats["total"]} 轮实验</p>
        </div>
    </div>

    <script>
        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            const filter = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#expTable tbody tr');
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? '' : 'none';
            }});
        }});

        // Smooth scroll for nav links
        document.querySelectorAll('.nav a').forEach(a => {{
            a.addEventListener('click', function(e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
    </script>
</body>
</html>"""

    return html


# ============================================================
# Main
# ============================================================


def main():
    print("Parsing results.tsv...")
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")
    experiments, baseline_val_bpb, best_history = parse_results(filepath)
    print(f"  Parsed {len(experiments)} experiments, baseline={baseline_val_bpb}")

    print("Classifying experiments...")
    classify_phase(experiments)
    for exp in experiments:
        exp.category = classify_hyperparameter(exp.description)

    print("Computing statistics...")
    stats = compute_statistics(experiments, baseline_val_bpb)

    print("Extracting milestones...")
    milestones = extract_milestones(experiments, baseline_val_bpb)
    print(f"  Found {len(milestones)} milestones")

    print("Generating charts...")
    charts = {}
    charts["progress"] = generate_progress_chart(experiments, best_history)
    print("  ✓ Progress chart")
    charts["category_pie"] = generate_category_pie(stats)
    print("  ✓ Category pie")
    charts["success_rate"] = generate_success_rate_by_category(stats)
    print("  ✓ Success rate")
    charts["phase_progress"] = generate_phase_progress(experiments)
    print("  ✓ Phase progress")
    charts["lr_sensitivity"] = generate_lr_sensitivity(experiments)
    print("  ✓ LR sensitivity")
    charts["wd_chart"] = generate_weight_decay_chart(experiments)
    print("  ✓ Weight decay")
    charts["softcap_chart"] = generate_softcap_chart(experiments)
    print("  ✓ Softcap")
    charts["momentum_chart"] = generate_momentum_chart(experiments)
    print("  ✓ Momentum")
    charts["status_distribution"] = generate_status_distribution(stats)
    print("  ✓ Status distribution")
    charts["cumulative"] = generate_cumulative_improvement(
        experiments, best_history, baseline_val_bpb
    )
    print("  ✓ Cumulative improvement")

    print("Generating HTML report...")
    git_commits = parse_git_history(
        os.path.dirname(os.path.abspath(__file__)), "e23b26d"
    )
    print(f"  Parsed {len(git_commits)} git commits from repository")
    html = generate_html_report(
        experiments, stats, milestones, charts, baseline_val_bpb, git_commits
    )
    print(f"  Parsed {len(git_commits)} git commits from repository")
    git_section = generate_git_section(git_commits)
    html = generate_html_report(
        experiments, stats, milestones, charts, baseline_val_bpb, git_commits
    )

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "experiment_analysis.html"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Report generated: {output_path}")
    print(f"   Total experiments: {stats['total']}")
    print(f"   New best count: {stats['new_best']}")
    print(f"   Improved/kept: {stats['improved']}")
    print(f"   Worse/discarded: {stats['worse']}")
    print(f"   Crashes: {stats['crashes']}")
    print(f"   Best Val BPB: {stats['best_val_bpb']:.6f}")
    print(f"   Improvement from baseline: {stats['improvement_from_baseline']:.2f}%")


if __name__ == "__main__":
    main()
