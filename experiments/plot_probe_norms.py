import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _finite(values: List[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        v = float(v)
        if math.isfinite(v):
            out.append(v)
    return out


def _parse_after_from_label(label: str) -> Optional[int]:
    label = label.strip()
    if not label.startswith("after="):
        return None
    rest = label[len("after=") :]
    num = ""
    for ch in rest:
        if ch.isdigit() or (ch == "-" and not num):
            num += ch
        else:
            break
    if not num or not num.lstrip("-").isdigit():
        return None
    return int(num)


def _linear_fit(x: List[float], y: List[float]) -> Optional[Dict[str, float]]:
    if len(x) != len(y):
        raise ValueError("x/y length mismatch")
    if len(x) < 2:
        return None

    x_mean = mean(x)
    y_mean = mean(y)
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - x_mean
        dy = yi - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx == 0.0:
        return None

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    if syy == 0.0:
        r2 = 0.0
    else:
        sse = 0.0
        for xi, yi in zip(x, y):
            y_hat = slope * xi + intercept
            err = yi - y_hat
            sse += err * err
        r2 = 1.0 - (sse / syy)
    r = sxy / math.sqrt(sxx * syy) if (sxx > 0.0 and syy > 0.0) else 0.0

    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "r": float(r)}


def _fit_text(*, xlabel: str, ylabel: str, fit: Dict[str, float]) -> str:
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    r2 = float(fit["r2"])
    r = float(fit["r"])
    sign = "+" if intercept >= 0 else "-"
    return (
        f"{ylabel} = {slope:.4g} * {xlabel} {sign} {abs(intercept):.4g}\n"
        f"R² = {r2:.3f}  r = {r:.3f}"
    )


def _parse_case_id(raw: str) -> Any:
    raw = raw.strip()
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


def _parse_case_id_list(spec: Optional[str]) -> Optional[List[Any]]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return [_parse_case_id(x) for x in spec.split(",") if x.strip()]


def _build_case_series(
    checkpoints: List[Dict[str, Any]], *, y_key: str
) -> Dict[Any, List[Tuple[int, float, float]]]:
    """
    Returns mapping: case_id -> [(after_edits, weight_norm, y), ...] sorted by after_edits.
    """
    out: Dict[Any, List[Tuple[int, float, float]]] = {}
    for ckpt in checkpoints:
        after = int(ckpt["after_edits"])
        w = ckpt.get("weight_norm")
        if w is None:
            continue
        w = float(w)
        if not math.isfinite(w):
            continue

        for s in ckpt.get("samples", []):
            case_id = s.get("case_id")
            y = s.get(y_key)
            if case_id is None or y is None:
                continue
            y = float(y)
            if not math.isfinite(y):
                continue
            out.setdefault(case_id, []).append((after, w, y))

    for case_id in list(out.keys()):
        out[case_id].sort(key=lambda t: t[0])
        if len(out[case_id]) < 2:
            out.pop(case_id, None)
    return out


def _select_case_ids_for_paths(
    series: Dict[Any, List[Tuple[int, float, float]]],
    *,
    k: int,
    strategy: str,
    seed: int,
) -> List[Any]:
    if k <= 0:
        return []
    if not series:
        return []

    strategy = strategy.strip().lower()
    if strategy == "random":
        rng = random.Random(seed)
        keys = list(series.keys())
        rng.shuffle(keys)
        return keys[:k]

    if strategy == "max_delta":
        scored: List[Tuple[float, Any]] = []
        for case_id, pts in series.items():
            pts = sorted(pts, key=lambda t: t[0])
            delta = abs(float(pts[-1][2]) - float(pts[0][2]))
            scored.append((delta, case_id))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [cid for _, cid in scored[:k]]

    if strategy == "max_var":
        scored = []
        for case_id, pts in series.items():
            ys = [float(y) for _, _, y in pts]
            scored.append((pstdev(ys) if len(ys) > 1 else 0.0, case_id))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [cid for _, cid in scored[:k]]

    raise ValueError(f"Unknown --path_strategy {strategy!r}; expected max_var|max_delta|random")


def _summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    xs = sorted(float(x) for x in values)
    return {
        "n": int(len(xs)),
        "mean": float(mean(xs)),
        "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(xs[0]),
        "p05": _percentile(xs, 0.05),
        "p25": _percentile(xs, 0.25),
        "p50": _percentile(xs, 0.50),
        "p75": _percentile(xs, 0.75),
        "p95": _percentile(xs, 0.95),
        "max": float(xs[-1]),
    }


def _parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _resolve_run_dir(path: Path) -> Tuple[Path, Path, Optional[Path]]:
    """
    Returns (run_dir, jsonl_path, config_path_or_None).

    - If `path` is a file: assumes it's the JSONL file.
    - If `path` is a directory:
        - If it contains probe_norms.jsonl: uses it as run_dir.
        - Else if it contains run_XXX directories: uses the max run_XXX.
    """

    if path.is_file():
        run_dir = path.parent
        jsonl_path = path
        cfg = run_dir / "config.json"
        return run_dir, jsonl_path, (cfg if cfg.exists() else None)

    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    jsonl_path = path / "probe_norms.jsonl"
    if jsonl_path.exists():
        cfg = path / "config.json"
        return path, jsonl_path, (cfg if cfg.exists() else None)

    run_dirs = [
        p
        for p in path.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[-1].isnumeric()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"Could not find probe_norms.jsonl or run_*/ under: {path}")

    run_dir = max(run_dirs, key=lambda p: int(p.name.split("_")[-1]))
    jsonl_path = run_dir / "probe_norms.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing probe_norms.jsonl under: {run_dir}")
    cfg = run_dir / "config.json"
    return run_dir, jsonl_path, (cfg if cfg.exists() else None)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def _extract_checkpoint_arrays(ckpt: Dict[str, Any]) -> Dict[str, List[float]]:
    samples = ckpt.get("samples", [])
    v0 = [float(s["v0_norm"]) for s in samples]
    vstar = [float(s["v_star_norm_orig"]) for s in samples]
    ratio = [vs / v if v != 0 else float("inf") for v, vs in zip(v0, vstar)]
    diff = [vs - v for v, vs in zip(v0, vstar)]
    return {"v0_norm": v0, "v_star_norm_orig": vstar, "ratio": ratio, "diff": diff}


def _pick_focus_after_edits(checkpoints: List[Dict[str, Any]], focus: str) -> int:
    if not checkpoints:
        raise ValueError("No checkpoints found")
    focus = focus.strip().lower()
    if focus == "last":
        return int(checkpoints[-1]["after_edits"])
    if focus == "first":
        return int(checkpoints[0]["after_edits"])
    return int(focus)


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except Exception:
        return None


def _plot_weight_norm(checkpoints: List[Dict[str, Any]], out_path: Path, *, title: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs = [int(c["after_edits"]) for c in checkpoints]
    ys = [float(c["weight_norm"]) for c in checkpoints]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_xlabel("after_edits")
    ax.set_ylabel("||W_v|| (Frobenius)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_hist_overlay(
    *,
    series: List[Tuple[str, List[float]]],
    out_path: Path,
    bins: int,
    title: str,
    xlabel: str,
    density: bool,
    style: str,
    legend: str,
    cmap: str,
    alpha: float,
    linewidth: float,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    if bins <= 0:
        raise ValueError("--bins must be > 0")

    clean_series: List[Tuple[str, List[float]]] = []
    for label, values in series:
        cleaned = _finite(values)
        if cleaned:
            clean_series.append((label, cleaned))
    if not clean_series:
        return False

    all_values = [v for _, values in clean_series for v in values]
    x_min = min(all_values)
    x_max = max(all_values)
    if x_max == x_min:
        x_max = x_min + 1e-6
    bin_edges = [x_min + i * (x_max - x_min) / bins for i in range(bins + 1)]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    style = style.strip().lower()
    if style not in {"fill", "step"}:
        raise ValueError(f"Unsupported hist style {style!r}; expected fill|step")
    legend = legend.strip().lower()
    if legend not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported legend mode {legend!r}; expected auto|on|off")

    use_colorbar = style == "step" and legend == "auto" and len(clean_series) > 10
    after_vals = [_parse_after_from_label(lbl) for lbl, _ in clean_series] if use_colorbar else []
    if use_colorbar and any(v is None for v in after_vals):
        use_colorbar = False

    if use_colorbar:
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        after_vals_i = [int(v) for v in after_vals if v is not None]
        norm = colors.Normalize(vmin=min(after_vals_i), vmax=max(after_vals_i))
        cmap_obj = plt.get_cmap(cmap)
        for (label, values), after in zip(clean_series, after_vals_i):
            ax.hist(
                values,
                bins=bin_edges,
                density=density,
                histtype="step",
                linewidth=linewidth,
                color=cmap_obj(norm(after)),
            )

        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="after_edits")
    else:
        for label, values in clean_series:
            if style == "step":
                ax.hist(
                    values,
                    bins=bin_edges,
                    density=density,
                    histtype="step",
                    linewidth=linewidth,
                    label=label,
                )
            else:
                ax.hist(
                    values,
                    bins=bin_edges,
                    alpha=alpha,
                    density=density,
                    label=label,
                )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    ax.grid(True, alpha=0.25)
    if legend == "on" or (legend == "auto" and not use_colorbar and len(clean_series) <= 10):
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_ecdf_overlay(
    *,
    series: List[Tuple[str, List[float]]],
    out_path: Path,
    title: str,
    xlabel: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    for label, values in series:
        xs = sorted(values)
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=label, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_boxplot(
    *,
    values_by_label: List[Tuple[str, List[float]]],
    out_path: Path,
    title: str,
    ylabel: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    labels = [lbl for lbl, _ in values_by_label]
    series = [vals for _, vals in values_by_label]
    if not series:
        return False

    fig = plt.figure(figsize=(max(10, len(labels) * 0.4), 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_quantile_evolution(
    *,
    checkpoints_summary: List[Dict[str, Any]],
    metric_key: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs: List[int] = []
    p05: List[float] = []
    p50: List[float] = []
    p95: List[float] = []
    for ckpt in checkpoints_summary:
        stats = (ckpt.get("stats") or {}).get(metric_key) or {}
        if not stats or not stats.get("n"):
            continue
        xs.append(int(ckpt["after_edits"]))
        p05.append(float(stats["p05"]))
        p50.append(float(stats["p50"]))
        p95.append(float(stats["p95"]))

    if not xs:
        return False

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, p50, label="median (p50)", linewidth=1.5)
    ax.fill_between(xs, p05, p95, alpha=0.2, label="p05–p95")
    ax.set_title(title)
    ax.set_xlabel("after_edits")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_scatter(
    *,
    x: List[float],
    y: List[float],
    c: Optional[List[float]] = None,
    cmap: str = "viridis",
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    fit_line: bool = False,
    paths: Optional[List[Tuple[str, List[float], List[float]]]] = None,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    if not x or not y or len(x) != len(y):
        return False
    if c is not None and len(c) != len(x):
        raise ValueError("scatter color array length mismatch")

    x_vals: List[float] = []
    y_vals: List[float] = []
    c_vals: Optional[List[float]] = [] if c is not None else None
    if c is None:
        for xi, yi in zip(x, y):
            xi = float(xi)
            yi = float(yi)
            if math.isfinite(xi) and math.isfinite(yi):
                x_vals.append(xi)
                y_vals.append(yi)
    else:
        assert c_vals is not None
        for xi, yi, ci in zip(x, y, c):
            xi = float(xi)
            yi = float(yi)
            ci = float(ci)
            if math.isfinite(xi) and math.isfinite(yi) and math.isfinite(ci):
                x_vals.append(xi)
                y_vals.append(yi)
                c_vals.append(ci)
    if not x_vals:
        return False

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    if c is None:
        ax.scatter(x_vals, y_vals, s=10, alpha=0.6)
    else:
        assert c_vals is not None
        sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap=cmap, s=8, alpha=0.15)
        fig.colorbar(sc, ax=ax, label="after_edits")
    if fit_line:
        fit = _linear_fit(x_vals, y_vals)
        if fit is not None:
            x_min = min(x_vals)
            x_max = max(x_vals)
            y_min = fit["slope"] * x_min + fit["intercept"]
            y_max = fit["slope"] * x_max + fit["intercept"]
            ax.plot([x_min, x_max], [y_min, y_max], color="black", linestyle="--", linewidth=1.2)
            ax.text(
                0.03,
                0.97,
                _fit_text(xlabel=xlabel, ylabel=ylabel, fit=fit),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.8},
            )
    if paths:
        path_cmap = plt.get_cmap("tab10")
        for i, (label, px, py) in enumerate(paths):
            if not px or not py or len(px) != len(py):
                continue
            path_x: List[float] = []
            path_y: List[float] = []
            for xi, yi in zip(px, py):
                xi = float(xi)
                yi = float(yi)
                if math.isfinite(xi) and math.isfinite(yi):
                    path_x.append(xi)
                    path_y.append(yi)
            if len(path_x) < 2:
                continue

            color = path_cmap(i % 10)
            ax.plot(
                path_x,
                path_y,
                color=color,
                linewidth=2.0,
                alpha=0.95,
                marker="o",
                markersize=3,
                zorder=5,
            )
            ax.annotate(
                label,
                (path_x[-1], path_y[-1]),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                va="bottom",
                fontsize="small",
                color=color,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7},
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_weight_and_mean_norms(
    *,
    checkpoints_summary: List[Dict[str, Any]],
    out_path: Path,
    title: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs: List[int] = []
    w_norms: List[float] = []
    v0_means: List[float] = []
    vstar_means: List[float] = []
    for ckpt in checkpoints_summary:
        stats = ckpt.get("stats") or {}
        v0_mean = (stats.get("v0_norm") or {}).get("mean")
        vstar_mean = (stats.get("v_star_norm_orig") or {}).get("mean")
        w_norm = ckpt.get("weight_norm")
        if v0_mean is None or vstar_mean is None or w_norm is None:
            continue
        if not (math.isfinite(float(v0_mean)) and math.isfinite(float(vstar_mean)) and math.isfinite(float(w_norm))):
            continue
        xs.append(int(ckpt["after_edits"]))
        w_norms.append(float(w_norm))
        v0_means.append(float(v0_mean))
        vstar_means.append(float(vstar_mean))

    if not xs:
        return False

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    (l_w,) = ax1.plot(xs, w_norms, color="tab:blue", marker="o", linewidth=1.2, label="||W_v||")
    ax1.set_xlabel("after_edits")
    ax1.set_ylabel("||W_v|| (Frobenius)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    (l_v0,) = ax2.plot(
        xs, v0_means, color="tab:orange", linewidth=1.2, label="mean(v0_norm)"
    )
    (l_vstar,) = ax2.plot(
        xs, vstar_means, color="tab:green", linewidth=1.2, label="mean(v_star_norm_orig)"
    )
    ax2.set_ylabel("mean(norm)")

    ax1.set_title(title)
    ax1.legend(handles=[l_w, l_v0, l_vstar], fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_ratio_lines(
    *,
    checkpoints_summary: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    y_label: str,
    series: List[Tuple[str, str]],
):
    """
    Plot ratio line charts.

    series: list of (label, expr) where expr is one of:
      - "w/v0_mean"
      - "w/vstar_mean"
      - "vstar_mean/v0_mean"
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs: List[int] = []
    ys_by_label: Dict[str, List[float]] = {label: [] for label, _ in series}

    for ckpt in checkpoints_summary:
        stats = ckpt.get("stats") or {}
        v0_mean = (stats.get("v0_norm") or {}).get("mean")
        vstar_mean = (stats.get("v_star_norm_orig") or {}).get("mean")
        w_norm = ckpt.get("weight_norm")
        if v0_mean is None or vstar_mean is None or w_norm is None:
            continue
        v0_mean = float(v0_mean)
        vstar_mean = float(vstar_mean)
        w_norm = float(w_norm)
        if not (math.isfinite(v0_mean) and math.isfinite(vstar_mean) and math.isfinite(w_norm)):
            continue
        if v0_mean == 0 or vstar_mean == 0:
            continue

        xs.append(int(ckpt["after_edits"]))
        for label, expr in series:
            if expr == "w/v0_mean":
                ys_by_label[label].append(w_norm / v0_mean)
            elif expr == "w/vstar_mean":
                ys_by_label[label].append(w_norm / vstar_mean)
            elif expr == "vstar_mean/v0_mean":
                ys_by_label[label].append(vstar_mean / v0_mean)
            else:
                raise ValueError(f"Unknown ratio expr {expr!r}")

    if not xs:
        return False

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    for label in ys_by_label:
        ax.plot(xs, ys_by_label[label], marker="o", linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("after_edits")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_std_var_evolution(
    *,
    checkpoints_summary: List[Dict[str, Any]],
    out_path: Path,
    title: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs: List[int] = []
    v0_std: List[float] = []
    vstar_std: List[float] = []
    v0_var: List[float] = []
    vstar_var: List[float] = []

    for ckpt in checkpoints_summary:
        stats = ckpt.get("stats") or {}
        v0_stats = stats.get("v0_norm") or {}
        vstar_stats = stats.get("v_star_norm_orig") or {}
        v0_s = v0_stats.get("std")
        vstar_s = vstar_stats.get("std")
        if v0_s is None or vstar_s is None:
            continue
        v0_s = float(v0_s)
        vstar_s = float(vstar_s)
        if not (math.isfinite(v0_s) and math.isfinite(vstar_s)):
            continue

        xs.append(int(ckpt["after_edits"]))
        v0_std.append(v0_s)
        vstar_std.append(vstar_s)
        v0_var.append(v0_s * v0_s)
        vstar_var.append(vstar_s * vstar_s)

    if not xs:
        return False

    fig = plt.figure(figsize=(8, 6))
    ax_std = fig.add_subplot(2, 1, 1)
    ax_var = fig.add_subplot(2, 1, 2, sharex=ax_std)

    ax_std.plot(xs, v0_std, marker="o", linewidth=1.2, label="std(v0_norm)")
    ax_std.plot(xs, vstar_std, marker="o", linewidth=1.2, label="std(v_star_norm_orig)")
    ax_std.set_ylabel("std")
    ax_std.grid(True, alpha=0.25)
    ax_std.legend(fontsize="small")

    ax_var.plot(xs, v0_var, marker="o", linewidth=1.2, label="var(v0_norm)")
    ax_var.plot(xs, vstar_var, marker="o", linewidth=1.2, label="var(v_star_norm_orig)")
    ax_var.set_xlabel("after_edits")
    ax_var.set_ylabel("variance")
    ax_var.grid(True, alpha=0.25)
    ax_var.legend(fontsize="small")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_std_var_over_weight_evolution(
    *,
    checkpoints_summary: List[Dict[str, Any]],
    out_path: Path,
    title: str,
):
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    xs: List[int] = []
    v0_std_over_w: List[float] = []
    vstar_std_over_w: List[float] = []
    v0_var_over_w: List[float] = []
    vstar_var_over_w: List[float] = []

    for ckpt in checkpoints_summary:
        w_norm = ckpt.get("weight_norm")
        stats = ckpt.get("stats") or {}
        v0_stats = stats.get("v0_norm") or {}
        vstar_stats = stats.get("v_star_norm_orig") or {}
        v0_s = v0_stats.get("std")
        vstar_s = vstar_stats.get("std")
        if w_norm is None or v0_s is None or vstar_s is None:
            continue

        w_norm = float(w_norm)
        v0_s = float(v0_s)
        vstar_s = float(vstar_s)
        if not (math.isfinite(w_norm) and math.isfinite(v0_s) and math.isfinite(vstar_s)):
            continue
        if w_norm == 0.0:
            continue

        xs.append(int(ckpt["after_edits"]))
        v0_std_over_w.append(v0_s / w_norm)
        vstar_std_over_w.append(vstar_s / w_norm)
        v0_var_over_w.append((v0_s * v0_s) / w_norm)
        vstar_var_over_w.append((vstar_s * vstar_s) / w_norm)

    if not xs:
        return False

    fig = plt.figure(figsize=(8, 6))
    ax_std = fig.add_subplot(2, 1, 1)
    ax_var = fig.add_subplot(2, 1, 2, sharex=ax_std)

    ax_std.plot(xs, v0_std_over_w, marker="o", linewidth=1.2, label="std(v0_norm) / ||W_v||")
    ax_std.plot(
        xs,
        vstar_std_over_w,
        marker="o",
        linewidth=1.2,
        label="std(v_star_norm_orig) / ||W_v||",
    )
    ax_std.set_ylabel("std / ||W_v||")
    ax_std.grid(True, alpha=0.25)
    ax_std.legend(fontsize="small")

    ax_var.plot(xs, v0_var_over_w, marker="o", linewidth=1.2, label="var(v0_norm) / ||W_v||")
    ax_var.plot(
        xs,
        vstar_var_over_w,
        marker="o",
        linewidth=1.2,
        label="var(v_star_norm_orig) / ||W_v||",
    )
    ax_var.set_xlabel("after_edits")
    ax_var.set_ylabel("variance / ||W_v||")
    ax_var.grid(True, alpha=0.25)
    ax_var.legend(fontsize="small")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        default="probe_results/memit_norm_probe",
        help="Run dir, parent dir containing run_*/ folders, or a probe_norms.jsonl path.",
    )
    p.add_argument("--out_dir", default=None, help="Defaults to <run_dir>/plots")
    p.add_argument("--focus", default="last", help="Checkpoint after_edits to focus on: int|first|last")
    p.add_argument(
        "--compare",
        default=None,
        help="Comma-separated after_edits list to overlay (e.g. 20,200,500) or 'all'.",
    )
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--density", action="store_true", help="Plot normalized histograms")
    p.add_argument("--hist_style", default="fill", choices=["fill", "step"])
    p.add_argument("--legend", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--hist_alpha", type=float, default=0.45)
    p.add_argument("--line_width", type=float, default=1.0)
    p.add_argument("--path_k", type=int, default=6, help="How many case trajectories to draw on W-v* scatter.")
    p.add_argument("--path_cases", default=None, help="Comma-separated case_ids to draw trajectories for.")
    p.add_argument("--path_strategy", default="max_var", choices=["max_var", "max_delta", "random"])
    p.add_argument("--path_seed", type=int, default=0)
    args = p.parse_args()

    run_dir, jsonl_path, cfg_path = _resolve_run_dir(Path(args.path))
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(cfg_path) if cfg_path else None
    checkpoints = list(_iter_jsonl(jsonl_path))
    checkpoints.sort(key=lambda c: int(c["after_edits"]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {jsonl_path}")

    focus_after = _pick_focus_after_edits(checkpoints, args.focus)
    if args.compare is not None and args.compare.strip().lower() == "all":
        compare_after = sorted({int(c["after_edits"]) for c in checkpoints})
    else:
        compare_after = _parse_int_list(args.compare)
    if compare_after is None:
        compare_after = sorted({int(checkpoints[0]["after_edits"]), int(checkpoints[-1]["after_edits"]), focus_after})

    ckpt_by_after = {int(c["after_edits"]): c for c in checkpoints}
    missing = [a for a in compare_after if a not in ckpt_by_after]
    if missing:
        raise ValueError(f"--compare has after_edits not present in JSONL: {missing}")
    if focus_after not in ckpt_by_after:
        raise ValueError(f"--focus {focus_after} not present in JSONL")

    # Summary stats for each checkpoint.
    summary = {
        "run_dir": str(run_dir),
        "jsonl_path": str(jsonl_path),
        "config_path": str(cfg_path) if cfg_path else None,
        "config": config,
        "checkpoints": [],
    }

    for ckpt in checkpoints:
        after = int(ckpt["after_edits"])
        arrays = _extract_checkpoint_arrays(ckpt)
        summary["checkpoints"].append(
            {
                "after_edits": after,
                "v_layer": ckpt.get("v_layer"),
                "weight_name": ckpt.get("weight_name"),
                "weight_norm": float(ckpt.get("weight_norm")),
                "stats": {k: _summarize(v) for k, v in arrays.items()},
            }
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {summary_path}")

    # Plots.
    title_prefix = ""
    if config and isinstance(config, dict):
        model_name = config.get("model_name")
        if model_name:
            title_prefix = f"{model_name} | "

    _plot_weight_norm(
        checkpoints,
        out_dir / "weight_norm_vs_edits.png",
        title=f"{title_prefix}v-layer weight norm vs edits",
    )

    # Overlay distributions across compare checkpoints (v0, v*).
    v0_series = []
    vstar_series = []
    ratio_series = []
    diff_series = []
    for after in compare_after:
        arrays = _extract_checkpoint_arrays(ckpt_by_after[after])
        v0_series.append((f"after={after}", arrays["v0_norm"]))
        vstar_series.append((f"after={after}", arrays["v_star_norm_orig"]))
        ratio_series.append((f"after={after}", arrays["ratio"]))
        diff_series.append((f"after={after}", arrays["diff"]))

    _plot_hist_overlay(
        series=v0_series,
        out_path=out_dir / "hist_v0_norm_compare.png",
        bins=args.bins,
        title=f"{title_prefix}v0_norm distribution (compare)",
        xlabel="v0_norm",
        density=bool(args.density),
        style=str(args.hist_style),
        legend=str(args.legend),
        cmap=str(args.cmap),
        alpha=float(args.hist_alpha),
        linewidth=float(args.line_width),
    )
    _plot_hist_overlay(
        series=vstar_series,
        out_path=out_dir / "hist_v_star_norm_orig_compare.png",
        bins=args.bins,
        title=f"{title_prefix}v_star_norm_orig distribution (compare)",
        xlabel="v_star_norm_orig",
        density=bool(args.density),
        style=str(args.hist_style),
        legend=str(args.legend),
        cmap=str(args.cmap),
        alpha=float(args.hist_alpha),
        linewidth=float(args.line_width),
    )
    _plot_hist_overlay(
        series=ratio_series,
        out_path=out_dir / "hist_ratio_vstar_over_v0_compare.png",
        bins=args.bins,
        title=f"{title_prefix}ratio v_star_norm_orig / v0_norm (compare)",
        xlabel="v_star_norm_orig / v0_norm",
        density=bool(args.density),
        style=str(args.hist_style),
        legend=str(args.legend),
        cmap=str(args.cmap),
        alpha=float(args.hist_alpha),
        linewidth=float(args.line_width),
    )
    _plot_hist_overlay(
        series=diff_series,
        out_path=out_dir / "hist_diff_vstar_minus_v0_compare.png",
        bins=args.bins,
        title=f"{title_prefix}diff v_star_norm_orig - v0_norm (compare)",
        xlabel="v_star_norm_orig - v0_norm",
        density=bool(args.density),
        style=str(args.hist_style),
        legend=str(args.legend),
        cmap=str(args.cmap),
        alpha=float(args.hist_alpha),
        linewidth=float(args.line_width),
    )

    # Focused checkpoint ECDFs.
    focus_ckpt = ckpt_by_after[focus_after]
    focus_arrays = _extract_checkpoint_arrays(focus_ckpt)
    _plot_ecdf_overlay(
        series=[(f"after={focus_after} v0", focus_arrays["v0_norm"]), (f"after={focus_after} v*", focus_arrays["v_star_norm_orig"])],
        out_path=out_dir / f"ecdf_v0_vs_vstar_after_{focus_after}.png",
        title=f"{title_prefix}ECDF v0_norm vs v_star_norm_orig (after={focus_after})",
        xlabel="norm",
    )
    # Scatter with all checkpoints (default view).
    all_v0: List[float] = []
    all_vstar: List[float] = []
    all_w: List[float] = []
    all_after: List[float] = []
    for ckpt in checkpoints:
        after = float(int(ckpt["after_edits"]))
        arrays = _extract_checkpoint_arrays(ckpt)
        w_norm = ckpt.get("weight_norm")
        if w_norm is None:
            continue
        w_norm = float(w_norm)
        if not math.isfinite(w_norm):
            continue
        n_added = 0
        for v0, vstar in zip(arrays["v0_norm"], arrays["v_star_norm_orig"]):
            v0 = float(v0)
            vstar = float(vstar)
            if not (math.isfinite(v0) and math.isfinite(vstar)):
                continue
            all_v0.append(v0)
            all_vstar.append(vstar)
            all_w.append(w_norm)
            n_added += 1
        all_after.extend([after] * n_added)

    _plot_scatter(
        x=all_v0,
        y=all_vstar,
        c=all_after,
        cmap=str(args.cmap),
        out_path=out_dir / "scatter_v0_vs_vstar.png",
        title=f"{title_prefix}v0_norm vs v_star_norm_orig (all checkpoints)",
        xlabel="v0_norm",
        ylabel="v_star_norm_orig",
        fit_line=True,
    )
    _plot_scatter(
        x=all_v0,
        y=all_vstar,
        c=all_after,
        cmap=str(args.cmap),
        out_path=out_dir / "scatter_v0_vs_vstar_all.png",
        title=f"{title_prefix}v0_norm vs v_star_norm_orig (all checkpoints)",
        xlabel="v0_norm",
        ylabel="v_star_norm_orig",
        fit_line=True,
    )
    _plot_scatter(
        x=all_w,
        y=all_v0,
        c=all_after,
        cmap=str(args.cmap),
        out_path=out_dir / "scatter_w_norm_vs_v0_norm.png",
        title=f"{title_prefix}W_norm vs v0_norm (all checkpoints)",
        xlabel="W_norm",
        ylabel="v0_norm",
        fit_line=True,
    )

    path_case_ids = _parse_case_id_list(args.path_cases)
    vstar_paths: List[Tuple[str, List[float], List[float]]] = []
    if (path_case_ids and len(path_case_ids) > 0) or int(args.path_k) > 0:
        vstar_case_series = _build_case_series(checkpoints, y_key="v_star_norm_orig")
        if path_case_ids is None:
            path_case_ids = _select_case_ids_for_paths(
                vstar_case_series,
                k=int(args.path_k),
                strategy=str(args.path_strategy),
                seed=int(args.path_seed),
            )
        for case_id in path_case_ids:
            pts = vstar_case_series.get(case_id)
            if not pts or len(pts) < 2:
                continue
            vstar_paths.append(
                (
                    str(case_id),
                    [w for _, w, y in pts],
                    [y for _, w, y in pts],
                )
            )
    if vstar_paths:
        print(f"Trajectory cases for W-v*: {[label for label, _, _ in vstar_paths]}")
    _plot_scatter(
        x=all_w,
        y=all_vstar,
        c=all_after,
        cmap=str(args.cmap),
        out_path=out_dir / "scatter_w_norm_vs_v_star_norm.png",
        title=f"{title_prefix}W_norm vs v_star_norm_orig (all checkpoints)",
        xlabel="W_norm",
        ylabel="v_star_norm_orig",
        fit_line=True,
        paths=vstar_paths,
    )

    # Scatter for a single checkpoint (kept for close-up inspection).
    _plot_scatter(
        x=focus_arrays["v0_norm"],
        y=focus_arrays["v_star_norm_orig"],
        out_path=out_dir / f"scatter_v0_vs_vstar_after_{focus_after}.png",
        title=f"{title_prefix}v0_norm vs v_star_norm_orig (after={focus_after})",
        xlabel="v0_norm",
        ylabel="v_star_norm_orig",
        fit_line=True,
    )

    # All-checkpoint boxplots (distribution drift).
    v0_all = [(f"{int(c['after_edits'])}", _extract_checkpoint_arrays(c)["v0_norm"]) for c in checkpoints]
    vstar_all = [(f"{int(c['after_edits'])}", _extract_checkpoint_arrays(c)["v_star_norm_orig"]) for c in checkpoints]
    _plot_boxplot(
        values_by_label=v0_all,
        out_path=out_dir / "boxplot_v0_norm_all.png",
        title=f"{title_prefix}v0_norm distribution across checkpoints",
        ylabel="v0_norm",
    )
    _plot_boxplot(
        values_by_label=vstar_all,
        out_path=out_dir / "boxplot_v_star_norm_orig_all.png",
        title=f"{title_prefix}v_star_norm_orig distribution across checkpoints",
        ylabel="v_star_norm_orig",
    )

    # Quantile evolution plots.
    _plot_quantile_evolution(
        checkpoints_summary=summary["checkpoints"],
        metric_key="v0_norm",
        out_path=out_dir / "quantiles_v0_norm.png",
        title=f"{title_prefix}v0_norm quantiles vs edits",
        ylabel="v0_norm",
    )
    _plot_quantile_evolution(
        checkpoints_summary=summary["checkpoints"],
        metric_key="v_star_norm_orig",
        out_path=out_dir / "quantiles_v_star_norm_orig.png",
        title=f"{title_prefix}v_star_norm_orig quantiles vs edits",
        ylabel="v_star_norm_orig",
    )

    _plot_weight_and_mean_norms(
        checkpoints_summary=summary["checkpoints"],
        out_path=out_dir / "line_weight_norm_and_mean_v_norms.png",
        title=f"{title_prefix}||W_v|| and mean v0/v* norms vs edits",
    )
    _plot_ratio_lines(
        checkpoints_summary=summary["checkpoints"],
        out_path=out_dir / "line_ratio_weight_over_mean_v_norms.png",
        title=f"{title_prefix}||W_v|| / mean(v0_norm) and ||W_v|| / mean(v*) vs edits",
        y_label="ratio",
        series=[
            ("||W_v|| / mean(v0_norm)", "w/v0_mean"),
            ("||W_v|| / mean(v_star_norm_orig)", "w/vstar_mean"),
        ],
    )
    _plot_ratio_lines(
        checkpoints_summary=summary["checkpoints"],
        out_path=out_dir / "line_ratio_mean_vstar_over_v0.png",
        title=f"{title_prefix}mean(v_star_norm_orig) / mean(v0_norm) vs edits",
        y_label="ratio",
        series=[("mean(v_star_norm_orig) / mean(v0_norm)", "vstar_mean/v0_mean")],
    )
    _plot_std_var_evolution(
        checkpoints_summary=summary["checkpoints"],
        out_path=out_dir / "line_std_var_v0_vstar.png",
        title=f"{title_prefix}std/var of v0 and v* norms vs edits",
    )
    _plot_std_var_over_weight_evolution(
        checkpoints_summary=summary["checkpoints"],
        out_path=out_dir / "line_std_var_over_weight_v0_vstar.png",
        title=f"{title_prefix}std/var (v0,v*) divided by ||W_v|| vs edits",
    )

    print(f"Plots saved under {out_dir}")


if __name__ == "__main__":
    main()
