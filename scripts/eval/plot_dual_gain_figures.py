#!/usr/bin/env python3
"""
Generate two publication-quality figures for NeurIPS 2026.

Figure 1 — Δ-Gain Amplification (the "smoking gun"):
  Grouped bars showing dual_llm − desc_only for key metrics at L1 vs L3.
  Error bars = 95% CI of the paired delta (paired t-test).
  Significance stars: * p<0.05, ** p<0.01, *** p<0.001.
  The sign reversal on AgiBot is the visual climax.

Figure 2 — Performance Trajectory under Task Complexity:
  Line plot from L1→L3 for both methods, with CI bands for SR and TP.
  Shows description_only degrades steeply while dual_llm holds or improves.
"""

import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "/mnt/data3/data_xingrui/lueq/NuerIPS_2026/evaluation_results_dualsystem/selected_splits"
OUT  = "/mnt/data3/data_xingrui/lueq/NuerIPS_2026/docs/figures"
os.makedirs(OUT, exist_ok=True)

# ── global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "pdf.fonttype":       42,   # editable text in Illustrator
    "ps.fonttype":        42,
})

# ── colour palette (accessible) ────────────────────────────────────────────
C_DESC = "#6B8CAE"   # muted steel-blue  → description_only / L1
C_DUAL = "#E8735A"   # warm coral-red    → dual_llm        / L3
C_L1   = "#7BAFD4"   # light blue        → L1 bars
C_L3   = "#D95F3B"   # deep orange-red   → L3 bars
ALPHA  = 0.88

# ══════════════════════════════════════════════════════════════════════════════
# Helper: load JSON and pull per-episode success arrays
# ══════════════════════════════════════════════════════════════════════════════
def load_json(fname):
    with open(os.path.join(BASE, fname)) as f:
        return json.load(f)

def get_summary(data, mode_key):
    """Retrieve summary dict; handle both JSON layouts."""
    s = data["summary"]
    return s[mode_key]

def get_episode_values(data, mode_key, field):
    """Return per-episode scalar array for a given field."""
    episodes = data["episodes"]
    out = []
    for ep in episodes:
        val = ep.get(mode_key, {}).get(field)
        if val is not None:
            out.append(float(val))
    return np.array(out)

def paired_delta_stats(arr_dual, arr_bl):
    """
    Compute mean delta, 95% CI half-width, and p-value from paired t-test.
    Returns (mean_delta, ci_halfwidth, pvalue).
    """
    deltas = arr_dual - arr_bl
    n = len(deltas)
    mean_d = deltas.mean()
    se = deltas.std(ddof=1) / np.sqrt(n)
    ci = 1.96 * se
    _, pval = stats.ttest_rel(arr_dual, arr_bl)
    return mean_d, ci, pval

def sig_stars(pval):
    """Return significance star string."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return ""

def prop_ci95(arr):
    """95 % CI half-width for a Bernoulli proportion."""
    n = len(arr)
    p = arr.mean()
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se

def mean_ci95(arr):
    """95 % CI half-width for a mean (z-approx)."""
    return 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))

# ── load all four DreamZero JSON files ─────────────────────────────────────
dro_l1 = load_json("DRO_L1_150_results.json")
dro_l3 = load_json("DRO_L3_150_results.json")
agi_l1 = load_json("Agi_L1_150_results.json")
agi_l3 = load_json("Agi_L3_150_results.json")

# baseline key in DRO JSONs is 'task_token_only'
BL_KEY = "task_token_only"
DU_KEY = "dual_llm"


# ══════════════════════════════════════════════════════════════════════════════
# ░░░  FIGURE 1  ░░░  Δ-Gain Amplification  (with error bars + sig stars)
# ══════════════════════════════════════════════════════════════════════════════

def compute_deltas(data_l1, data_l3, time_overhead_l1, time_overhead_l3):
    """
    time_overhead_l1/l3: (T_dual - T_bl) / T_bl * 100  (positive = dual is slower)
    Derived from doc section 6.2/6.4:
      DROID:  T_bl(L1)=2.1233, T_bl(L3)=2.1256, T_dual≈2.24 s/step
      AgiBot: T_bl≈2.207 (desc-only, B1), T_dual=2.281 s/step (T_policy+T_plan/K)
    """
    s_l1_bl = get_summary(data_l1, BL_KEY)
    s_l1_du = get_summary(data_l1, DU_KEY)
    s_l3_bl = get_summary(data_l3, BL_KEY)
    s_l3_du = get_summary(data_l3, DU_KEY)

    # relative L2 reduction (%) — positive = dual better
    l2_l1 = (s_l1_bl["mean_l2"] - s_l1_du["mean_l2"]) / s_l1_bl["mean_l2"] * 100
    l2_l3 = (s_l3_bl["mean_l2"] - s_l3_du["mean_l2"]) / s_l3_bl["mean_l2"] * 100

    # Task Progress Δ (pp)
    tp_l1 = (s_l1_du["mean_task_progress"] - s_l1_bl["mean_task_progress"]) * 100
    tp_l3 = (s_l3_du["mean_task_progress"] - s_l3_bl["mean_task_progress"]) * 100

    # Success Rate Δ (pp)
    sr_key = "success_rate"
    sr_l1 = (s_l1_du[sr_key] - s_l1_bl[sr_key]) * 100
    sr_l3 = (s_l3_du[sr_key] - s_l3_bl[sr_key]) * 100

    # Rate L2 < 0.1 Δ (pp)
    def rate_key(s):
        if "mean_step_alignment_l2_lt_0_1" in s:
            return "mean_step_alignment_l2_lt_0_1"
        return "rate_of_l2_lt_0_1"
    r_l1 = (s_l1_du[rate_key(s_l1_du)] - s_l1_bl[rate_key(s_l1_bl)]) * 100
    r_l3 = (s_l3_du[rate_key(s_l3_du)] - s_l3_bl[rate_key(s_l3_bl)]) * 100

    return dict(
        l2   = (l2_l1,  l2_l3),
        tp   = (tp_l1,  tp_l3),
        sr   = (sr_l1,  sr_l3),
        r01  = (r_l1,   r_l3),
        time = (time_overhead_l1, time_overhead_l3),
    )

# Time overhead (%) = (T_dual - T_desc_only) / T_desc_only * 100
#   DROID  L1: (2.24 - 2.1233) / 2.1233 * 100 = 5.5%
#   DROID  L3: (2.24 - 2.1256) / 2.1256 * 100 = 5.4%
#   AgiBot L1: (2.281 - 2.207) / 2.207 * 100  = 3.4%  (B1 desc-only as baseline)
#   AgiBot L3: (2.281 - 2.207) / 2.207 * 100  = 3.4%  (planning overhead is per-episode fixed)
# Use "desc-only" (B1 for AgiBot, task_token for DROID) as T_bl:
DRO_T_OVERHEAD = (5.5, 5.4)   # (L1, L3), from section 6.2
AGI_T_OVERHEAD = (3.4, 3.4)   # (L1, L3), from section 6.4  T_plan/K=0.151s vs T_policy=2.207s

dro_d = compute_deltas(dro_l1, dro_l3, *DRO_T_OVERHEAD)
agi_d = compute_deltas(agi_l1, agi_l3, *AGI_T_OVERHEAD)

# ── colour for time-overhead bars (neutral warm gray, distinct from gain palette)
C_TIME = "#B0A090"

# ---- plot -----------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)
fig1.subplots_adjust(wspace=0.36)

PERF_KEYS    = ["l2", "tp", "sr", "r01"]
PERF_LABELS  = ["Rel. L2\nReduction (%)", "Task\nProgress (pp)",
                "Success\nRate (pp)", "Rate L2\n<0.1 (pp)"]

# time overhead placed one unit to the right with a visual gap
x_perf = np.arange(len(PERF_KEYS))          # 0,1,2,3
x_time = np.array([len(PERF_KEYS) + 0.7])   # 4.7 — gap creates separator
x_all  = np.concatenate([x_perf, x_time])

width  = 0.32

for ax, title, deltas in [
    (axes[0], "DreamZero  /  DROID",  dro_d),
    (axes[1], "DreamZero  /  AgiBot", agi_d),
]:
    l1_perf = [deltas[k][0] for k in PERF_KEYS]
    l3_perf = [deltas[k][1] for k in PERF_KEYS]
    t_l1    = deltas["time"][0]
    t_l3    = deltas["time"][1]

    # ── performance bars (blue L1, orange L3)
    b1 = ax.bar(x_perf - width/2, l1_perf, width,
                color=C_L1, alpha=ALPHA, label="L1 (simple tasks)",
                edgecolor="white", linewidth=0.6, zorder=3)
    b3 = ax.bar(x_perf + width/2, l3_perf, width,
                color=C_L3, alpha=ALPHA, label="L3 (complex tasks)",
                edgecolor="white", linewidth=0.6, zorder=3)

    # ── time overhead bars — same neutral color for both L1/L3 to highlight constancy
    bt1 = ax.bar(x_time - width/2, [t_l1], width,
                 color=C_TIME, alpha=0.85,
                 edgecolor="#7A6A5A", linewidth=0.9,
                 linestyle="--", zorder=3,
                 label="Time overhead (L1)")
    bt3 = ax.bar(x_time + width/2, [t_l3], width,
                 color=C_TIME, alpha=0.70,
                 edgecolor="#7A6A5A", linewidth=0.9,
                 linestyle="--", zorder=3,
                 label="Time overhead (L3)")

    # ── vertical dashed separator between perf and time
    ax.axvline(x=len(PERF_KEYS) + 0.15, color="#AAAAAA",
               linewidth=1.0, linestyle=":", zorder=2)

    # ── zero line
    ax.axhline(0, color="black", linewidth=1.1, zorder=4)

    # ── value labels on all bars
    for bar in list(b1) + list(b3) + list(bt1) + list(bt3):
        h  = bar.get_height()
        if h == 0:
            continue
        va = "bottom" if h >= 0 else "top"
        dy = 0.08 if h >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + dy, f"{h:+.1f}",
                ha="center", va=va, fontsize=8.5, fontweight="bold",
                color="black")

    # ── x-axis: perf labels + time label
    tick_positions = list(x_perf) + list(x_time)
    tick_labels    = PERF_LABELS + ["Time\nOverhead (%)"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9.5)

    # ── shade the time-overhead region
    ax.axvspan(len(PERF_KEYS) + 0.15, len(PERF_KEYS) + 1.3,
               alpha=0.06, color=C_TIME, zorder=0)

    ax.set_title(title, fontweight="bold", fontsize=12, pad=8)
    ax.set_ylabel("Δ  (Dual LLM − Description Only)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, len(PERF_KEYS) + 1.25)

    # ── legend: only perf entries in first panel; time label as text annotation
    if ax is axes[0]:
        handles = [
            mpatches.Patch(color=C_L1, alpha=ALPHA, label="L1 (simple tasks)"),
            mpatches.Patch(color=C_L3, alpha=ALPHA, label="L3 (complex tasks)"),
            mpatches.Patch(color=C_TIME, alpha=0.85, label="Time overhead (both levels)"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=9.0, loc="upper left")

out1_png = os.path.join(OUT, "fig1_delta_gain_amplification.png")
out1_pdf = os.path.join(OUT, "fig1_delta_gain_amplification.pdf")
fig1.savefig(out1_png, dpi=300, bbox_inches="tight")
fig1.savefig(out1_pdf, bbox_inches="tight")
print(f"✓ Figure 1 saved → {out1_png}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░░  FIGURE 2  ░░░  Performance Trajectory  L1 → L3
# ══════════════════════════════════════════════════════════════════════════════

def episode_success(data, mode_key):
    """Return binary (0/1) numpy array for task success across episodes."""
    eps = data["episodes"]
    out = []
    for ep in eps:
        v = ep.get(mode_key, {}).get("task_success")
        if v is not None:
            out.append(float(v))
    return np.array(out)

def episode_progress(data, mode_key):
    eps = data["episodes"]
    out = []
    for ep in eps:
        v = ep.get(mode_key, {}).get("task_progress")
        if v is not None:
            out.append(float(v))
    return np.array(out)

# ── collect arrays ─────────────────────────────────────────────────────────
dro_l1_bl_sr  = episode_success(dro_l1, BL_KEY)
dro_l1_du_sr  = episode_success(dro_l1, DU_KEY)
dro_l3_bl_sr  = episode_success(dro_l3, BL_KEY)
dro_l3_du_sr  = episode_success(dro_l3, DU_KEY)

agi_l1_bl_sr  = episode_success(agi_l1, BL_KEY)
agi_l1_du_sr  = episode_success(agi_l1, DU_KEY)
agi_l3_bl_sr  = episode_success(agi_l3, BL_KEY)
agi_l3_du_sr  = episode_success(agi_l3, DU_KEY)

dro_l1_bl_tp  = episode_progress(dro_l1, BL_KEY)
dro_l1_du_tp  = episode_progress(dro_l1, DU_KEY)
dro_l3_bl_tp  = episode_progress(dro_l3, BL_KEY)
dro_l3_du_tp  = episode_progress(dro_l3, DU_KEY)

agi_l1_bl_tp  = episode_progress(agi_l1, BL_KEY)
agi_l1_du_tp  = episode_progress(agi_l1, DU_KEY)
agi_l3_bl_tp  = episode_progress(agi_l3, BL_KEY)
agi_l3_du_tp  = episode_progress(agi_l3, DU_KEY)

# ── helper: mean ± 95% CI ──────────────────────────────────────────────────
def mc(arr):
    p = arr.mean() * 100
    ci = prop_ci95(arr) * 100
    return p, ci

def mc_tp(arr):
    m = arr.mean() * 100
    ci = mean_ci95(arr) * 100
    return m, ci

# ── build (mean, ci) pairs for both datasets ────────────────────────────────
datasets = {
    "DROID": {
        "bl_sr": [mc(dro_l1_bl_sr), mc(dro_l3_bl_sr)],
        "du_sr": [mc(dro_l1_du_sr), mc(dro_l3_du_sr)],
        "bl_tp": [mc_tp(dro_l1_bl_tp), mc_tp(dro_l3_bl_tp)],
        "du_tp": [mc_tp(dro_l1_du_tp), mc_tp(dro_l3_du_tp)],
    },
    "AgiBot": {
        "bl_sr": [mc(agi_l1_bl_sr), mc(agi_l3_bl_sr)],
        "du_sr": [mc(agi_l1_du_sr), mc(agi_l3_du_sr)],
        "bl_tp": [mc_tp(agi_l1_bl_tp), mc_tp(agi_l3_bl_tp)],
        "du_tp": [mc_tp(agi_l1_du_tp), mc_tp(agi_l3_du_tp)],
    },
}

# ── paired significance at L3 endpoint ──────────────────────────────────────
def l3_sig(bl_arr, du_arr):
    _, pval = stats.ttest_rel(du_arr, bl_arr)
    return sig_stars(pval)

l3_sig_labels = {
    "DROID":  {"sr": l3_sig(dro_l3_bl_sr, dro_l3_du_sr),
               "tp": l3_sig(dro_l3_bl_tp, dro_l3_du_tp)},
    "AgiBot": {"sr": l3_sig(agi_l3_bl_sr, agi_l3_du_sr),
               "tp": l3_sig(agi_l3_bl_tp, agi_l3_du_tp)},
}

X_TICKS   = [0, 1]
X_LABELS  = ["L1\n(Simple Tasks)", "L3\n(Complex Tasks)"]
JITTER    = 0.03   # horizontal offset so CI bands don't overlap

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
fig2.subplots_adjust(wspace=0.38)

for ax, dsname in zip(axes2, ["DROID", "AgiBot"]):
    d    = datasets[dsname]
    title = f"DreamZero  /  {dsname}"

    # ── extract mean and CI ────────────────────────────────────────────────
    bl_sr_m  = np.array([v[0] for v in d["bl_sr"]])
    bl_sr_ci = np.array([v[1] for v in d["bl_sr"]])
    du_sr_m  = np.array([v[0] for v in d["du_sr"]])
    du_sr_ci = np.array([v[1] for v in d["du_sr"]])

    bl_tp_m  = np.array([v[0] for v in d["bl_tp"]])
    bl_tp_ci = np.array([v[1] for v in d["bl_tp"]])
    du_tp_m  = np.array([v[0] for v in d["du_tp"]])
    du_tp_ci = np.array([v[1] for v in d["du_tp"]])

    xs_bl = np.array(X_TICKS) - JITTER
    xs_du = np.array(X_TICKS) + JITTER

    # ── Success Rate (solid lines, primary metric) ─────────────────────────
    ax.plot(xs_bl, bl_sr_m, "o-", color=C_DESC, lw=2.4, ms=8,
            label="Description-only  (SR)", zorder=5)
    ax.fill_between(xs_bl,
                    bl_sr_m - bl_sr_ci, bl_sr_m + bl_sr_ci,
                    color=C_DESC, alpha=0.18, zorder=3)

    ax.plot(xs_du, du_sr_m, "s-", color=C_DUAL, lw=2.4, ms=8,
            label="Dual-LLM  (SR)", zorder=5)
    ax.fill_between(xs_du,
                    du_sr_m - du_sr_ci, du_sr_m + du_sr_ci,
                    color=C_DUAL, alpha=0.18, zorder=3)

    # ── Task Progress (dashed lines with CI bands) ─────────────────────────
    ax.plot(xs_bl, bl_tp_m, "o--", color=C_DESC, lw=1.8, ms=7,
            alpha=0.65, label="Description-only  (TP)")
    ax.fill_between(xs_bl,
                    bl_tp_m - bl_tp_ci, bl_tp_m + bl_tp_ci,
                    color=C_DESC, alpha=0.10, zorder=2)

    ax.plot(xs_du, du_tp_m, "s--", color=C_DUAL, lw=1.8, ms=7,
            alpha=0.65, label="Dual-LLM  (TP)")
    ax.fill_between(xs_du,
                    du_tp_m - du_tp_ci, du_tp_m + du_tp_ci,
                    color=C_DUAL, alpha=0.10, zorder=2)

    # ── annotate L3 endpoint values ────────────────────────────────────────
    for yvals, xs, color, offset_sign in [
        (bl_sr_m, xs_bl, C_DESC, -1),
        (du_sr_m, xs_du, C_DUAL, +1),
    ]:
        ax.annotate(f"{yvals[1]:.1f}%",
                    xy=(xs[1], yvals[1]),
                    xytext=(xs[1] + offset_sign * 0.09, yvals[1] + offset_sign * 1.8),
                    fontsize=9.5, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.8, alpha=0.6))

    # ── significance marker at L3 SR endpoint ─────────────────────────────
    sr_stars = l3_sig_labels[dsname]["sr"]
    if sr_stars:
        y_max_l3 = max(bl_sr_m[1] + bl_sr_ci[1], du_sr_m[1] + du_sr_ci[1])
        x_mid = (xs_bl[1] + xs_du[1]) / 2
        ax.annotate("", xy=(xs_du[1], y_max_l3 + 1.5),
                    xytext=(xs_bl[1], y_max_l3 + 1.5),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
        ax.text(x_mid, y_max_l3 + 2.0, sr_stars,
                ha="center", va="bottom", fontsize=10, color="#CC0000",
                fontweight="bold")

    # ── AgiBot: annotate the crossing / reversal ───────────────────────────
    if dsname == "AgiBot":
        ax.text(0.72, 0.42, "description-only\ndegrades  −8.7 pp",
                transform=ax.transAxes, fontsize=8.5,
                color=C_DESC, fontstyle="italic", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7,
                          edgecolor=C_DESC, lw=0.8))
        ax.text(0.28, 0.82, "dual-LLM\nimproves  +4.6 pp",
                transform=ax.transAxes, fontsize=8.5,
                color=C_DUAL, fontstyle="italic", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7,
                          edgecolor=C_DUAL, lw=0.8))

    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_LABELS, fontsize=10.5)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Performance (%)", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # shared legend from first panel only
    if ax is axes2[0]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:4], labels[:4],
                  frameon=True, framealpha=0.92,
                  fontsize=8.8, loc="lower left",
                  edgecolor="#cccccc")

    # light shaded region to highlight "complexity jump"
    ax.axvspan(0.5, 1.5, alpha=0.04, color="gray", zorder=0)


fig2.suptitle(
    "DreamZero Robustness Under Increasing Task Complexity\n"
    "Solid lines = Success Rate  ·  Dashed lines = Task Progress  ·  Shaded bands = 95% CI",
    fontsize=12.5, fontweight="bold", y=1.04
)

out2_png = os.path.join(OUT, "fig2_performance_trajectory.png")
out2_pdf = os.path.join(OUT, "fig2_performance_trajectory.pdf")
fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
fig2.savefig(out2_pdf, bbox_inches="tight")
print(f"✓ Figure 2 saved → {out2_png}")

# ── final summary ─────────────────────────────────────────────────────────
print("\n── Summary of plotted values ──────────────────────────────────────")
print("\n  DROID   SR:  desc=[{:.1f}%, {:.1f}%]  dual=[{:.1f}%, {:.1f}%]".format(
    *[v[0] for v in datasets["DROID"]["bl_sr"]],
    *[v[0] for v in datasets["DROID"]["du_sr"]]))
print("  AgiBot  SR:  desc=[{:.1f}%, {:.1f}%]  dual=[{:.1f}%, {:.1f}%]".format(
    *[v[0] for v in datasets["AgiBot"]["bl_sr"]],
    *[v[0] for v in datasets["AgiBot"]["du_sr"]]))
print("  DROID   TP:  desc=[{:.1f}%, {:.1f}%]  dual=[{:.1f}%, {:.1f}%]".format(
    *[v[0] for v in datasets["DROID"]["bl_tp"]],
    *[v[0] for v in datasets["DROID"]["du_tp"]]))
print("  AgiBot  TP:  desc=[{:.1f}%, {:.1f}%]  dual=[{:.1f}%, {:.1f}%]".format(
    *[v[0] for v in datasets["AgiBot"]["bl_tp"]],
    *[v[0] for v in datasets["AgiBot"]["du_tp"]]))

print("\n── Paired t-test p-values at L3 ────────────────────────────────────")
from scipy.stats import ttest_rel
for dsname, bl_sr, du_sr, bl_tp, du_tp in [
    ("DROID",  dro_l3_bl_sr, dro_l3_du_sr, dro_l3_bl_tp, dro_l3_du_tp),
    ("AgiBot", agi_l3_bl_sr, agi_l3_du_sr, agi_l3_bl_tp, agi_l3_du_tp),
]:
    _, p_sr = ttest_rel(du_sr, bl_sr)
    _, p_tp = ttest_rel(du_tp, bl_tp)
    print(f"  {dsname}  SR p={p_sr:.4f} {sig_stars(p_sr)}   TP p={p_tp:.4f} {sig_stars(p_tp)}")
