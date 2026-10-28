"""Figures for the preprint. EVERY value below is copied from a run log or from
NB-17's output. Nothing is fitted, smoothed or extrapolated except where the
caption says so explicitly, and no figure shows a relationship we did not test."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

# ---- measured: 13 attacked checkpoints (NB-17 runs.csv) --------------------
# (label, exposure h, clean WER on the 100 attacked utts, attack success %, kind)
RUNS = [
 ("1h s1337",      0.98, 0.6355, 75.0, "orig"),
 ("1h s1337 re-run",0.98, 0.6251, 41.0, "orig"),
 ("1h s1338",      0.98, 0.5905, 57.0, "repl"),
 ("1h s1339",      0.98, 0.6240, 39.0, "repl"),
 ("5h s1337",      4.90, 0.4233, 79.0, "orig"),
 ("5h s1338",      4.90, 0.4152, 96.0, "repl"),
 ("5h s1339",      4.90, 0.4187, 59.0, "repl"),
 ("25h s1337",    24.60, 0.3114, 66.0, "orig"),
 ("25h s1338",    24.60, 0.2987, 78.0, "repl"),
 ("25h s1339",    24.60, 0.2907, 57.0, "repl"),
 ("100h ms-1000", 38.80, 0.6943, 37.0, "orig"),
 ("100h ms-2000", 77.70, 0.4752, 46.0, "orig"),
 ("100h ms-3500", 98.02, 0.3449, 53.0, "orig"),
]
C_ORIG, C_REPL = "#1f3b73", "#c2401a"

# =========================== FIGURE 1 =====================================
fig, ax = plt.subplots(figsize=(6.2, 3.5))
by_h = {}
for lab, h, w, a, k in RUNS: by_h.setdefault(h, []).append((a, k, lab))
for h, v in by_h.items():
    if len(v) > 1:                        # vertical range bar: measured min..max
        ax.plot([h, h], [min(x[0] for x in v), max(x[0] for x in v)],
                color="0.65", lw=5, solid_capstyle="butt", zorder=1)
for lab, h, w, a, k in RUNS:
    ax.scatter(h, a, s=46, zorder=3, marker="o" if k == "orig" else "D",
               facecolor=C_ORIG if k == "orig" else C_REPL,
               edgecolor="white", linewidth=0.8)
for h, v in by_h.items():
    if len(v) > 1:
        ax.annotate(f"{max(x[0] for x in v)-min(x[0] for x in v):.0f} pp\nspread",
                    (h, max(x[0] for x in v)), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color="0.35")
ax.plot([38.80, 98.02], [30, 30], color=C_ORIG, lw=1.0, alpha=0.6)
for _x in (38.80, 98.02):
    ax.plot([_x, _x], [30, 31.4], color=C_ORIG, lw=1.0, alpha=0.6)
ax.annotate("three checkpoints of ONE 100 h run", (61, 29.2), ha="center",
            va="top", fontsize=7, color=C_ORIG)
ax.set_xscale("log")
ax.set_xticks([1, 5, 25, 100]); ax.set_xticklabels(["1", "5", "25", "100"])
ax.set_xlabel("training audio the checkpoint has seen (hours, log scale)")
ax.set_ylabel("targeted attack success (%)")
ax.set_ylim(25, 118); ax.grid(axis="y", color="0.92", lw=0.7)
ax.scatter([], [], marker="o", facecolor=C_ORIG, edgecolor="white",
           s=46, label="original ladder (one run each)")
ax.scatter([], [], marker="D", facecolor=C_REPL, edgecolor="white",
           s=46, label="replicate (seed 1338 / 1339)")
ax.plot([], [], color="0.65", lw=5, label="range across runs of one condition")
ax.legend(frameon=False, fontsize=7.5, loc="upper right",
          handlelength=1.4, borderaxespad=0.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_runs.pdf"); plt.close(fig)

# =========================== FIGURE 2 =====================================
# the four within-environment replicate pairs (NB-17 variance_by_group)
PAIRS = [("1 h, transformers 4.44.2", 41.0, 75.0, 24.0, "1.164e-06"),
         ("1 h, transformers 5.16.1", 39.0, 57.0, 12.7, "1.135e-02"),
         ("5 h, transformers 5.16.1", 59.0, 96.0, 26.2, "7.840e-10"),
         ("25 h, transformers 5.16.1", 57.0, 78.0, 14.8, "3.241e-04")]
fig, ax = plt.subplots(figsize=(6.2, 2.9))
for i, (lab, lo, hi, sd, pv) in enumerate(PAIRS):
    y = len(PAIRS) - 1 - i
    ax.plot([lo, hi], [y, y], color="0.6", lw=1.4, zorder=1)
    ax.scatter([lo, hi], [y, y], s=44, color=C_REPL, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.annotate(f"{hi-lo:.1f} pp   $p$={pv}", ((lo+hi)/2, y),
                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)
ax.set_yticks(range(len(PAIRS)))
ax.set_yticklabels([p[0] for p in PAIRS][::-1], fontsize=8)
ax.set_xlabel("targeted attack success (%)"); ax.set_xlim(30, 105)
ax.set_ylim(-0.6, len(PAIRS) - 0.25)
ax.grid(axis="x", color="0.92", lw=0.7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_pairs_of_runs.pdf"); plt.close(fig)

# =========================== FIGURE 3 =====================================
# iso-WER pair effects against the between-run SE of a two-run contrast
SE = 21.57 * np.sqrt(2)
EFF = [("Pair 1  1 h vs ms-1000  (40x)", 38.0),
       ("Pair 2  5 h vs ms-2000  (16x)", 33.0),
       ("Pair 3  25 h vs ms-3500  (4x)", 13.0)]
fig, ax = plt.subplots(figsize=(6.2, 2.5))
for i, (lab, e) in enumerate(EFF):
    y = len(EFF) - 1 - i
    ax.plot([e - 1.96*SE, e + 1.96*SE], [y, y], color=C_ORIG, lw=1.6)
    ax.scatter(e, y, s=52, color=C_ORIG, zorder=3, edgecolor="white", linewidth=0.8)
    ax.annotate(f"{e:.0f} pp = {e/SE:.2f} SE", (e, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7.5)
ax.axvline(0, color="0.2", lw=1, ls="--")
ax.set_yticks(range(len(EFF))); ax.set_yticklabels([e[0] for e in EFF][::-1], fontsize=8)
ax.set_xlabel("effect on attack success (pp), with 95% interval from the\n"
              f"between-run standard error of a two-run contrast ($\\sigma\\sqrt{{2}}$ = {SE:.1f} pp)")
ax.set_ylim(-0.6, len(EFF) - 0.3); ax.grid(axis="x", color="0.92", lw=0.7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pair_power.pdf"); plt.close(fig)

# =========================== FIGURE 4 =====================================
FITS = [("6 conditions, clustered by condition\n(one run each, 600 obs.)", 0.1517, 0.0797, 0.2237),
        ("7 conditions, clustered by condition\n(+ the re-run, 700 obs.)",  0.1007, -0.0063, 0.2076),
        ("10 runs, clustered by training run\n(13 conditions, 1300 obs.)",  0.0986, 0.0302, 0.1670)]
fig, ax = plt.subplots(figsize=(6.2, 2.7))
for i, (lab, b, lo, hi) in enumerate(FITS):
    y = len(FITS) - 1 - i
    col = C_REPL if i == len(FITS) - 1 else C_ORIG
    ax.plot([lo, hi], [y, y], color=col, lw=1.8)
    ax.scatter(b, y, s=56, color=col, zorder=3, edgecolor="white", linewidth=0.8)
    ax.annotate(f"{b:+.4f}  [{lo:+.3f}, {hi:+.3f}]", (b, y),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
ax.axvline(0, color="0.2", lw=1, ls="--")
ax.set_yticks(range(len(FITS))); ax.set_yticklabels([f[0] for f in FITS][::-1], fontsize=7.5)
ax.set_xlabel(r"$\beta_1$: change in $\log_{10}\epsilon_{50}$ per decade of training audio,"
              "\nat matched clean word error rate")
ax.set_xlim(-0.06, 0.28); ax.set_ylim(-0.6, len(FITS) - 0.25)
ax.grid(axis="x", color="0.92", lw=0.7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_ancova.pdf"); plt.close(fig)

# =========================== FIGURE 5 =====================================
# analytic: n = 15.698 sigma^2 / delta^2. Band from the chi-square CI on sigma.
K = 2 * (1.959964 + 0.841621) ** 2
d = np.linspace(10, 70, 300)
fig, ax = plt.subplots(figsize=(6.2, 3.1))
ax.fill_between(d, K*12.15**2/d**2, K*58.27**2/d**2, color=C_ORIG, alpha=0.13,
                label=r"from the 95% interval on $\sigma$  [12.2, 58.3] pp")
ax.plot(d, K*21.57**2/d**2, color=C_ORIG, lw=1.8,
        label=r"from $\sigma$ = 21.6 pp (measured, 4 d.f.)")
for lab, e, n in [("pair 3\n13 pp", 13.0, 44), ("pair 2\n33 pp", 33.0, 7), ("pair 1\n38 pp", 38.0, 6)]:
    ax.scatter(e, K*21.57**2/e**2, s=46, color=C_REPL, zorder=4, edgecolor="white", linewidth=0.8)
    ax.annotate(f"{lab}\n{n} runs", (e, K*21.57**2/e**2), textcoords="offset points",
                xytext=(8, 6), fontsize=7.5, color=C_REPL)
ax.axhline(2, color="0.5", lw=0.9, ls=":")
ax.annotate("2 runs per condition", (66, 2.2), ha="right", fontsize=7.5, color="0.4")
ax.set_yscale("log"); ax.set_ylim(1, 400); ax.set_xlim(10, 70)
ax.set_xlabel("effect to be detected (pp of attack success)")
ax.set_ylabel("training runs needed per condition")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.grid(color="0.93", lw=0.7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_power.pdf"); plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(OUT)): print("  ", f, os.path.getsize(f"{OUT}/{f}"), "bytes")
