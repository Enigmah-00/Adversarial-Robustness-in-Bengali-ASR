#!/usr/bin/env python3
"""The reviewer-requested reanalysis, with no third-party dependencies.

Everything here runs on the committed CSVs in data/results/ using only the
standard library, so it reproduces on any Python 3.8+ without a package index.
That is deliberate: the original analysis ran in a Kaggle notebook whose library
versions could not be pinned, and this script exists partly so the numbers can be
rechecked without reproducing that environment.

Run:  python3 analysis/reanalysis.py [path-to-repo-root]

Sections, each answering one request:

  1  attack-measurement variance, separated from run-to-run variance
  2  sigma under three pooling choices
  3  the ANCOVA refit: CR1, wild cluster bootstrap, two-way clustering
  4  a run-level regression, so the reader sees how few points carry the slope
  5  sensitivity: drop the collinear milestones, add a library-version term
  6  sensitivity to the censoring ceiling
  7  perturbation audibility (SNR) and bisection resolution
  8  the unmatched ladder, redone with replicate means
"""
import csv, glob, io, itertools, json, math, os, random, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(ROOT, "data", "results")
CEIL = 2e-3          # the eps50 ceiling; censored rows are imputed here
SEED = 20261029      # bootstrap seed, fixed so the numbers are reproducible
B = 9999             # bootstrap replications

# ---------------------------------------------------------------------------
# Condition metadata. `hours` is EXPOSURE (distinct audio seen), not the rung
# label, which is why a milestone of the 100 h run is 38.8 h and not 98 h.
# `wer` is clean WER on the same 100 utterances the attack is measured on.
# `trun` is the TRAINING RUN, and it is the cluster: the three 100 h milestones
# are three checkpoints of ONE run and are not independent of each other.
# ---------------------------------------------------------------------------
COND = {
    "1h final":     dict(f="eps50_rung1_final_T5_n100.csv",            hours=0.98,  wer=0.6355, trun="1h-r1",    env="tf4.44.2", seed=1337, rung=1,   grp="rung1-final"),
    "1h run2":      dict(f="eps50_rung1-trajfinal_T5_n100.csv",        hours=0.98,  wer=0.6251, trun="1h-r2",    env="tf4.44.2", seed=1337, rung=1,   grp="rung1-final"),
    "1h s1338":     dict(f="eps50_rung1-s1338_T5_n100.csv",            hours=0.98,  wer=0.5905, trun="1h-s1338", env="tf5.16.1", seed=1338, rung=1,   grp="rung1-final"),
    "1h s1339":     dict(f="eps50_rung1-s1339_T5_n100.csv",            hours=0.98,  wer=0.6240, trun="1h-s1339", env="tf5.16.1", seed=1339, rung=1,   grp="rung1-final"),
    "5h final":     dict(f="eps50_rung5_final_T5_n100.csv",            hours=4.90,  wer=0.4233, trun="5h-r1",    env="tf4.44.2", seed=1337, rung=5,   grp="rung5-final"),
    "5h s1338":     dict(f="eps50_rung5-s1338_T5_n100.csv",            hours=4.90,  wer=0.4152, trun="5h-s1338", env="tf5.16.1", seed=1338, rung=5,   grp="rung5-final"),
    "5h s1339":     dict(f="eps50_rung5-s1339_T5_n100.csv",            hours=4.90,  wer=0.4187, trun="5h-s1339", env="tf5.16.1", seed=1339, rung=5,   grp="rung5-final"),
    "25h final":    dict(f="eps50_rung25_final_T5_n100.csv",           hours=24.60, wer=0.3114, trun="25h-r1",   env="tf4.44.2", seed=1337, rung=25,  grp="rung25-final"),
    "25h s1338":    dict(f="eps50_rung25-s1338_T5_n100.csv",           hours=24.60, wer=0.2987, trun="25h-s1338",env="tf5.16.1", seed=1338, rung=25,  grp="rung25-final"),
    "25h s1339":    dict(f="eps50_rung25-s1339_T5_n100.csv",           hours=24.60, wer=0.2907, trun="25h-s1339",env="tf5.16.1", seed=1339, rung=25,  grp="rung25-final"),
    "100h ms-1000": dict(f="eps50_rung100_milestone-1000_T5_n100.csv", hours=38.80, wer=0.6943, trun="100h-r1",  env="tf4.44.2", seed=1337, rung=100, grp=None),
    "100h ms-2000": dict(f="eps50_rung100_milestone-2000_T5_n100.csv", hours=77.70, wer=0.4752, trun="100h-r1",  env="tf4.44.2", seed=1337, rung=100, grp=None),
    "100h ms-3500": dict(f="eps50_rung100_milestone-3500_T5_n100.csv", hours=98.02, wer=0.3449, trun="100h-r1",  env="tf4.44.2", seed=1337, rung=100, grp=None),
}
# Attacked twice, same weights. These give the ATTACK-measurement variance that
# the published sigma silently contains.
# NOTE (2026-11-14): this is ONE pair, and it is the 1h-traj pair that the paper
# excludes as an outlier -- its 20 pp gap is one-directional (McNemar p=0.0045),
# its eps50 disagreement is 1.5 brackets against 1.0 for every other pair, and
# its transformers version was never recorded. The figure printed below (17.72 pp,
# df 1) is therefore NOT the paper's attack-measurement sd. The paper reports
# 2.91 pp on df 3 from the three version-controlled pairs, which is computed by
# analysis/attack_replication.py from the attackrep_*.csv files. Run that script
# for Section VII-C; this section is kept because it shows what a single pair
# supports on its own, which is the point the paper makes about df 1.
REATTACK = {
    "1h run2": "eps50_rung1-traj-SECOND-MEASUREMENT_T5_n100.csv",
}
# Published ASR, re-derived below as a self-check on the file-to-label mapping.
EXPECT_ASR = {"1h final": 75.0, "1h run2": 41.0, "1h s1338": 57.0, "1h s1339": 39.0,
              "5h final": 79.0, "5h s1338": 96.0, "5h s1339": 59.0,
              "25h final": 66.0, "25h s1338": 78.0, "25h s1339": 57.0,
              "100h ms-1000": 37.0, "100h ms-2000": 46.0, "100h ms-3500": 53.0}
PAIR_EFFECTS = {1: ("40x", 38.0), 5: ("16x", 33.0), 25: ("4x", 13.0)}

# ---------------------------------------------------------------------------
# The 100-hour replicate: seed 1338, trained by NB-19, attacked by NB-21 in
# three sessions (2026-11-03/04). This is the one thing the 2026-10-29 run of
# this script listed as impossible, and it is the reason to rerun it: these
# three conditions are a SECOND training run spanning 38.80, 77.70 and 98.02 h,
# so exposure above 24.6 hours is no longer supplied by a single cluster.
#
# grp is None, exactly as for the s1337 milestones. That is deliberate: the
# three milestones of one trajectory are three EXPOSURES, not three replicates
# of one condition, so they must never enter a sigma pooling group keyed on
# rung. Leaving grp None keeps every number in section 2 unchanged, and the
# rung-100 sigma is computed from explicit milestone pairs in NB-22 instead.
#
# Set INCLUDE_100H_REPLICATE=0 to reproduce the 2026-10-29 output exactly.
# ---------------------------------------------------------------------------
REPLICATE_100H = {
    "100h s1338 ms-1000": dict(f="eps50_rung100-s1338-ms1000_T5_n100.csv", hours=38.80, wer=0.6955, trun="100h-s1338", env="tf5.20.2", seed=1338, rung=100, grp=None),
    "100h s1338 ms-2000": dict(f="eps50_rung100-s1338-ms2000_T5_n100.csv", hours=77.70, wer=0.4960, trun="100h-s1338", env="tf5.20.2", seed=1338, rung=100, grp=None),
    "100h s1338 ms-3500": dict(f="eps50_rung100-s1338-ms3500_T5_n100.csv", hours=98.02, wer=0.3495, trun="100h-s1338", env="tf5.20.2", seed=1338, rung=100, grp=None),
}
REPLICATE_100H_ASR = {"100h s1338 ms-1000": 34.0,
                      "100h s1338 ms-2000": 48.0,
                      "100h s1338 ms-3500": 56.0}

# The milestone pairs, for the rung-dependent sigma. Keyed by optimiser step so
# a milestone is never paired with a different exposure.
RUNG100_PAIRS = [
    ("step 1000", 38.80, "100h ms-1000", "100h s1338 ms-1000"),
    ("step 2000", 77.70, "100h ms-2000", "100h s1338 ms-2000"),
    ("step 3500", 98.02, "100h ms-3500", "100h s1338 ms-3500"),
]

INCLUDE_100H_REPLICATE = os.environ.get("INCLUDE_100H_REPLICATE", "1") != "0"
if INCLUDE_100H_REPLICATE:
    COND.update(REPLICATE_100H)
    EXPECT_ASR.update(REPLICATE_100H_ASR)


H1 = lambda s: print("\n" + "=" * 78 + "\n  " + s + "\n" + "=" * 78)
H2 = lambda s: print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


# ===========================================================================
# plumbing
# ===========================================================================
def load(fn):
    """One eps50 CSV -> rows. Censored rows are imputed at the ceiling."""
    p = os.path.join(RES, fn)
    out = []
    with io.open(p, encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            cens = str(d["censored"]).strip().lower() in ("true", "1", "yes")
            raw = d.get("eps50", "").strip()
            if cens:
                eps = CEIL
            else:
                # A non-censored row with no eps50 is a corrupt file, never a
                # success. Imputing it would invent a measurement.
                assert raw, "%s: row %s is not censored but carries no eps50" % (fn, d["utt_id"])
                eps = float(raw)
            snr = d.get("snr_db", "").strip()
            bw = d.get("bracket_width", "").strip()
            out.append(dict(utt=d["utt_id"], eps=eps, cens=cens,
                            snr=float(snr) if snr else None,
                            bw=float(bw) if bw else None,
                            dur=float(d["duration"]) if d.get("duration") else None))
    return out


def c4(df):
    """E[s]/sigma for a chi-distributed s on `df` degrees of freedom."""
    return math.exp(0.5 * math.log(2.0 / df) + math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))


def pooled_sd(groups):
    """Pooled within-group sd and its df. `groups` is a list of lists."""
    num = den = 0.0
    for a in groups:
        if len(a) < 2:
            continue
        m = sum(a) / len(a)
        num += sum((x - m) ** 2 for x in a)
        den += len(a) - 1
    return (num / den) ** 0.5 if den else None, int(den)


def solve(A, b):
    """Gaussian elimination with partial pivoting. A is square, b a vector."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        assert abs(M[p][i]) > 1e-14, "singular design matrix"
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r == i:
                continue
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def inv(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols(X, y):
    k = len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    beta = solve(XtX, Xty)
    resid = [y[i] - sum(X[i][a] * beta[a] for a in range(k)) for i in range(len(X))]
    return beta, resid, inv(XtX)


def cr1(X, y, beta, resid, XtXi, clusters):
    """Cluster-robust (CR1) covariance, the estimator the paper reports."""
    k, n = len(X[0]), len(X)
    byc = {}
    for i, g in enumerate(clusters):
        byc.setdefault(g, []).append(i)
    G = len(byc)
    meat = [[0.0] * k for _ in range(k)]
    for idx in byc.values():
        u = [sum(X[i][a] * resid[i] for i in idx) for a in range(k)]
        for a in range(k):
            for b in range(k):
                meat[a][b] += u[a] * u[b]
    adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = [[adj * sum(XtXi[a][p] * meat[p][q] * XtXi[q][b] for p in range(k) for q in range(k))
          for b in range(k)] for a in range(k)]
    return V, G


def tcrit(df, p=0.975):
    """Two-sided 95% t quantile. Table for the small df we use, then a
    Cornish-Fisher expansion, which is accurate to <0.5% above df 10."""
    T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000}
    df = int(round(df))
    if df in T:
        return T[df]
    if df > 60:
        return 1.96
    z = 1.959964
    return z + (z ** 3 + z) / (4.0 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df ** 2)


# ===========================================================================
# 0 · load, and check the mapping against the published ASR
# ===========================================================================
H1("0 - data, and a check that each file carries the ASR the paper reports")
if INCLUDE_100H_REPLICATE:
    print("\n  100-hour replicate INCLUDED: 16 conditions, 12 training-run clusters.")
    print("  Set INCLUDE_100H_REPLICATE=0 to reproduce the 2026-10-29 output.")
else:
    print("\n  100-hour replicate EXCLUDED: reproducing the 2026-10-29 output.")
DATA, bad = {}, []
for name, m in COND.items():
    rows = load(m["f"])
    # Every attacked condition is 100 utterances. A partial file (the batches are
    # duration-sorted, so its missing rows are always the longest) must stop the
    # run here, not slip through the ASR tolerance below.
    assert len(rows) == 100, "%s: %s has %d rows, expected 100 -- a partial?" % (
        name, m["f"], len(rows))
    DATA[name] = rows
    asr = 100.0 * sum(1 for r in rows if not r["cens"]) / len(rows)
    exp = EXPECT_ASR[name]
    flag = "ok" if abs(asr - exp) < 0.75 else "MISMATCH"
    if flag != "ok":
        bad.append((name, exp, asr))
    print("  %-14s n=%3d  ASR %5.1f%%  (paper %5.1f)  %s" % (name, len(rows), asr, exp, flag))
    m["asr"] = asr
assert not bad, "file-to-label mapping is wrong for: %s" % bad
print("\n  All %d conditions reproduce the published attack-success figures." % len(COND))


# ===========================================================================
# 1 · attack-measurement variance
# ===========================================================================
H1("1 - how much of sigma is the ATTACK, not the training?")
print("""
  The published sigma treats every difference between two runs of one condition
  as training variance. It is not: each run was attacked once, so the figure also
  contains the variance of the attack itself. Where the same weights were
  attacked twice, that component is directly visible.
""")
att = []
for name, fn in REATTACK.items():
    a1 = COND[name]["asr"]
    rows2 = load(fn)
    a2 = 100.0 * sum(1 for r in rows2 if not r["cens"]) / len(rows2)
    d = abs(a1 - a2)
    att.append([a1, a2])
    print("  %-14s same weights, attacked twice: %.1f%% and %.1f%%  ->  %.1f pp apart"
          % (name, a1, a2, d))
sd_att, df_att = pooled_sd(att)
sd_att_c = sd_att / c4(df_att) if sd_att else None
print("\n  attack-measurement sd = %.2f pp raw, %.2f pp bias-corrected (df %d)"
      % (sd_att, sd_att_c, df_att))
print("  df %d is ONE pair. The point estimate is near-worthless on its own; what" % df_att)
print("  it establishes is that the component is not negligible.")

TOT = 21.574
H2("implied split, total sigma = %.3f pp" % TOT)
print("  attack sd | training sd | share of VARIANCE that is attack noise")
for a in (8.0, 12.0, sd_att_c, 16.0, 20.0):
    t2 = TOT ** 2 - a ** 2
    t = t2 ** 0.5 if t2 > 0 else float("nan")
    print("    %5.1f pp |   %6.2f pp  | %4.0f%%%s"
          % (a, t, 100 * a * a / TOT ** 2, "   <- measured, df %d" % df_att if abs(a - sd_att_c) < 1e-9 else ""))
print("""
  Read sigma as an UPPER BOUND on run-to-run variance. The two-run contrast that
  the matched pairs rely on still carries sigma*sqrt(2) = %.2f pp whichever way
  the split falls, so the negative result does not depend on the decomposition --
  but the attribution to "retraining" does, and cannot be sustained.""" % (TOT * 2 ** 0.5))


# ===========================================================================
# 2 · sigma under three pooling choices
# ===========================================================================
H1("2 - sigma depends on the pooling choice; here are all three")
reps = {}
for name, m in COND.items():
    if m["grp"]:
        reps.setdefault((m["grp"], m["env"]), []).append(name)

choices = []
g_env = [[COND[n]["asr"] for n in v] for v in reps.values()]
raw, df = pooled_sd(g_env)
choices.append(("within (group, environment)  [reported]", raw, df, raw / c4(df)))

g_rung = {}
for name, m in COND.items():
    if m["grp"]:
        g_rung.setdefault(m["rung"], []).append(m["asr"])
raw, df = pooled_sd(list(g_rung.values()))
choices.append(("by rung, pooling environments", raw, df, raw / c4(df)))

g_sc = [[COND[n]["asr"] for n in v] for k, v in reps.items()
        if len({COND[n]["seed"] for n in v}) > 1]
raw, df = pooled_sd(g_sc)
choices.append(("seed-contrast groups only", raw, df, raw / c4(df)))

print("  %-38s %8s %4s %9s %11s" % ("pooling", "raw sd", "df", "sigma", "sigma*sqrt2"))
for lab, raw, df, s in choices:
    print("  %-38s %7.3f %4d %8.3f %10.2f" % (lab, raw, df, s, s * 2 ** 0.5))

print("\n  Each matched-pair effect against each noise estimate:")
print("  %-38s %8s %8s %8s" % ("pooling", "40x/38pp", "16x/33pp", "4x/13pp"))
for lab, raw, df, s in choices:
    n2 = s * 2 ** 0.5
    print("  %-38s %7.2fx %7.2fx %7.2fx" % (lab, 38.0 / n2, 33.0 / n2, 13.0 / n2))
print("""
  No pair reaches 2x the noise under any choice, so "not distinguishable" holds
  throughout -- but it holds by a wide margin under the reported estimate and a
  narrow one under the by-rung estimate. The correct wording is "cannot be
  distinguished given our noise estimate", not "is noise".

  The by-rung estimate has more df and is the more efficient one. It is not the
  headline because each pair it adds differs in LIBRARY VERSION as well as in
  run, so it mixes run-to-run variance with a 4.44.2-to-5.16.1 difference. That
  is a real reason, and it is now stated rather than left implicit.""")


# ===========================================================================
# 3 · the ANCOVA, refit
# ===========================================================================
H1("3 - log10(eps50) ~ log10(exposure) + clean WER, clustered on the training run")
X, Y, CL, UT = [], [], [], []
for name, m in COND.items():
    for r in DATA[name]:
        X.append([1.0, math.log10(m["hours"]), m["wer"]])
        Y.append(math.log10(r["eps"]))
        CL.append(m["trun"])
        UT.append(r["utt"])
beta, resid, XtXi = ols(X, Y)
V, G = cr1(X, Y, beta, resid, XtXi, CL)
se = V[1][1] ** 0.5
t = beta[1] / se
df = G - 1
lo, hi = beta[1] - tcrit(df) * se, beta[1] + tcrit(df) * se
print("  n = %d utterance observations, G = %d training-run clusters" % (len(X), G))
print("  beta1 = %+.4f per decade   CR1 se %.4f   t %.3f on df %d" % (beta[1], se, t, df))
print("  95%% CI [%+.4f, %+.4f]   ->  eps50 x %.2f per decade [%.2f, %.2f]"
      % (lo, hi, 10 ** beta[1], 10 ** lo, 10 ** hi))
print("  WER coefficient = %+.4f" % beta[2])

H2("wild cluster bootstrap, Rademacher weights, H0: beta1 = 0")
print("  CR1 with %d clusters is anti-conservative; the bootstrap is the standard" % len(set(CL)))
print("""  remedy below ~30 clusters. The null is imposed by refitting without
  log10(exposure), then resampling cluster-wise sign flips.""")
Xr = [[row[0], row[2]] for row in X]
br, rr, _ = ols(Xr, Y)
fit_r = [br[0] * 1.0 + br[1] * row[2] for row in X]
cl_ids = sorted(set(CL))
rng = random.Random(SEED)
cnt = 0
for _ in range(B):
    w = {g: (1.0 if rng.random() < 0.5 else -1.0) for g in cl_ids}
    ys = [fit_r[i] + w[CL[i]] * rr[i] for i in range(len(X))]
    b2, r2, Xi2 = ols(X, ys)
    V2, _ = cr1(X, ys, b2, r2, Xi2, CL)
    if abs(b2[1] / V2[1][1] ** 0.5) >= abs(t):
        cnt += 1
p_wcb = (cnt + 1.0) / (B + 1.0)
print("  bootstrap p = %.4f   (%d of %d replications at least as extreme)" % (p_wcb, cnt, B))
print("  verdict: %s at the 5%% level" % ("SIGNIFICANT" if p_wcb < 0.05 else "NOT significant"))

# Rademacher weights admit only 2^G distinct sign patterns. At G <= 16 that is
# few enough to enumerate exactly, which removes simulation error from the
# p-value. Webb six-point weights are the standard recommendation at G <= 12,
# where Rademacher's 2^G support makes the bootstrap p-value coarse.
_cl = sorted(set(CL))
_Xr = [[row[0], row[2]] for row in X]
_br, _rr, _ = ols(_Xr, Y)
_fit = [_br[0] + _br[1] * row[2] for row in X]

def _wcb_p(weight_sets):
    cnt = tot = 0
    for w in weight_sets:
        ys = [_fit[i] + w[CL[i]] * _rr[i] for i in range(len(X))]
        b2, r2, Xi2 = ols(X, ys)
        V2, _g = cr1(X, ys, b2, r2, Xi2, CL)
        tot += 1
        if abs(b2[1] / V2[1][1] ** 0.5) >= abs(t) - 1e-12:
            cnt += 1
    return (cnt + 1.0) / (tot + 1.0), cnt, tot

if len(_cl) <= 16:
    _ex = (dict(zip(_cl, _s)) for _s in itertools.product([1.0, -1.0], repeat=len(_cl)))
    p_exact, _c, _t = _wcb_p(_ex)
    print("  exact enumeration of all %d Rademacher sign patterns: p = %.4f (%d)"
          % (_t, p_exact, _c))
else:
    p_exact = None

_W6 = [-(1.5 ** 0.5), -1.0, -(0.5 ** 0.5), (0.5 ** 0.5), 1.0, (1.5 ** 0.5)]
_rngw = random.Random(SEED)
p_webb, _c, _t = _wcb_p([{k: _rngw.choice(_W6) for k in _cl} for _ in range(B)])
print("  Webb six-point weights, %d draws: p = %.4f (%d)" % (_t, p_webb, _c))
print("  verdict on the exact Rademacher p: %s at the 5%% level"
      % ("SIGNIFICANT" if (p_exact or p_wcb) < 0.05 else "NOT significant"))

H2("leave-one-cluster-out: which runs carry the slope?")
print("  With G = %d the slope can rest on very few clusters. Dropping each in" % len(set(CL)))
print("  turn shows how few. Note the exposure values per cluster first.")
_seen = {}
for name, m in COND.items():
    _seen.setdefault(m["trun"], set()).add(m["hours"])
for k in sorted(_seen, key=lambda k: min(_seen[k])):
    print("    %-12s exposure %s" % (k, sorted(_seen[k])))
_top = sorted(k for k in _seen if max(_seen[k]) > 24.6)
if len(_top) == 1:
    print("    Every exposure above 24.6 h sits in ONE cluster (%s)." % _top[0])
else:
    print("    Exposures above 24.6 h are supplied by %d independent clusters: %s."
          % (len(_top), ", ".join(_top)))
print()
_loco_incl = []
for drop in sorted(set(CL)):
    idx = [i for i in range(len(X)) if CL[i] != drop]
    ba, ra, Xia = ols([X[i] for i in idx], [Y[i] for i in idx])
    Va, Ga = cr1([X[i] for i in idx], [Y[i] for i in idx], ba, ra, Xia, [CL[i] for i in idx])
    sa = Va[1][1] ** 0.5; da = Ga - 1
    loa, hia = ba[1] - tcrit(da) * sa, ba[1] + tcrit(da) * sa
    print("    drop %-12s beta1 %+8.4f  CI [%+.4f, %+.4f]  %s"
          % (drop, ba[1], loa, hia, "excludes 0" if loa * hia > 0 else "INCLUDES 0"))
    if loa * hia <= 0:
        _loco_incl.append(drop)
print()
if not _loco_incl:
    print("    No single cluster decides it: every one of the %d drops leaves an" % len(set(CL)))
    print("    interval that excludes zero. Note this is still CR1, which is")
    print("    anti-conservative at this G; the bootstrap above is the stricter test.")
elif len(_loco_incl) == 1 and len(_top) == 1 and _loco_incl[0] == _top[0]:
    print("    One cluster decides it. Drop %s and the interval includes zero;" % _loco_incl[0])
    print("    drop any of the other %d and it does not. beta1 is identified by a" % (len(set(CL)) - 1))
    print("    single training run, because that run supplies every exposure level above")
    print("    24.6 hours. This is the same fact the bootstrap is reacting to: a sign flip")
    print("    on one cluster moves the whole top of the range.")
else:
    print("    Dropping any of %s puts zero inside the interval." % ", ".join(_loco_incl))

H2("two-way clustering on (training run) and (utterance)")
print("""  The same 100 utterances appear in every run, so utterance effects are
  crossed with run effects. Two-way: V_run + V_utt - V_intersection.""")
Vr, Gr = cr1(X, Y, beta, resid, XtXi, CL)
Vu, Gu = cr1(X, Y, beta, resid, XtXi, UT)
both = ["%s|%s" % (a, b) for a, b in zip(CL, UT)]
Vb, Gb = cr1(X, Y, beta, resid, XtXi, both)
v2 = Vr[1][1] + Vu[1][1] - Vb[1][1]
print("  G_run %d, G_utt %d, G_both %d" % (Gr, Gu, Gb))
print("  se: run-only %.4f | utterance-only %.4f | two-way %.4f"
      % (Vr[1][1] ** 0.5, Vu[1][1] ** 0.5, v2 ** 0.5 if v2 > 0 else float("nan")))
if v2 > 0:
    se2 = v2 ** 0.5
    d2 = min(Gr, Gu) - 1
    print("  beta1 %+.4f, t %.3f on df %d, 95%% CI [%+.4f, %+.4f]"
          % (beta[1], beta[1] / se2, d2, beta[1] - tcrit(d2) * se2, beta[1] + tcrit(d2) * se2))
else:
    print("  two-way variance is negative, which happens in small samples; report")
    print("  the run-clustered se and say so.")


# ===========================================================================
# 3b · the same fit with NB-25's decision margin added
# ===========================================================================
# NB-25 (2026-11-11, 16 checkpoints x 100 utterances, forward passes only, no
# attack) measured a per-frame decision margin: mean over valid frames of
# (top logit - runner-up logit). Its own output showed the margin correlates
# with attack success (r = +0.43 across checkpoints) but NOT with clean WER
# (r = +0.03), and that the three iso-WER pairs are 2.0-4.6 margin units apart
# despite being matched on WER. That is the definition of an omitted
# covariate, and it is the supervisor's review point 4. This section puts the
# margin in the fit and asks whether beta1 survives.
#
# Section 3 above is untouched, so every published number still reproduces.
# ---------------------------------------------------------------------------
MARG = {}
MBAR = {}
_mfile = os.path.join(RES, "margins_n100.csv")
if os.path.exists(_mfile):
    _acc = {}
    with io.open(_mfile, encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            k = (d["tag"], d["utt_id"])
            assert k not in MARG, "margins_n100.csv: duplicate %s" % (k,)
            MARG[k] = dict(mean=float(d["margin_mean"]),
                           p10=float(d["margin_p10"]),
                           ent=float(d["entropy_mean"]),
                           ctcpf=float(d["ctc_loss_per_frame"]))
            for f in ("mean", "p10", "ent", "ctcpf"):
                _acc.setdefault(d["tag"], {}).setdefault(f, []).append(MARG[k][f])
    for tg, fv in _acc.items():
        MBAR[tg] = {f: sum(v) / len(v) for f, v in fv.items()}

if not MARG:
    H1("3b - decision-margin covariate: margins_n100.csv not present, SKIPPED")
    print("  Put NB-25's margins_n100.csv in data/results/ to enable this section.")
else:
    H1("3b - the same fit with NB-25's decision margin added")
    print("  margins_n100.csv: %d (checkpoint, utterance) rows over %d checkpoints"
          % (len(MARG), len(MBAR)))
    _have = [n for n in COND if n in MBAR]
    _miss = [n for n in COND if n not in MBAR]
    print("  conditions in the fit with a margin: %d of %d" % (len(_have), len(COND)))
    if _miss:
        print("  NO MARGIN, so excluded from every spec in this section: %s"
              % ", ".join(_miss))
        print("  (NB-25 did not attack or score these; a margin needs one forward")
        print("   pass over the 100 utterances, about 1 minute per checkpoint.)")

    def _pear(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((x - mb) ** 2 for x in b) ** 0.5
        return num / (da * db) if da > 0 and db > 0 else float("nan")

    def _vif(Xm, j):
        """1/(1-R^2) from regressing column j on the others. = SST/SSE."""
        yj = [row[j] for row in Xm]
        Xo = [[row[a] for a in range(len(row)) if a != j] for row in Xm]
        _b, r, _ = ols(Xo, yj)
        m = sum(yj) / len(yj)
        sst = sum((v - m) ** 2 for v in yj)
        sse = sum(v * v for v in r)
        return float("inf") if sse <= 1e-300 else sst / sse

    H2("is the margin collinear with exposure?")
    print("""  The milestone checkpoints supply every exposure above 24.6 h AND every
  low margin. If those two facts are the same fact, the design cannot tell
  exposure from margin and a wide interval is the honest answer, not a null.""")
    print()
    _ck = [n for n in _have]
    _lh = [math.log10(COND[n]["hours"]) for n in _ck]
    _wr = [COND[n]["wer"] for n in _ck]
    _mm = [MBAR[n]["mean"] for n in _ck]
    _mp = [MBAR[n]["p10"] for n in _ck]
    _en = [MBAR[n]["ent"] for n in _ck]
    print("  checkpoint-level Pearson r over %d checkpoints:" % len(_ck))
    print("    log10(hours) vs margin_mean   %+.3f" % _pear(_lh, _mm))
    print("    log10(hours) vs margin_p10    %+.3f" % _pear(_lh, _mp))
    print("    log10(hours) vs entropy       %+.3f" % _pear(_lh, _en))
    print("    log10(hours) vs clean WER     %+.3f" % _pear(_lh, _wr))
    print("    clean WER    vs margin_mean   %+.3f" % _pear(_wr, _mm))

    def _run_spec(label, zfun, zlabel, exact=True):
        """Fit log10(eps) ~ log10(hours) + wer [+ z], CR1 + exact WCB + LOCO."""
        Xe, Ye, CLe = [], [], []
        for name, m in COND.items():
            if name not in MBAR:
                continue
            for r in DATA[name]:
                row = [1.0, math.log10(m["hours"]), m["wer"]]
                if zfun is not None:
                    z = zfun(name, r["utt"])
                    assert z is not None, "%s: no %s for %s" % (name, zlabel, r["utt"])
                    row.append(z)
                Xe.append(row)
                Ye.append(math.log10(r["eps"]))
                CLe.append(m["trun"])
        k = len(Xe[0])
        b, rs, Xi = ols(Xe, Ye)
        Ve, Ge = cr1(Xe, Ye, b, rs, Xi, CLe)
        se = Ve[1][1] ** 0.5
        te = b[1] / se
        dfe = Ge - 1
        lo = b[1] - tcrit(dfe) * se
        hi = b[1] + tcrit(dfe) * se
        H2(label)
        print("  n = %d, G = %d clusters, k = %d regressors" % (len(Xe), Ge, k))
        print("  beta1 (log10 exposure) = %+.4f   CR1 se %.4f   t %.3f on df %d"
              % (b[1], se, te, dfe))
        print("  95%% CI [%+.4f, %+.4f]  ->  eps50 x %.2f per decade [%.2f, %.2f]"
              % (lo, hi, 10 ** b[1], 10 ** lo, 10 ** hi))
        print("  clean-WER coefficient  = %+.4f" % b[2])
        if k > 3:
            sez = Ve[3][3] ** 0.5
            print("  %-22s = %+.4f   CR1 se %.4f   t %.3f"
                  % (zlabel + " coefficient", b[3], sez, b[3] / sez))
            print("  VIF  log10(hours) %.2f | clean WER %.2f | %s %.2f"
                  % (_vif(Xe, 1), _vif(Xe, 2), zlabel, _vif(Xe, 3)))
        else:
            print("  VIF  log10(hours) %.2f | clean WER %.2f" % (_vif(Xe, 1), _vif(Xe, 2)))
        p_ex = None
        if exact:
            drop = [i for i in range(k) if i != 1]
            Xr = [[row[a] for a in drop] for row in Xe]
            br, rr, _ = ols(Xr, Ye)
            fitr = [sum(Xr[i][a] * br[a] for a in range(len(drop)))
                    for i in range(len(Xe))]
            cl = sorted(set(CLe))
            cnt = tot = 0
            for sgn in itertools.product([1.0, -1.0], repeat=len(cl)):
                w = dict(zip(cl, sgn))
                ys = [fitr[i] + w[CLe[i]] * rr[i] for i in range(len(Xe))]
                b2, r2, Xi2 = ols(Xe, ys)
                V2, _g = cr1(Xe, ys, b2, r2, Xi2, CLe)
                tot += 1
                if abs(b2[1] / V2[1][1] ** 0.5) >= abs(te) - 1e-12:
                    cnt += 1
            p_ex = (cnt + 1.0) / (tot + 1.0)
            print("  wild cluster bootstrap, exact over all %d sign patterns: p = %.4f"
                  % (tot, p_ex))
            print("  verdict: %s at the 5%% level"
                  % ("SIGNIFICANT" if p_ex < 0.05 else "NOT significant"))
        inc = []
        for d in sorted(set(CLe)):
            idx = [i for i in range(len(Xe)) if CLe[i] != d]
            ba, ra, Xia = ols([Xe[i] for i in idx], [Ye[i] for i in idx])
            Va, Ga = cr1([Xe[i] for i in idx], [Ye[i] for i in idx], ba, ra, Xia,
                         [CLe[i] for i in idx])
            sa = Va[1][1] ** 0.5
            la = ba[1] - tcrit(Ga - 1) * sa
            ha = ba[1] + tcrit(Ga - 1) * sa
            if la * ha <= 0:
                inc.append(d)
        print("  leave-one-cluster-out: %s"
              % ("no drop puts zero in the CR1 interval"
                 if not inc else "zero enters when dropping " + ", ".join(inc)))
        return dict(label=label, beta1=b[1], se=se, lo=lo, hi=hi, p=p_ex,
                    bz=(b[3] if k > 3 else None), n=len(Xe),
                    vif1=_vif(Xe, 1), loco=inc)

    _S = []
    _S.append(_run_spec(
        "spec A - published spec, no margin (reference on the same conditions)",
        None, ""))
    _S.append(_run_spec(
        "spec B - + checkpoint-mean margin  (the direct analogue of clean WER)",
        lambda n, u: MBAR[n]["mean"], "margin_mean"))
    _S.append(_run_spec(
        "spec C - + per-utterance margin  (absorbs utterance difficulty too)",
        lambda n, u: MARG[(n, u)]["mean"], "margin(utt)"))
    _S.append(_run_spec(
        "spec D - + checkpoint-mean margin_p10  (the hardest decile of frames)",
        lambda n, u: MBAR[n]["p10"], "margin_p10"))
    _S.append(_run_spec(
        "spec E - + checkpoint-mean output entropy",
        lambda n, u: MBAR[n]["ent"], "entropy"))

    H2("summary: does the exposure slope survive the margin?")
    print("  %-46s %9s %9s %18s %7s" % ("spec", "beta1", "x/decade", "95% CI", "WCB p"))
    for s in _S:
        print("  %-46s %+9.4f %9.2f  [%+.4f,%+.4f] %7.4f"
              % (s["label"].split(" - ")[0] + " " + s["label"].split(" - ")[1][:38],
                 s["beta1"], 10 ** s["beta1"], s["lo"], s["hi"],
                 s["p"] if s["p"] is not None else float("nan")))
    print()
    _a, _b = _S[0], _S[1]
    _shrink = (1.0 - _b["beta1"] / _a["beta1"]) * 100.0 if _a["beta1"] else float("nan")
    print("  Adding the checkpoint-mean margin moves beta1 from %+.4f to %+.4f,"
          % (_a["beta1"], _b["beta1"]))
    print("  a change of %+.1f%% of the reference estimate, and the CR1 standard" % -_shrink)
    print("  error goes %.4f -> %.4f (VIF on log10(hours) %.2f -> %.2f)."
          % (_a["se"], _b["se"], _a["vif1"], _b["vif1"]))
    _sgn = [s for s in _S[1:] if s["beta1"] * _a["beta1"] > 0]
    _sig = [s for s in _S if s["p"] is not None and s["p"] < 0.05]
    print("  %d of the %d margin specs keep beta1's sign; %d of %d specs reach"
          % (len(_sgn), len(_S) - 1, len(_sig), len(_S)))
    print("  p < 0.05 on the exact bootstrap.")
    print("""
  Read it this way. The pre-registered rule in paper/NUMBERS.md fixes the
  wording from the exact bootstrap p, so apply that rule to the spec the
  review asked for (spec B), not to spec A.""")


# ===========================================================================
# 4 · a run-level regression
# ===========================================================================
H1("4 - the same fit on run-level means, so the reader sees how few points there are")
print("  %d condition means, %d independent training runs. This is the honest" % (len(COND), len({m["trun"] for m in COND.values()})))
print("  picture of the evidence: the utterance-level n of %d is not that many" % sum(len(DATA[n]) for n in COND))
print("  independent observations.")
Xr2, Yr2, lab = [], [], []
for name, m in COND.items():
    mu = sum(math.log10(r["eps"]) for r in DATA[name]) / len(DATA[name])
    Xr2.append([1.0, math.log10(m["hours"]), m["wer"]])
    Yr2.append(mu)
    lab.append(name)
b3, r3, Xi3 = ols(Xr2, Yr2)
n3, k3 = len(Xr2), 3
s2 = sum(e * e for e in r3) / (n3 - k3)
se3 = (s2 * Xi3[1][1]) ** 0.5
d3 = n3 - k3
print("  n = %d condition means, df %d" % (n3, d3))
print("  beta1 = %+.4f   se %.4f   t %.3f   95%% CI [%+.4f, %+.4f]"
      % (b3[1], se3, b3[1] / se3, b3[1] - tcrit(d3) * se3, b3[1] + tcrit(d3) * se3))
print("  %s at the 5%% level  (OLS on the means: treats every milestone mean as"
      % ("significant" if abs(b3[1] / se3) > tcrit(d3) else "NOT significant"))
print("  an independent point, which the milestones of one run are not)")
# The milestone means of one training run are not independent, so the same fit
# with its standard error clustered on the training run is the honest version.
_trc = [COND[n]["trun"] for n in lab]
V3c, G3c = cr1(Xr2, Yr2, b3, r3, Xi3, _trc)
se3c = V3c[1][1] ** 0.5
d3c = G3c - 1
lo3c, hi3c = b3[1] - tcrit(d3c) * se3c, b3[1] + tcrit(d3c) * se3c
print("  same fit, se clustered on training run (G = %d, df %d):" % (G3c, d3c))
print("  beta1 = %+.4f   se %.4f   t %.3f   95%% CI [%+.4f, %+.4f]   %s"
      % (b3[1], se3c, b3[1] / se3c, lo3c, hi3c,
         "excludes 0" if lo3c * hi3c > 0 else "INCLUDES 0"))


# ===========================================================================
# 5 · sensitivity: the collinear milestones, and library version
# ===========================================================================
H1("5 - what happens without the 100 h milestones, and with a version term")
_ms_runs = sorted({m["trun"] for m in COND.values() if not m["grp"]})
print("""  The 100 h milestones are the top of the exposure range and the most
  leveraged points in the slope. They are checkpoints at steps 1000, 2000 and
  3500 of %d training run(s): %s. Within every such run, exposure is
  perfectly collinear with training step, so more audio and more optimisation
  cannot be told apart. A second run adds an independent cluster; it does NOT
  break that collinearity, because both runs were checkpointed at the same steps.
  The version indicator below is pre/post the library upgrade (4.44.2 vs later).""" % (len(_ms_runs), ", ".join(_ms_runs)))

def refit(keep, note, extra=None):
    Xa, Ya, Ca = [], [], []
    for name in keep:
        m = COND[name]
        for r in DATA[name]:
            row = [1.0, math.log10(m["hours"]), m["wer"]]
            if extra:
                row.append(extra(m))
            Xa.append(row); Ya.append(math.log10(r["eps"])); Ca.append(m["trun"])
    ba, ra, Xia = ols(Xa, Ya)
    Va, Ga = cr1(Xa, Ya, ba, ra, Xia, Ca)
    s = Va[1][1] ** 0.5
    d = Ga - 1
    print("  %-34s beta1 %+.4f  se %.4f  df %2d  CI [%+.4f, %+.4f]  %s"
          % (note, ba[1], s, d, ba[1] - tcrit(d) * s, ba[1] + tcrit(d) * s,
             "excludes 0" if (ba[1] - tcrit(d) * s) * (ba[1] + tcrit(d) * s) > 0 else "INCLUDES 0"))
    return ba, Va, Ga

allc = list(COND)
no_ms = [n for n in allc if COND[n]["grp"]]
refit(allc, "all %d conditions [reported]" % len(allc))
refit(no_ms, "10 replicate runs, no milestones")
refit(allc, "+ library-version indicator", extra=lambda m: 0.0 if m["env"] == "tf4.44.2" else 1.0)
print("""
  Exposure in the no-milestone fit spans 0.98 to 24.6 h, a factor of 25 rather
  than 100, so a wider interval there is expected and is not by itself evidence
  against the slope. What matters is whether the sign and rough size survive.""")


# ===========================================================================
# 6 · the censoring ceiling
# ===========================================================================
H1("6 - sensitivity to the imputation ceiling")
ncens = sum(1 for n in allc for r in DATA[n] if r["cens"])
print("  %d of %d observations are right-censored (%.1f%%) and imputed at %.0e."
      % (ncens, sum(len(DATA[n]) for n in allc), 100.0 * ncens / sum(len(DATA[n]) for n in allc), CEIL))
print("  The cheap check here is whether the slope depends on where the ceiling is")
print("  put; section 6b models the censoring directly.\n")
_c = CEIL
for ceil in (2e-3, 3e-3, 5e-3, 1e-2):
    Xa, Ya, Ca = [], [], []
    for name in allc:
        m = COND[name]
        for r in DATA[name]:
            Xa.append([1.0, math.log10(m["hours"]), m["wer"]])
            Ya.append(math.log10(ceil if r["cens"] else r["eps"]))
            Ca.append(m["trun"])
    ba, ra, Xia = ols(Xa, Ya)
    Va, Ga = cr1(Xa, Ya, ba, ra, Xia, Ca)
    s = Va[1][1] ** 0.5; d = Ga - 1
    print("  ceiling %.0e  beta1 %+.4f  CI [%+.4f, %+.4f]  %s"
          % (ceil, ba[1], ba[1] - tcrit(d) * s, ba[1] + tcrit(d) * s,
             "excludes 0" if (ba[1] - tcrit(d) * s) * (ba[1] + tcrit(d) * s) > 0 else "INCLUDES 0"))
print("\n  The ceiling is a lower bound on the censored values, so every estimate")
print("  here is conservative with respect to the true (larger) eps50.")


H2("6b - censored-normal (Tobit) fit, clustered on the training run")
FLOOR = 1e-4
# The bisection searches [1e-4, CEIL] in 32 log-steps, so a success in the
# lowest step may truly lie below the floor: that row is LEFT-censored at the
# step's upper edge. A row that never reached its target is RIGHT-censored at
# the ceiling. Everything else is treated as exact (resolution ~10%).
FLOOR_EDGE = FLOOR * (CEIL / FLOOR) ** (1.0 / 32)
print("""  Instead of imputing, model the censoring. log10(eps50) is normal around the
  linear predictor; a row that never reached its target contributes
  P(y > log10 ceiling), and a success in the lowest bisection step, whose true
  value may lie below the 1e-4 floor, contributes P(y < that step's upper edge).
  Standard errors are a sandwich clustered on the training run.""")
TX, TY, TK, TC = [], [], [], []      # design, value, kind (0 exact, 1 right, -1 left), cluster
for name in allc:
    m = COND[name]
    for r in DATA[name]:
        TX.append([1.0, math.log10(m["hours"]), m["wer"]])
        TC.append(m["trun"])
        if r["cens"]:
            TY.append(math.log10(CEIL)); TK.append(1)
        elif r["eps"] <= FLOOR_EDGE * (1 + 1e-9):
            TY.append(math.log10(FLOOR_EDGE)); TK.append(-1)
        else:
            TY.append(math.log10(r["eps"])); TK.append(0)
n_right = sum(1 for k in TK if k == 1)
n_left = sum(1 for k in TK if k == -1)
print("  %d exact | %d right-censored at %.0e | %d left-censored below %.4e"
      % (len(TK) - n_right - n_left, n_right, CEIL, n_left, FLOOR_EDGE))

_SQ2 = math.sqrt(2.0)
_LSQ2PI = 0.5 * math.log(2.0 * math.pi)

def _Phi(x):
    return 0.5 * math.erfc(-x / _SQ2)

def _logPhi(x):
    v = _Phi(x)
    if v > 1e-300:
        return math.log(v)
    return -0.5 * x * x - _LSQ2PI - math.log(-x)        # Mills asymptote, x << 0

def _mills(x):
    """phi(x) / Phi(x), stable for very negative x."""
    v = _Phi(x)
    if v > 1e-300:
        return math.exp(-0.5 * x * x - _LSQ2PI) / v
    return -x

def _tobit(theta):
    b0, b1, b2, ls = theta
    sig = math.exp(ls)
    ll, sc = 0.0, []
    for x, y, k in zip(TX, TY, TK):
        z = (y - (b0 * x[0] + b1 * x[1] + b2 * x[2])) / sig
        if k == 0:
            ll += -0.5 * z * z - ls - _LSQ2PI
            gm, gs = z / sig, z * z - 1.0
        elif k == 1:
            ll += _logPhi(-z)
            lam = _mills(-z)
            gm, gs = lam / sig, lam * z
        else:
            ll += _logPhi(z)
            kap = _mills(z)
            gm, gs = -kap / sig, -kap * z
        sc.append([gm * x[0], gm * x[1], gm * x[2], gs])
    return ll, sc

def _grad(theta):
    _, sc = _tobit(theta)
    return [sum(r[j] for r in sc) for j in range(4)]

def _hess(theta, h=1e-5):
    H = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        tp = list(theta); tm = list(theta)
        tp[i] += h; tm[i] -= h
        gp, gm_ = _grad(tp), _grad(tm)
        for j in range(4):
            H[i][j] = (gp[j] - gm_[j]) / (2 * h)
    return [[0.5 * (H[i][j] + H[j][i]) for j in range(4)] for i in range(4)]

# start from the imputed OLS fit
_b, _r, _ = ols(TX, [math.log10(CEIL) if k == 1 else y for y, k in zip(TY, TK)])
theta = [_b[0], _b[1], _b[2], math.log((sum(e * e for e in _r) / (len(_r) - 3)) ** 0.5)]
ll_old, _ = _tobit(theta)
for it in range(60):
    g = _grad(theta)
    H = _hess(theta)
    step = solve(H, [-v for v in g])
    t = 1.0
    while True:
        cand = [theta[i] + t * step[i] for i in range(4)]
        ll_new, _ = _tobit(cand)
        if ll_new >= ll_old - 1e-12 or t < 1e-6:
            break
        t *= 0.5
    theta, done = cand, max(abs(t * v) for v in step) < 1e-10
    ll_old = ll_new
    if done:
        break
assert max(abs(v) for v in _grad(theta)) < 1e-4, "Tobit did not converge"

A = [[-v for v in row] for row in _hess(theta)]
Ai = inv(A)
_, sc = _tobit(theta)
cl_sum = {}
for row, g_ in zip(sc, TC):
    acc = cl_sum.setdefault(g_, [0.0] * 4)
    for j in range(4):
        acc[j] += row[j]
G_t = len(cl_sum)
meat = [[sum(v[i] * v[j] for v in cl_sum.values()) for j in range(4)] for i in range(4)]
mm = lambda P, Q: [[sum(P[i][k] * Q[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
Vt = mm(mm(Ai, meat), Ai)
Vt = [[v * G_t / (G_t - 1.0) for v in row] for row in Vt]
se_t = Vt[1][1] ** 0.5
se_m = Ai[1][1] ** 0.5
d_t = G_t - 1
lo_t, hi_t = theta[1] - tcrit(d_t) * se_t, theta[1] + tcrit(d_t) * se_t
print("  converged in %d iterations, log-likelihood %.3f" % (it + 1, ll_old))
print("  beta1 = %+.4f per decade   cluster se %.4f (model se %.4f)   df %d"
      % (theta[1], se_t, se_m, d_t))
print("  95%% CI [%+.4f, %+.4f]  %s   ->  eps50 x %.2f per decade [%.2f, %.2f]"
      % (lo_t, hi_t, "excludes 0" if lo_t * hi_t > 0 else "INCLUDES 0",
         10 ** theta[1], 10 ** lo_t, 10 ** hi_t))
print("  WER coefficient %+.4f   residual sd %.4f (log10 units)" % (theta[2], math.exp(theta[3])))
print("  imputed OLS (section 3) for comparison: beta1 %+.4f" % beta[1])
print()
if theta[1] > beta[1]:
    print("  Modelling the censoring raises the slope above the imputed fit, which is")
    print("  what section 6 implies: imputing at the ceiling understates it.")
else:
    print("  Modelling the censoring does NOT raise the slope above the imputed fit,")
    print("  contrary to what section 6 implies. Investigate before reporting.")
print("  The cluster-robust se at this G is optimistic for the same reason as CR1;")
print("  read this as a sensitivity check, not a replacement for the bootstrap.")


# ===========================================================================
# 7 · audibility and bisection resolution
# ===========================================================================
# ===========================================================================
# 6c · the censoring runs the WRONG WAY (NB-24, 2026-11-12)
# ===========================================================================
H1("6c - NB-24: censored rows are mostly FALSE NEGATIVES, not resistant")

print("""  Sections 6 and 6b both assume a censored row's true eps50 lies ABOVE the
  ceiling, so imputing at the ceiling is conservative. NB-24 re-attacked the
  censored utterances of rung1-final with a deliberately stronger attack --
  fp32, 5 restarts, 2000 steps, no stall rule, same learning rate at the
  ceiling -- and 22 of 25 of them SUCCEEDED. Their true eps50 is at or below
  2e-3, not above it.

  That inverts the direction of the sensitivity analysis. Section 6 sweeps the
  ceiling up and finds the slope rising; the sweep that matters goes down.""")

PHI_MEASURED = 22.0 / 25.0        # NB-24, rung1-final, Wilson 95% [0.70, 0.96]

_tot_obs = sum(len(DATA[n]) for n in allc)
_ncens = sum(1 for n in allc for r in DATA[n] if r["cens"])
print("\n  %d of %d observations (%.1f%%) carry an imputed value."
      % (_ncens, _tot_obs, 100.0 * _ncens / _tot_obs))
print("  Per-condition censoring, which is what makes the bias differential:")
_cz = []
for name in allc:
    k = sum(1 for r in DATA[name] if r["cens"])
    _cz.append((COND[name]["hours"], name, k, len(DATA[name])))
for h, name, k, n in sorted(_cz):
    print("    %-28s %6.2f h   %3d/%3d censored (%4.1f%%)"
          % (name, h, k, n, 100.0 * k / n))


def _fit_imputed(impute):
    """OLS + CR1 on log10(eps50), with `impute(cond_name, row)` supplying the
    value for a censored row. Returns beta1, lo, hi, df, and the X/Y/CL used."""
    Xa, Ya, Ca = [], [], []
    for name in allc:
        m = COND[name]
        for r in DATA[name]:
            Xa.append([1.0, math.log10(m["hours"]), m["wer"]])
            Ya.append(math.log10(impute(name, r) if r["cens"] else r["eps"]))
            Ca.append(m["trun"])
    ba, ra, Xia = ols(Xa, Ya)
    Va, Ga = cr1(Xa, Ya, ba, ra, Xia, Ca)
    s = Va[1][1] ** 0.5
    d = Ga - 1
    return ba[1], ba[1] - tcrit(d) * s, ba[1] + tcrit(d) * s, d, Xa, Ya, Ca, s


def _wcb(Xa, Ya, Ca, b1, se1):
    """Exact Rademacher wild cluster bootstrap, null imposed by dropping the
    exposure column. Same construction as section 3, rebuilt locally because
    that one closes over the published fit."""
    cl = sorted(set(Ca))
    if len(cl) > 16:
        return None
    Xr = [[row[0], row[2]] for row in Xa]
    br, rr, _ = ols(Xr, Ya)
    fit = [br[0] + br[1] * row[2] for row in Xa]
    t0 = abs(b1 / se1)
    cnt = tot = 0
    for signs in itertools.product([1.0, -1.0], repeat=len(cl)):
        w = dict(zip(cl, signs))
        ys = [fit[i] + w[Ca[i]] * rr[i] for i in range(len(Xa))]
        b2, r2, Xi2 = ols(Xa, ys)
        V2, _g = cr1(Xa, ys, b2, r2, Xi2, Ca)
        tot += 1
        if abs(b2[1] / V2[1][1] ** 0.5) >= t0 - 1e-12:
            cnt += 1
    return (cnt + 1.0) / (tot + 1.0)


# --- 6c-1 · a flat downward sweep -----------------------------------------
H2("6c-1 - impute censored rows BELOW the ceiling, not above it")
print("  A flat sweep, transparent and assumption-free: every censored row is")
print("  imputed at a fixed fraction of the ceiling.\n")
print("    fraction  imputed at    beta1     95% CI            x/decade   WCB p")
for frac in (1.0, 0.75, 0.5, 0.35, 0.25):
    b1, lo, hi, d, Xa, Ya, Ca, se = _fit_imputed(lambda n, r, f=frac: f * CEIL)
    p = _wcb(Xa, Ya, Ca, b1, se)
    print("      %.2f    %.2e   %+.4f  [%+.4f,%+.4f]   %5.2fx   %s%s"
          % (frac, frac * CEIL, b1, lo, hi, 10.0 ** b1,
             "%.4f" % p if p is not None else "  n/a",
             "" if lo * hi > 0 else "   CI INCLUDES 0"))

# --- 6c-2 · the measured false-negative rate ------------------------------
H2("6c-2 - a false-negative mixture at the measured rate")
print("""  With probability phi a censored row is a false negative and behaves like
  the successes actually measured on that same checkpoint; otherwise it sits
  at the ceiling. The expected log10 value is the geometric interpolation
  phi*log10(median success) + (1-phi)*log10(ceiling), which is exactly what a
  regression on log10 needs and introduces no RNG.

  phi = 0 reproduces the published fit. phi = %.2f is NB-24's measured rate at
  rung1-final.\n""" % PHI_MEASURED)

_MED = {}
for name in allc:
    ok = sorted(r["eps"] for r in DATA[name] if not r["cens"])
    _MED[name] = ok[len(ok) // 2] if ok else CEIL

def _mix(name, r, phi):
    lo = math.log10(_MED[name])
    return 10.0 ** (phi * lo + (1.0 - phi) * math.log10(CEIL))

print("    phi     beta1     95% CI            x/decade   WCB p")
for phi in (0.0, 0.25, 0.5, PHI_MEASURED, 1.0):
    b1, lo, hi, d, Xa, Ya, Ca, se = _fit_imputed(lambda n, r, p=phi: _mix(n, r, p))
    p = _wcb(Xa, Ya, Ca, b1, se)
    print("    %.2f   %+.4f  [%+.4f,%+.4f]   %5.2fx   %s%s"
          % (phi, b1, lo, hi, 10.0 ** b1,
             "%.4f" % p if p is not None else "  n/a",
             "" if lo * hi > 0 else "   CI INCLUDES 0"))

# --- 6c-3 · the dangerous case: a DIFFERENTIAL rate -----------------------
H2("6c-3 - how different would the rate have to be to overturn the sign?")
print("""  A uniform false-negative rate attenuates the slope. A rate that DIFFERS
  with exposure confounds it, because the correction then moves the two ends of
  the ladder by different amounts. NB-24 measured 22/25 at 0.98 h and only
  reached 2/4 at the 100 h milestone before the session was killed, which
  settles nothing: those intervals are [0.70,0.96] and [0.15,0.85].

  Here the low-exposure conditions (below the median exposure) take phi_lo and
  the high-exposure ones phi_hi. Correcting the LOW end harder raises the
  slope; correcting the HIGH end harder lowers it.\n""")

_hrs = sorted(COND[n]["hours"] for n in allc)
_HMED = _hrs[len(_hrs) // 2]
print("  split at %.2f h: %d conditions below, %d at or above\n"
      % (_HMED,
         sum(1 for n in allc if COND[n]["hours"] < _HMED),
         sum(1 for n in allc if COND[n]["hours"] >= _HMED)))

def _mixd(name, r, plo, phi_):
    phi = plo if COND[name]["hours"] < _HMED else phi_
    return _mix(name, r, phi)

print("    phi_lo  phi_hi    beta1     95% CI            WCB p")
GRID = [(0.88, 0.88), (0.88, 0.50), (0.88, 0.25), (0.88, 0.00),
        (0.50, 0.88), (0.25, 0.88), (0.00, 0.88)]
for plo, phi_ in GRID:
    b1, lo, hi, d, Xa, Ya, Ca, se = _fit_imputed(
        lambda n, r, a=plo, b=phi_: _mixd(n, r, a, b))
    p = _wcb(Xa, Ya, Ca, b1, se)
    print("     %.2f    %.2f    %+.4f  [%+.4f,%+.4f]   %s%s"
          % (plo, phi_, b1, lo, hi,
             "%.4f" % p if p is not None else "  n/a",
             "" if lo * hi > 0 else "   CI INCLUDES 0"))

# --- 6c-4 · where does the point estimate cross zero? ---------------------
H2("6c-4 - the breaking point")
print("  Holding phi_lo at NB-24's measured 0.88, how large must phi_hi be")
print("  before the point estimate reaches zero?\n")
_cross = None
_prev = None
for i in range(0, 101):
    phi_ = i / 100.0
    b1, lo, hi, d, Xa, Ya, Ca, se = _fit_imputed(
        lambda n, r, b=phi_: _mixd(n, r, 0.88, b))
    if _prev is not None and _prev[1] > 0 >= b1:
        _cross = (phi_, b1)
        break
    _prev = (phi_, b1)
if _cross:
    print("    beta1 reaches zero at phi_hi = %.2f (beta1 = %+.4f), with"
          " phi_lo = 0.88." % _cross)
    print("    So the high-exposure false-negative rate would have to be at")
    print("    least that large, against 0.88 at the low end, for the measured")
    print("    slope to be an artefact of censoring alone.")
else:
    print("    beta1 does not reach zero anywhere on phi_hi in [0, 1] with")
    print("    phi_lo = 0.88. With the low-exposure rate at its measured value,")
    print("    censoring alone cannot flip the sign; the phi_lo = 0.00 row of")
    print("    6c-3 shows that a much lower low-exposure rate could.")

# --- 6c-5 · no rate assumed for the 5-hour runs ----------------------------
H2("6c-5 - only the 1-hour runs held at 0.88; no rate assumed at 5 hours")
print("""  6c-3 and 6c-4 give the 5-hour runs phi_lo because they sit below the
  median exposure, but no 5-hour run was ever re-attacked. Here only the four
  1-hour runs keep 0.88, the rate measured on one of them. beta1 is affine in
  each group's rate, because the imputed log10 value is linear in phi and OLS
  is linear in y, so its extremes over [0, 1] lie at the ends, which both
  grids include.\n""")


def _mix3(name, r, p5, ph):
    h = COND[name]["hours"]
    return _mix(name, r, 0.88 if h < 1.0 else (p5 if h < 10.0 else ph))


_shared = [_fit_imputed(lambda n, r, b=i / 20.0: _mix3(n, r, b, b))[0]
           for i in range(21)]
print("    one rate shared by every other run, 0 to 1:   beta1 %+.4f to %+.4f"
      % (min(_shared), max(_shared)))
_sep = [(_fit_imputed(lambda n, r, a=i / 20.0, b=j / 20.0: _mix3(n, r, a, b))[0],
         i / 20.0, j / 20.0) for i in range(21) for j in range(21)]
_lowest = min(_sep)
print("    5-hour runs and the longer runs each free:   lowest beta1 %+.4f"
      " (5 h at %.2f, the rest at %.2f)" % _lowest)
print("    %s" % ("No such rate makes beta1 negative." if min(_shared) > 0 and _lowest[0] > 0
                  else "SOME RATE MAKES BETA1 NEGATIVE."))

print("""
  What this does and does not settle. It does not rescue the slope: the
  published point estimate is biased upward, because 42% of the observations
  are imputed on an assumption that is wrong most of the time where it has been
  tested. It does bound the damage, and it names the one measurement that would
  close the question -- the false-negative rate at high exposure, on a sample
  large enough to have an interval. NB-24 as written cannot deliver that
  affordably: a non-flip costs about two GPU-hours, so the 47 censored rows of
  rung100-ms3500 are roughly 47 GPU-hours. A capped budget on a sample of 16
  gives the rate to about +/-12 points for about 2 GPU-hours per checkpoint.""")


# ===========================================================================
# 6d · the false-negative rate, now measured at three exposures (NB-24b)
# ===========================================================================
H1("6d - NB-24b: the measured rate falls with exposure, and what that does")

print("""  Section 6c had to sweep the false-negative rate on a grid because only one
  checkpoint had been re-attacked. NB-24b measured two more at a budget of 3
  restarts x 2000 steps, which reproduces every flip NB-24's 5 x 2000 found.""")

# exposure hours -> (flipped, sampled). rung1-final is NB-24's, at 5 x 2000;
# restarts 3 and 4 contributed nothing there, so it is a 3 x 2000 result too.
MEASURED = [(0.98, 22, 25, "rung1-final"),
            (24.60, 13, 16, "rung25-final"),
            (98.02, 9, 16, "rung100-ms3500")]
# deepest restart-0 flip, out of the 2000-step budget: the saturation check.
DEEPEST = {0.98: 950, 24.60: 1350, 98.02: 1800}


def _wilson(k, n, z=1.96):
    p = float(k) / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda k: (math.comb(r1, k) * math.comb(n - r1, c1 - k)) / math.comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(pr(k) for k in range(lo, hi + 1) if pr(k) <= p0 + 1e-12))


print("  %-16s %9s %9s %8s   %-16s %s"
      % ("checkpoint", "exposure", "flipped", "rate", "Wilson 95%", "deepest flip"))
for h, k, n, tag in MEASURED:
    lo, hi = _wilson(k, n)
    print("  %-16s %8.2fh %5d/%-3d %7.1f%%   [%4.1f, %4.1f]      %4d of 2000 (%.0f%%)"
          % (tag, h, k, n, 100.0 * k / n, 100 * lo, 100 * hi,
             DEEPEST[h], 100.0 * DEEPEST[h] / 2000))

H2("6d-1 - is the rate really falling, or is the budget biting unevenly?")
a = MEASURED[0]; c = MEASURED[-1]
print("  Fisher exact on the extremes, %.0f%% against %.0f%%: p = %.4f"
      % (100.0 * a[1] / a[2], 100.0 * c[1] / c[2],
         _fisher(a[1], a[2] - a[1], c[1], c[2] - c[1])))
# Cochran-Armitage, scores = log10(exposure)
xs = [math.log10(h) for h, _, _, _ in MEASURED]
ks = [k for _, k, _, _ in MEASURED]
ns = [n for _, _, n, _ in MEASURED]
N, K = sum(ns), sum(ks)
pbar = float(K) / N
xbar = sum(n * x for n, x in zip(ns, xs)) / N
z = (sum(k * (x - xbar) for k, x in zip(ks, xs))
     / math.sqrt(pbar * (1 - pbar) * sum(n * (x - xbar) ** 2 for n, x in zip(ns, xs))))
_p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
print("  Cochran-Armitage trend over log10 exposure: z = %+.3f, p = %.4f" % (z, _p))

print("""
  Both say the rate falls with exposure, with the utterance as the unit. Each
  exposure is one checkpoint, so neither is a result about exposure, and
  neither settles it, because the 2000-step budget is not equally binding at
  the three checkpoints. The deepest restart-0 flip used 48% of the budget at
  0.98 h, 68% at 24.60 h and 90% at 98.02 h, and only the last had a restart-0
  flip still arriving past step 1500. On restart 0 the low-exposure searches
  had stopped finding successes well before the limit and the high-exposure
  one had not, so 56% is a looser lower bound than 88% is. (Later restarts did
  still succeed late at the lower exposures, at 1850 and 1900 steps.) Part of
  the measured fall may be the budget. We therefore do not report a rate
  function -- we bracket the slope between the measured rates and a uniform
  rate.""")

H2("6d-2 - refitting beta1 between those two readings")

_MED6d = {}
for name in allc:
    ok = sorted(r["eps"] for r in DATA[name] if not r["cens"])
    _MED6d[name] = ok[len(ok) // 2] if ok else CEIL


def _phi_interp(hours, pts):
    """phi at this exposure, linear in log10(hours) through the measured points,
    flat outside their range and clipped to [0, 1]."""
    xs = [math.log10(h) for h, _, _ in pts]
    ys = [p for _, p, _ in pts]
    x = math.log10(hours)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return min(1.0, max(0.0, ys[i] + t * (ys[i + 1] - ys[i])))
    return ys[-1]


def _fit_phi(phi_of_hours, label, show=True):
    Xa, Ya, Ca = [], [], []
    for name in allc:
        m = COND[name]
        phi = phi_of_hours(m["hours"])
        imp = 10.0 ** (phi * math.log10(_MED6d[name]) + (1 - phi) * math.log10(CEIL))
        for r in DATA[name]:
            Xa.append([1.0, math.log10(m["hours"]), m["wer"]])
            Ya.append(math.log10(imp if r["cens"] else r["eps"]))
            Ca.append(m["trun"])
    ba, ra, Xia = ols(Xa, Ya)
    Va, Ga = cr1(Xa, Ya, ba, ra, Xia, Ca)
    s = Va[1][1] ** 0.5
    d = Ga - 1
    lo, hi = ba[1] - tcrit(d) * s, ba[1] + tcrit(d) * s
    # exact Rademacher WCB, null imposed by dropping the exposure column
    cl = sorted(set(Ca))
    p = None
    if len(cl) <= 16:
        Xr = [[row[0], row[2]] for row in Xa]
        br, rr, _ = ols(Xr, Ya)
        fit = [br[0] + br[1] * row[2] for row in Xa]
        t0 = abs(ba[1] / s)
        cnt = tot = 0
        for signs in itertools.product([1.0, -1.0], repeat=len(cl)):
            w = dict(zip(cl, signs))
            ys = [fit[i] + w[Ca[i]] * rr[i] for i in range(len(Xa))]
            b2, r2, Xi2 = ols(Xa, ys)
            V2, _g = cr1(Xa, ys, b2, r2, Xi2, Ca)
            tot += 1
            if abs(b2[1] / V2[1][1] ** 0.5) >= t0 - 1e-12:
                cnt += 1
        p = (cnt + 1.0) / (tot + 1.0)
    if show:
        print("  %-42s %+.4f  [%+.4f,%+.4f]  %5.2fx  %s%s"
              % (label, ba[1], lo, hi, 10.0 ** ba[1],
                 "%.4f" % p if p is not None else " n/a",
                 "" if lo * hi > 0 else "  CI INCLUDES 0"))
    return ba[1], lo, hi, p


PTS_MEAS = [(h, float(k) / n, tag) for h, k, n, tag in MEASURED]
PTS_LO = [(h, _wilson(k, n)[0], tag) for h, k, n, tag in MEASURED]
PTS_HI = [(h, _wilson(k, n)[1], tag) for h, k, n, tag in MEASURED]
PHI_UNIFORM = MEASURED[0][1] / float(MEASURED[0][2])     # the rate measured at 0.98 h, 0.88

print("  %-42s %7s  %-19s %6s  %s"
      % ("reading", "beta1", "95% CI", "x/dec", "WCB p"))
b_pub, *_ = _fit_phi(lambda h: 0.0, "published: censored imputed at the ceiling")
b_uni, _l, _h, _p = _fit_phi(lambda h: PHI_UNIFORM,
                             "uniform at the 0.98 h rate, 0.88")
b_mea, _l2, _h2, _p2 = _fit_phi(lambda h: _phi_interp(h, PTS_MEAS),
                                "the measured rates, interpolated")
_fit_phi(lambda h: _phi_interp(h, PTS_LO), "  measured, each at its Wilson lower")
_fit_phi(lambda h: _phi_interp(h, PTS_HI), "  measured, each at its Wilson upper")

print("""
  The published fit is the one the paper reports. If the false-negative rate
  falls with exposure as measured, correcting the censoring moves beta1 UP:
  the ceiling would be propping up the low-exposure end more than the
  high-exposure end, flattening the slope. A uniform rate moves it slightly
  down. Every reading above is positive and none has an interval covering
  zero.""")
print("  bracket on beta1 from this evidence: %+.4f (uniform) to %+.4f (as measured),"
      % (min(b_uni, b_mea), max(b_uni, b_mea)))
print("  against %+.4f published. Neither end changes its sign; 6c-3 shows the"
      % b_pub)
print("  readings outside the bracket that could.")
print("""
  Reported position: the published specification stays primary, because the
  wording rule of section 3 was fixed on it. This is a sensitivity analysis,
  and its upper reading favours the paper -- a reason to be careful with it,
  not a reason to promote it.""")


H1("7 - are the perturbations actually inaudible, and how fine is the bisection?")
print("""  The introduction claims the perturbations sound normal. The attack CSVs
  carry a per-utterance SNR, so the claim can be checked rather than asserted.
  Successful attacks only -- a censored row never reached its target.\n""")
print("  %-14s %6s %7s %7s %7s %7s" % ("condition", "n", "min", "median", "mean", "max"))
allsnr = []
for name in allc:
    v = sorted(r["snr"] for r in DATA[name] if not r["cens"] and r["snr"] is not None)
    if not v:
        continue
    allsnr += v
    print("  %-14s %6d %7.1f %7.1f %7.1f %7.1f"
          % (name, len(v), v[0], v[len(v) // 2], sum(v) / len(v), v[-1]))
if allsnr:
    allsnr.sort()
    q = lambda p: allsnr[min(len(allsnr) - 1, int(p * len(allsnr)))]
    print("\n  pooled (n=%d): median %.1f dB, 5th pct %.1f dB, min %.1f dB"
          % (len(allsnr), q(0.5), q(0.05), allsnr[0]))
    print("  %d of %d successful attacks sit below 20 dB SNR (%.1f%%)."
          % (sum(1 for x in allsnr if x < 20), len(allsnr),
             100.0 * sum(1 for x in allsnr if x < 20) / len(allsnr)))
    print("  Report the distribution, not a claim. No listening test was run, so")
    print("  'inaudible' is not supportable; 'small relative to the signal' is.")
bws = sorted(r["bw"] for n in allc for r in DATA[n] if r["bw"] is not None)
if bws:
    print("\n  bisection bracket width (ratio hi/lo at termination):")
    print("    min %.4f  median %.4f  max %.4f" % (bws[0], bws[len(bws) // 2], bws[-1]))
    print("    A median bracket of %.3f means eps50 is resolved to about %.1f%%." % (bws[len(bws) // 2], 100 * (bws[len(bws) // 2] - 1)))
    print("    The pooled effect is a factor of %.2f per decade, so the resolution" % (10 ** beta[1]))
    print("    is %s relative to the effect being measured."
          % ("comfortably finer" if (bws[len(bws) // 2] - 1) < 0.5 * (10 ** beta[1] - 1) else "NOT comfortably finer"))


# ===========================================================================
# 8 · the unmatched ladder, with replicate means
# ===========================================================================
H1("8 - the unmatched ladder: single runs against replicate means")
print("""  The paper argues the unmatched comparison 'points the wrong way' from
  75% at 1 h against 79% at 5 h. That is a 4-point gap, far inside sigma. With
  every run of each rung averaged, the same argument is made by a much wider
  margin.\n""")
print("  %-6s %-30s %8s %8s" % ("rung", "runs", "single", "mean"))
for rung in (1, 5, 25):
    names = [n for n in allc if COND[n]["rung"] == rung and COND[n]["grp"]]
    a = [COND[n]["asr"] for n in names]
    single = next(COND[n]["asr"] for n in names if COND[n]["trun"].endswith("-r1"))
    print("  %-6d %-30s %7.1f%% %7.1f%%" % (rung, ", ".join(sorted(names)), single, sum(a) / len(a)))
m1 = sum(COND[n]["asr"] for n in allc if COND[n]["rung"] == 1 and COND[n]["grp"]) / 4
m5 = sum(COND[n]["asr"] for n in allc if COND[n]["rung"] == 5 and COND[n]["grp"]) / 3
print("\n  1 h -> 5 h as published:        %+.1f pp  (75.0 -> 79.0)" % (79.0 - 75.0))
print("  1 h -> 5 h on replicate means:  %+.1f pp  (%.1f -> %.1f)" % (m5 - m1, m1, m5))
print("  The conclusion is unchanged and the evidence is %.1fx stronger."
      % ((m5 - m1) / (79.0 - 75.0)))

H1("done")
print("""  Everything above is CPU-only and reproduces from the committed CSVs.
  What it cannot do, and what therefore belongs in Limitations:

    * separate training from attack variance with any precision. One re-attacked
      checkpoint gives df 1. Three checkpoints at three PGD seeds would give a
      usable interval, at roughly 10 GPU-hours.
    * break the collinearity between exposure and training step at the top of
      the range. A second 100-hour run (now included, if INCLUDE_100H_REPLICATE
      is on) adds an independent cluster but does NOT break it: both runs were
      checkpointed at the same steps. That needs checkpoints matched on
      optimisation but not on audio.
    * test attack strength properly: step size scaled with epsilon, restarts on
      failures, longer budgets. That would change every eps50 value, so it is a
      full re-measurement rather than an addition.
    * a listening test for perceptibility. SNR is reported instead.""")
