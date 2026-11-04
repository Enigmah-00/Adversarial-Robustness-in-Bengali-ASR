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
import csv, glob, io, json, math, os, random, sys

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
  75%% at 1 h against 79%% at 5 h. That is a 4-point gap, far inside sigma. With
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
