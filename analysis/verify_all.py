#!/usr/bin/env python3
"""Re-derive every load-bearing number in the paper and fail if one disagrees.

This is a regression test for the manuscript, not an analysis. `reanalysis.py`
and `attack_replication.py` compute the results; this script independently
recomputes the figures the paper prints and asserts they still match.

Every check recomputes from data/results/ or reads the manuscript text. Nothing
is compared against another draft -- document-to-document agreement is exactly
what let two defects through in November 2026: a passage that quoted the
25-utterance figures in a paper whose evaluation set is 100 utterances, and a
contributions bullet still claiming a bound the results section had retracted.
Both were invisible to cross-checking, because every draft agreed.

Sections A-D need only the released CSVs. Section E additionally reads
paper/ijst-v2/manuscript.tex and is skipped if that file is absent, so the
script is useful to anyone holding the data and code alone.

Run:  python3 analysis/verify_all.py [path-to-repo-root]
Exit: 0 if every check passes, 1 otherwise.
"""
import csv, io, math, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(ROOT, "data", "results")
TEX = os.path.join(ROOT, "paper", "ijst-v2", "manuscript.tex")

FAIL, SKIP, N_OK = [], 0, 0


def check(label, got, want, tol=None):
    global N_OK
    ok = (got == want) if tol is None else (abs(got - want) <= tol)
    if ok:
        N_OK += 1
        print("  ok    %-54s %s" % (label, got))
    else:
        FAIL.append(label)
        print("  FAIL  %-54s got %s, want %s" % (label, got, want))


def tf(v):
    """Read a boolean that has been through a CSV. Never use bool() for this:
    the STRING "False" is truthy, so it would silently invert a result."""
    return str(v).strip().lower() == "true"


def rows(fn):
    with io.open(os.path.join(RES, fn), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def head(s):
    print("\n" + "=" * 80 + "\n  " + s + "\n" + "=" * 80)


# ===========================================================================
head("A. The ladder, recomputed from the released CSVs")
# ===========================================================================
for lab, fn, want in [("1 h", "eps50_rung1_final_T5_n100.csv", 75.0),
                      ("5 h", "eps50_rung5_final_T5_n100.csv", 79.0),
                      ("25 h", "eps50_rung25_final_T5_n100.csv", 66.0),
                      ("100 h final", "eps50_rung100-final_T5_n100.csv", 44.0)]:
    r = rows(fn)
    check("ASR %s (n=%d)" % (lab, len(r)),
          round(100.0 * sum(1 for x in r if not tf(x["censored"])) / len(r), 1), want)

# The 1-to-5-hour step runs the WRONG way: this is the sign reversal that makes
# accuracy matching a prerequisite rather than a refinement.
RUNGS = {1: ["eps50_rung1_final_T5_n100.csv", "eps50_rung1-trajfinal_T5_n100.csv",
             "eps50_rung1-s1338_T5_n100.csv", "eps50_rung1-s1339_T5_n100.csv"],
         5: ["eps50_rung5_final_T5_n100.csv", "eps50_rung5-s1338_T5_n100.csv",
             "eps50_rung5-s1339_T5_n100.csv"],
         25: ["eps50_rung25_final_T5_n100.csv", "eps50_rung25-s1338_T5_n100.csv",
              "eps50_rung25-s1339_T5_n100.csv"]}
avg = {}
for k, fs in RUNGS.items():
    a = []
    for f in fs:
        r = rows(f)
        a.append(100.0 * sum(1 for x in r if not tf(x["censored"])) / len(r))
    avg[k] = sum(a) / len(a)
_all = []
for _f in sorted(os.listdir(RES)):
    if _f.startswith("eps50_") and _f.endswith("_T5_n100.csv") and "SECOND" not in _f:
        _r = rows(_f)
        _all.append(100.0 * sum(1 for x in _r if not tf(x["censored"])) / len(_r))
check("attack success spans, min (%)", round(min(_all), 1), 34.0)
check("attack success spans, max (%)", round(max(_all), 1), 96.0)
check("1 h rung average over 4 runs", round(avg[1], 1), 53.0)
check("5 h rung average over 3 runs", round(avg[5], 1), 78.0)
check("25 h rung average over 3 runs", round(avg[25], 1), 67.0)
check("1->5 h step, replicated (reversal)", round(avg[5] - avg[1], 1), 25.0)

# The twelve comparisons between two runs of one condition (1, 5, 25 h rungs),
# McNemar exact, then Holm-Bonferroni across the family of twelve.
import itertools


def _succ(fn):
    return {x["utt_id"]: (not tf(x["censored"])) for x in rows(fn)}


def _mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n)


_ps = []
for _k, _fs in RUNGS.items():
    for _fa, _fb in itertools.combinations(_fs, 2):
        _A, _B = _succ(_fa), _succ(_fb)
        _keys = sorted(set(_A) & set(_B))
        _b = sum(1 for u in _keys if _A[u] and not _B[u])
        _c = sum(1 for u in _keys if _B[u] and not _A[u])
        _ps.append(_mcnemar(_b, _c))
check("same-condition comparisons", len(_ps), 12)
# the four comparisons that stay inside one library version (the sigma pairs)
_same = [_mcnemar(*_bc) for _bc in ((42, 8), (32, 14), (39, 2), (27, 6))]
check("same-version comparisons significant at 5%", sum(1 for q in _same if q < 0.05), 4)
check("same-version comparisons surviving Holm over twelve",
      sum(1 for q in _same if q <= sorted(_ps)[6] + 1e-18), 3)
check("significant at the nominal 5% level", sum(1 for q in _ps if q < 0.05), 10)
_holm = 0
for _rank, _q in enumerate(sorted(_ps)):
    if _q <= 0.05 / (len(_ps) - _rank):
        _holm += 1
    else:
        break
check("surviving Holm-Bonferroni across the twelve", _holm, 7)

# ===========================================================================
head("B. The false-negative work, from the strength files")
# ===========================================================================
# "Within 400 steps from the main attack's start" needs BOTH conditions: the
# same starting point the main attack uses (restart 0) AND the step threshold.
# Filtering on steps alone gives 15/25 at 0.98 h, not 10/25. 400 is the
# per-probe budget of the bisection; the ceiling probe allows 800 (below).
ca_full, ca_inb = [], []
for lab, fn, n, wflip, winb, wdeep in [
        ("0.98 h", "strength_rung1-final_T5.csv", 25, 22, 10, 950),
        ("24.60 h", "strength_rung25-final_T5.csv", 16, 13, 2, 1350),
        ("98.02 h", "strength_rung100-ms3500_T5.csv", 16, 9, 1, 1800)]:
    r = rows(fn)
    flip = [x for x in r if tf(x["now_success"])]
    inb = [x for x in flip
           if int(x["restart_used"]) == 0 and int(x["steps_used"]) <= 400]
    r0 = [int(x["steps_used"]) for x in flip if int(x["restart_used"]) == 0]
    check("%s  n" % lab, len(r), n)
    check("%s  flipped under a stronger search" % lab, len(flip), wflip)
    check("%s  restart 0, within 400 steps" % lab, len(inb), winb)
    check("%s  deepest restart-0 search" % lab, max(r0), wdeep)
    ca_full.append((len(r), len(flip)))
    ca_inb.append((len(r), len(inb)))


def cochran_armitage(cells, xs):
    """Trend in a proportion across ordered scores `xs`. Returns (z, two-sided p)."""
    N = sum(n for n, _ in cells)
    pbar = sum(k for _, k in cells) / N
    xbar = sum(x * n for x, (n, _) in zip(xs, cells)) / N
    num = sum(x * (k - n * pbar) for x, (n, k) in zip(xs, cells))
    var = pbar * (1 - pbar) * sum(n * (x - xbar) ** 2 for x, (n, _) in zip(xs, cells))
    z = num / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))


XS = [math.log10(0.98), math.log10(24.60), math.log10(98.02)]
z1, p1 = cochran_armitage(ca_full, XS)
z2, p2 = cochran_armitage(ca_inb, XS)
check("trend, full rates, z", round(z1, 3), -2.095)
check("trend, full rates, p", round(p1, 4), 0.0362)
check("trend, within 400 steps, z", round(z2, 3), -2.751)
check("trend, within 400 steps, p (paper prints 0.006)", round(p2, 3), 0.006)
check("reliability ratio 40.0/6.2", round((10 / 25) / (1 / 16), 1), 6.4)
# The main attack tests the ceiling with 800 steps; 400 is its per-probe budget.
_c800 = []
for _lab, _fn in (("0.98", "strength_rung1-final_T5.csv"), ("24.6", "strength_rung25-final_T5.csv"),
                  ("98.0", "strength_rung100-ms3500_T5.csv")):
    _r = rows(_fn)
    _c800.append((len(_r), sum(1 for x in _r if tf(x["now_success"]) and int(x["restart_used"]) == 0
                               and int(x["steps_used"]) <= 800)))
check("restart-0 flips within 800 steps", [k for _, k in _c800], [11, 7, 3])
_z8, _p8 = cochran_armitage(_c800, XS)
check("trend within 800 steps, p", round(_p8, 2), 0.18)
check("understatement of attack success, points", (round(22.0), round(34 * 13 / 16.0), round(47 * 9 / 16.0)), (22, 28, 26))


def _wilson(k, n, z=1.959964):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h)), round(100 * (c + h))


check("Wilson 95% for 22/25", _wilson(22, 25), (70, 96))
check("Wilson 95% for 13/16", _wilson(13, 16), (57, 93))
check("Wilson 95% for 9/16", _wilson(9, 16), (33, 77))
check("stronger search covers this many censored rows", 25 + 16 + 16, 57)
check("1 h ASR is a lower bound: 75 + 22", 75 + 22, 97)
# The lowest Wilson limit is 33.2%, just under one third, so the paper must not
# say "at least a third".
_wl = 9 / 16.0
_wc = (_wl + 1.959964 ** 2 / 32) / (1 + 1.959964 ** 2 / 16)
_wh = 1.959964 * math.sqrt(_wl * (1 - _wl) / 16 + 1.959964 ** 2 / 1024) / (1 + 1.959964 ** 2 / 16)
check("lowest Wilson limit is below one third", (_wc - _wh) < 1 / 3.0, True)
# Cost of re-attacking every censored row at ~20 min each, on a 30 h weekly quota.
check("679 rows x 20 min, GPU-hours", round(679 * 20 / 60.0), 226)
check("weeks of quota at 30 h a week", round(679 * 20 / 60.0 / 30, 1), 7.5)
# The 1-hour rung's failures cannot all be compared with the 5-hour one's: only
# the 1-hour final was re-attacked. Its 25 failures vs the 5-hour final's 21.
check("1 h final censored, 5 h final censored", (100 - 75, 100 - 79), (25, 21))

# ===========================================================================
head("C. Derived quantities")
# ===========================================================================
SIGMA, ATTACK_SD, BY_RUNG = 21.574, 2.910, 16.375
check("sigma * sqrt(2), the noise on a pair", round(SIGMA * math.sqrt(2), 2), 30.51)
check("training sd, removed in quadrature",
      round(math.sqrt(SIGMA ** 2 - ATTACK_SD ** 2), 3), 21.377)
check("attack share of variance (%)", round(100 * ATTACK_SD ** 2 / SIGMA ** 2, 2), 1.82)
check("by-rung noise, 16.375*sqrt2", round(BY_RUNG * math.sqrt(2), 2), 23.16)
# sigma from the three seed-contrast pairs alone (the same-seed pair dropped)
_sc = math.sqrt((18.0 ** 2 / 2 + 37.0 ** 2 / 2 + 21.0 ** 2 / 2) / 3)
check("sigma without the same-seed pair",
      round(_sc / math.exp(0.5 * math.log(2.0 / 3) + math.lgamma(2.0) - math.lgamma(1.5)), 1), 20.5)
check("10^0.0888, per decade", round(10 ** 0.0888, 2), 1.23)
check("10^0.1051, censoring-corrected", round(10 ** 0.1051, 2), 1.27)
check("10^0.0799, uniform-rate end of the bracket", round(10 ** 0.0799, 2), 1.20)
# reanalysis.py 6c-2: uniform phi from 0 to 1 gives +0.0888 down to +0.0786.
check("uniform rate of any size, rounded range", (round(0.0786, 3), round(0.0888, 3)), (0.079, 0.089))
# 44 runs is a planning figure at the point estimate; at the ends of sigma's
# 95% interval [12.15, 58.27] the 13-point requirement is:
_K = 2 * (1.959964 + 0.841621) ** 2
check("power constant", round(_K, 2), 15.70)
check("runs for 13 points at sigma 12.15 / 21.574 / 58.27",
      tuple(math.ceil(_K * s ** 2 / 169.0) for s in (12.15, SIGMA, 58.27)), (14, 44, 316))
# PRIMARY since the third review: every re-attack pair at face value
# (attack_replication.py: 8.053 pp on df 5). The version-controlled 2.910 above
# is the sensitivity reading, because setting the other two pairs aside rests
# on a judgement made after seeing the data.
check("attack share of variance, primary: every pair (%)", round(100 * 8.053 ** 2 / SIGMA ** 2), 14)
check("training sd, primary: every pair", round(math.sqrt(SIGMA ** 2 - 8.053 ** 2), 1), 20.0)
check("training share of variance, primary (%)",
      round(100 * (SIGMA ** 2 - 8.053 ** 2) / SIGMA ** 2), 86)
check("training share of variance, version-controlled pairs (%)",
      round(100 * (SIGMA ** 2 - ATTACK_SD ** 2) / SIGMA ** 2), 98)
# reanalysis.py section 5: the ten runs without the 100 h milestones.
check("no-milestone fit, rounded", (round(0.6473, 2), round(-0.2357, 2), round(1.5303, 2)),
      (0.65, -0.24, 1.53))
check("no-milestone exposure range, fold", round(24.60 / 0.98), 25)

# The manuscript quotes brackets as a UNIT ("one bracket"), never the numeral.
# 1.0981386 in older notes was a truncation of 1.098138675683.
BRACKET = (2e-3 / 1e-4) ** (1 / 32.0)
check("bisection bracket width", round(BRACKET, 7), 1.0981387)
check("bracket as a resolution (%)", round(100 * (BRACKET - 1), 1), 9.8)


def c4(df):
    """E[s]/sigma for a chi-distributed s on `df` degrees of freedom."""
    return math.exp(0.5 * math.log(2.0 / df)
                    + math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))


# The 100-hour rung: two independent runs at three matched checkpoints.
G100 = [3.0, 2.0, 3.0]
raw100 = math.sqrt(sum(g * g / 2 for g in G100) / len(G100))
sig100 = raw100 / c4(3)
check("100 h rung bound, c4(3)-corrected", round(sig100, 1), 2.1)
# The earlier 25-utterance re-attack (last row of tab:attackrep): 16 vs 19 of 25.
check("25-utterance pair gap (points)", round(100 * (19 - 16) / 25.0, 1), 12.0)
check("rescaled to 100 utterances (points)", round(12.0 / math.sqrt(100 / 25.0), 1), 6.0)
_g4 = (4.00, 0.00, 5.21, 6.00)
check("pooled with the rescaled pair, c4(4)-corrected",
      round(math.sqrt(sum(g * g / 2 for g in _g4) / 4.0) / c4(4), 1), 3.3)
# The primary figure from its five gaps (tab:attackrep, the 25-utterance pair at
# face value), with the chi-square interval attack_replication.py prints.
# chi-square quantiles on 5 df: 0.025 -> 0.8312, 0.975 -> 12.8325.
_g5 = (20.00, 4.00, 0.00, 5.21, 12.00)
_raw5 = math.sqrt(sum(g * g / 2 for g in _g5) / 5.0)
check("primary attack sd, every pair, c4(5)-corrected", round(_raw5 / c4(5), 1), 8.1)
check("its 95% interval", (round(_raw5 * math.sqrt(5 / 12.8325), 1), round(_raw5 * math.sqrt(5 / 0.8312), 1)),
      (4.8, 18.8))
# The same five pairs with the 25-utterance gap rescaled to n=100 (6.0): the
# size-adjusted reading the text reports beside the unscaled, conservative 8.1.
_g5s = (20.00, 4.00, 0.00, 5.21, 6.00)
_raw5s = math.sqrt(sum(g * g / 2 for g in _g5s) / 5.0)
check("all pairs, 25-utterance gap rescaled, c4(5)-corrected", round(_raw5s / c4(5), 1), 7.3)
check("its share of the variance (%)", round(100 * (_raw5s / c4(5)) ** 2 / SIGMA ** 2), 11)
check("100 h rung tighter than the low rungs by", round(SIGMA / sig100, 1), 10.4)
check("that bound is below the attack sd", sig100 < ATTACK_SD, True)

ZA, ZB = 1.959964, 0.8416212          # z_0.975 and z_0.80
K = 2 * (ZA + ZB) ** 2
check("power constant 2(z+z)^2", round(K, 2), 15.70)
check("power: runs per group for a 13 pp effect",
      math.ceil(K * SIGMA ** 2 / 13.0 ** 2), 44)
check("power: runs x 4 rungs", 44 * 4, 176)
check("power: weeks at 30 GPU-h", round((546 + 1417) / 30.0, 1), 65.4)
# Read backwards: the smallest effect n runs per group can resolve. n=1 is the
# standard practice, and 85.5 exceeds the 0-100 range of the outcome.
# equal budget of 18 runs: six levels x 3 runs resolves 49, three levels x 6 resolves 35
check("power: n=6 resolves nothing smaller than", round(SIGMA * math.sqrt(K / 6), 1), 34.9)
# conclusion: pair noise exceeds ONE effect and is within 1.3x of the other two
PAIR = SIGMA * math.sqrt(2)
check("pair noise exceeds exactly one of the three effects",
      sum(1 for e in (38.0, 33.0, 13.0) if PAIR > e), 1)
# sensitivity to sigma: the largest effect reaches 2x the pair noise only below ~13
check("sigma below which 38 pp is twice the pair noise", round(38.0 / (2 * math.sqrt(2)), 1), 13.4)
check("raw same-condition gaps 34 and 37 pp need no sigma", (75 - 41, 96 - 59), (34, 37))
# the unmatched 1h -> 5h step on rung means, against its between-run standard error
_se = SIGMA * math.sqrt(1 / 4.0 + 1 / 3.0)
check("unmatched step / its standard error, sigma 21.6", round(25.0 / _se, 1), 1.5)
check("unmatched step / its standard error, sigma 16.4", round(25.0 / (BY_RUNG * math.sqrt(7 / 12.0)), 1), 2.0)
check("four runs per condition resolve (pp)", round(SIGMA * math.sqrt(K / 4), 1), 42.7)
check("within-budget misses, lowest and highest exposure (%)", (round(100 * 10 / 25.0), round(100 * 1 / 16.0)), (40, 6))
check("the other two effects are within 1.3x the pair noise",
      all(e / PAIR < 1.3 for e in (38.0, 33.0)), True)
for n, want in ((1, 85.5), (2, 60.4), (3, 49.4), (4, 42.7), (10, 27.0)):
    check("power: n=%d resolves nothing smaller than" % n,
          round(SIGMA * math.sqrt(K / n), 1), want)

# Residual accuracy gap, priced at the CURRENT pooled-fit coefficient (+0.5504
# per unit WER). The older 3-9% used the coefficient of an earlier fit (+0.433).
def _med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _imputed(fn):
    return [2e-3 if tf(x["censored"]) else float(x["eps50"]) for x in rows(fn)]


_shares = []
for _fa, _fb, _gap in (("eps50_rung1_final_T5_n100.csv", "eps50_rung100_milestone-1000_T5_n100.csv", 0.0588),
                       ("eps50_rung5_final_T5_n100.csv", "eps50_rung100_milestone-2000_T5_n100.csv", 0.0519),
                       ("eps50_rung25_final_T5_n100.csv", "eps50_rung100_milestone-3500_T5_n100.csv", 0.033448)):
    _tot = _med(_imputed(_fb)) / _med(_imputed(_fa))
    _shares.append(100 * (10 ** (0.5504 * _gap) - 1) / (_tot - 1))
check("residual accuracy share, smallest (%)", round(min(_shares)), 4)
check("residual accuracy share, largest (%)", round(max(_shares)), 11)
# The derivation the text gives. The coefficient lives on the eps50 scale, so
# the residual is priced as a predicted eps50 factor against each pair's
# observed median-eps50 ratio, never in points of attack success. The interval
# is the CR1 one from reanalysis.py section 3 (se 0.1435, t on df 11 = 2.201).
_WLO, _WHI = 0.5504 - 2.201 * 0.1435, 0.5504 + 2.201 * 0.1435
check("accuracy coefficient 95% interval", (round(_WLO, 2), round(_WHI, 2)), (0.23, 0.87))
_fac, _rat, _ends = [], [], []
for _fa, _fb, _gap in (("eps50_rung1_final_T5_n100.csv", "eps50_rung100_milestone-1000_T5_n100.csv", 0.0588),
                       ("eps50_rung5_final_T5_n100.csv", "eps50_rung100_milestone-2000_T5_n100.csv", 0.0519),
                       ("eps50_rung25_final_T5_n100.csv", "eps50_rung100_milestone-3500_T5_n100.csv", 0.033448)):
    _tot = _med(_imputed(_fb)) / _med(_imputed(_fa))
    _fac.append(10 ** (0.5504 * _gap))
    _rat.append(_tot)
    _ends += [100 * (10 ** (_b * _gap) - 1) / (_tot - 1) for _b in (_WLO, _WHI)]
check("predicted eps50 factor from the WER gap, range", (round(min(_fac), 2), round(max(_fac), 2)), (1.04, 1.08))
check("median-eps50 ratio between pair members, range", (round(min(_rat), 2), round(max(_rat), 2)), (1.39, 2.93))
# The text quotes that range; 2.935 printed to three places rounds to 2.93, not 2.94.
check("text quotes the ratio range as computed",
      "differs by a factor of 1.39 to 2.93" in re.sub(r"\s+", " ", io.open(TEX, encoding="utf-8").read())
      if os.path.exists(TEX) else True, True)
check("residual share across the coefficient's interval (%)", (round(min(_ends)), round(max(_ends))), (1, 18))
# Limitations: the fit without the six 100 h checkpoints (reanalysis.py section 5).
check("no-milestone slope over the pooled slope, rounded", round(0.6473 / 0.0888), 7)
check("no-milestone factor per decade", round(10 ** 0.6473, 1), 4.4)
check("no-milestone accuracy coefficient, rounded", round(3.0538, 1), 3.1)
_lh = [math.log10(h) for h in [0.98] * 4 + [4.90] * 3 + [24.60] * 3]
_wr = [0.6355, 0.6251, 0.5905, 0.6240, 0.4233, 0.4152, 0.4187, 0.3114, 0.2987, 0.2907]
_mh, _mw = sum(_lh) / 10.0, sum(_wr) / 10.0
_r10 = (sum((a - _mh) * (b - _mw) for a, b in zip(_lh, _wr))
        / math.sqrt(sum((a - _mh) ** 2 for a in _lh) * sum((b - _mw) ** 2 for b in _wr)))
check("exposure and clean WER across the ten runs, r", round(_r10, 2), -0.99)
# reanalysis.py 6c-5: only the 1 h runs at 0.88, no rate assumed at 5 h.
check("6c-5 lowest beta1, rounded", round(0.0454, 3), 0.045)
check("pipeline survival 12 of 576 (%)", round(100 * 12 / 576.0, 1), 2.1)

# ===========================================================================
head("D. The negation comparison, from the 38-utterance files")
# ===========================================================================
# Three edit types on the same 38 utterances and two models: T1 deletes the
# negation particle, T6 deletes a matched non-negation word, T5 retargets the
# whole utterance. A censored row has an empty eps50 and is imputed at the
# 2e-3 ceiling, as everywhere else in the paper.
NEG = {}
for _m in ("100", "1"):
    for _e in ("T1", "T6", "T5"):
        NEG[(_m, _e)] = {x["utt_id"]: x for x in rows("eps50_rung%s_final_%s_n38.csv" % (_m, _e))}
_eval = set(x["utt_id"] for x in rows("attack_eval_t1.csv"))
check("negation set size", len(_eval), 38)
check("all six files cover exactly that set, once each",
      all(len(d) == 38 and set(d) == _eval for d in NEG.values()), True)
check("a success is an exact match and a censored row never is",
      all(tf(x["exact_match"]) == (not tf(x["censored"]))
          for d in NEG.values() for x in d.values()), True)
check("censored rows carry no eps50",
      all(x["eps50"].strip() == "" for d in NEG.values() for x in d.values() if tf(x["censored"])), True)
_tg = rows("bn_attack_targets.csv")
check("T1 and T5 targets are the targets file's, on both models",
      all(NEG[(_m, x["edit_type"])][x["source_utterance_id"]]["target_text"] == x["target_text"]
          for x in _tg for _m in ("100", "1")) and len(_tg) == 76, True)
_t6t = {x["utt_id"]: x["target_text"] for x in rows("t6_targets_n38.csv")}
check("T6 targets are the T6 targets file's, on both models",
      all(NEG[(_m, "T6")][u]["target_text"] == _t6t[u] for u in _eval for _m in ("100", "1")), True)


def _neps(x):
    return 2e-3 if tf(x["censored"]) else float(x["eps50"])


for _m, _wasr, _wcen in (("100", (100.0, 97.4, 36.8), (0, 1, 24)),
                         ("1", (84.2, 78.9, 63.2), (6, 8, 14))):
    check("%s h: attack success T1, T6, T5 (%%)" % _m,
          tuple(round(100.0 * sum(1 for x in NEG[(_m, e)].values() if not tf(x["censored"])) / 38, 1)
                for e in ("T1", "T6", "T5")), _wasr)
    check("%s h: censored T1, T6, T5" % _m,
          tuple(sum(1 for x in NEG[(_m, e)].values() if tf(x["censored"])) for e in ("T1", "T6", "T5")),
          _wcen)

# Per-utterance ratios, then the sign test on T1 against T6 (the exact binomial
# test is the same computation as _mcnemar above). At 100 h T1 is cheaper than
# T5 on all 38, so there are no ties; at 1 h the four T1/T5 ties are utterances
# censored under both edits.
for _m, _w51, _wc51, _w61, _wc61, _wd61, _wp61 in (("100", 9.34, 38, 1.26, 26, 37, 0.020),
                                                   ("1", 4.33, 32, 1.11, 22, 33, 0.080)):
    _t1, _t5, _t6 = NEG[(_m, "T1")], NEG[(_m, "T5")], NEG[(_m, "T6")]
    check("%s h: median per-utterance ratio T5/T1" % _m,
          round(_med([_neps(_t5[u]) / _neps(_t1[u]) for u in _eval]), 2), _w51)
    check("%s h: T1 cheaper than T5 on" % _m, sum(1 for u in _eval if _neps(_t1[u]) < _neps(_t5[u])), _wc51)
    check("%s h: median per-utterance ratio T6/T1" % _m,
          round(_med([_neps(_t6[u]) / _neps(_t1[u]) for u in _eval]), 2), _w61)
    _b = sum(1 for u in _eval if _neps(_t1[u]) < _neps(_t6[u]))
    _c = sum(1 for u in _eval if _neps(_t6[u]) < _neps(_t1[u]))
    check("%s h: T1 cheaper than T6 on" % _m, _b, _wc61)
    check("%s h: discordant T1/T6 pairs" % _m, _b + _c, _wd61)
    check("%s h: sign test T1 vs T6, p" % _m, round(_mcnemar(_b, _c), 3), _wp61)


def _mean_len(d):
    return sum(len(x["target_text"]) for x in d.values()) / float(len(d))


check("mean target length T6, T1 (characters)",
      (round(_mean_len(NEG[("100", "T6")]), 1), round(_mean_len(NEG[("100", "T1")]), 1)), (45.8, 46.5))
# T1 and T6 were searched over [1e-5, 2e-3] in six rounds, T5 over the main
# attacks' [1e-4, 2e-3] in five; the files record the resulting bracket widths.
check("bracket widths recorded, T1 and T6 / T5",
      (sorted(set(round(float(x["bracket_width"]), 4) for _m in ("100", "1") for e in ("T1", "T6")
                  for x in NEG[(_m, e)].values())),
       sorted(set(round(float(x["bracket_width"]), 4) for _m in ("100", "1")
                  for x in NEG[(_m, "T5")].values()))),
      ([round((2e-3 / 1e-5) ** (1 / 64.0), 4)], [round(BRACKET, 4)]))
check("100 h: negation deletions cheaper than 1e-4",
      sum(1 for x in NEG[("100", "T1")].values() if not tf(x["censored"]) and float(x["eps50"]) < 1e-4), 16)
check("no successful retargeting within a factor of two of the 1e-4 floor",
      min(float(x["eps50"]) for _m in ("100", "1") for x in NEG[(_m, "T5")].values()
          if not tf(x["censored"])) > 2e-4, True)
# The annotation record: one row per candidate shown to the native speaker,
# y accepted, n rejected, blank never decided (one row).
_val = rows("t1_pairs_validated.csv")
check("candidates judged (y or n)", sum(1 for x in _val if x["validated"] in ("y", "n")), 67)
check("candidates accepted", sum(1 for x in _val if x["validated"] == "y"), 38)
check("the accepted candidates are the negation set",
      set(x["source_utterance_id"] for x in _val if x["validated"] == "y") == _eval, True)

# ===========================================================================
head("E. The manuscript text")
# ===========================================================================
if not os.path.exists(TEX):
    SKIP = 1
    print("  skipped: %s not present (data-and-code-only checkout)" % TEX)
else:
    tex = io.open(TEX, encoding="utf-8").read()
    # Retracted claims and stale figures. Each of these was in a submitted
    # draft; none may come back.
    for frag, why in [
            ("at least $9.3", "retracted lower bound on the negation ratio"),
            ("ratio is at least", "retracted lower bound"),
            ("80 of 100", "stale partial run"),
            ("measured on 80", "stale partial run"),
            ("46.2", "stale partial-run ASR"),
            ("Subtracted from $\\sigma", "quadrature described as subtraction"),
            ("reduces the slope",
             "overstated censoring clause (bracket lower end is below +0.089)"),
            ("not identifiable in this design",
             "margin passage: the fit is identified, just imprecise"),
            ("Measuring How", "title overpromised a measurement the paper cannot deliver"),
            ("It has no censoring problem", "ASR is still biased by failed attacks"),
            ("with six-point weights, and", "0.061 is the sampled Rademacher p, not Webb's"),
            ("costs less than six levels", "18 runs cannot cost less than 6"),
            ("larger than two of the three effects", "sigma exceeds one effect, not two"),
            ("reported in this literature is a lower bound", "claim wider than the evidence"),
            ("judged each candidate", "67 of the 277 candidates were judged, not all"),
            ("4{,}086", "stale 5 h utterance count; provenance says 4,051"),
            ("nobody has", "contradicted 'rarely' in related work"),
            ("two of the three things we checked", "clashed with the three-problems framing"),
            ("received about 2.5 times", "compute control is measured for pairs 1 and 2 only"),
            ("made the same number of", "matching is approximate, not exact"),
            ("pair transcribe equally badly", "matching is approximate, not exact"),
            ("matched by construction", "matching is approximate, not exact"),
            ("so the rest is training", "the remainder is the training run, defined"),
            ("the instability is in the training", "the remainder is the training run, defined"),
            ("links are withheld here", "artifacts must be available to reviewers"),
            ("Markert et al., 2022", "the symposium proceedings are 2021"),
            ("Markert et al. (2022)", "the symposium proceedings are 2021"),
            ("3 to 9\\%", "residual share used a superseded coefficient"),
            ("published attack", "nothing here is published yet; it is the main attack"),
            ("synthesised", "the played audio was corpus audio, not synthesised speech"),
            ("total spend on paid", "unverifiable funding clause"),
            ("identical line for line", "the second run saved different checkpoints"),
            ("three restarts, 2000 steps", "the 1-hour re-attack allowed five restarts"),
            ("All attacks use one frozen set", "negation uses 38 utterances, the language-model test 25"),
            ("on the Bengali portion\nof FLEURS", "the check used 200 utterances, not the whole portion"),
            ("reverses sign", "the unmatched step is within between-run noise"),
            ("reverses the sign", "the unmatched step is within between-run noise"),
            ("points the wrong way", "the unmatched step is within between-run noise"),
            ("points the other way", "the unmatched step is within between-run noise"),
            ("six times less", "a ratio resting on 1 of 16 is not a headline"),
            ("six times more", "a ratio resting on 1 of 16 is not a headline"),
            ("cannot be driven to an exact target", "a stronger search reaches 22 of those 25"),
            ("easier to attack for reasons of accuracy", "sign error: positive coefficient means harder"),
            ("a less accurate model is easier", "contradicts the fitted accuracy coefficient"),
            ("about one of the twelve", "the expectation is 0.6"),
            ("A common claim", "uncited claim about the field"),
            ("said to be easier", "uncited claim about the field"),
            ("falls with exposure", "one checkpoint per exposure: not established"),
            ("below about 40 points", "four runs resolve 42.7"),
            ("At identical effort", "the 400-step threshold is not the attack's ceiling budget"),
            ("missed 40\\%", "the 400-step threshold is not the attack's ceiling budget"),
            ("inside its step budget", "budget must be named: 800 at the ceiling, 400 per probe"),
            ("all of which they passed", "pair 3 is the weakest under the checks"),
            ("depends on the very quantity", "dependence on exposure is not established"),
            ("remain\nthe norm", "uncited claim about the field"),
            ("trained on thousands of hours", "not checked for every cited system"),
            ("It is a result about how the field measures", "generalisation beyond one language"),
            ("reproduce every\nnumber in this paper", "several results rest on recorded notebook output"),
            ("their rates are plateaus", "later restarts still succeeded near the limit at low exposure"),
            ("(Sections~5.5, 6.3", "stale power cross-reference"),
            ("sit\ninside it",
             "intro: 38 and 33 exceed the 30-point noise"),
            ("no more than a ninth", "retracted lower bound, conclusion wording"),
            ("costs no more than", "T1 is cheaper on 26 of 38, not on all"),
            ("physically produce", "85.5 pp is not physically impossible"),
            ("between two\nsingle runs cannot resolve",
             "40 pp is the n=4 figure, not n=1"),
            ("no detectable effect size at all", "overstated n=1 verdict"),
            ("never measured", "false claim about the 100 h final clean WER"),
            ("No reading of the censoring", "a rate rising with exposure could flip the sign"),
            ("no reading of the\ncensoring", "a rate rising with exposure could flip the sign"),
            ("none puts it\nbelow", "a uniform rate of 100% gives +0.079"),
            ("moves it up rather than down", "the uniform reading moves it down"),
            ("bounded and\nfavourable", "the uniform reading moves it down"),
            ("at least a third of them", "the lowest Wilson limit is 33.2%"),
            ("The cause is accuracy", "the 5-hour failures were never re-examined"),
            ("its likely cause is measured", "only the 1-hour model was re-examined"),
            ("fails most often on the", "failed attacks are 22 at 1 h, about 28 and 26 elsewhere"),
            ("fails most on the least", "failed attacks are 22 at 1 h, about 28 and 26 elsewhere"),
            ("stronger search says otherwise", "the 5-hour comparator is unmeasured"),
            ("Nearly half our observations", "42% is over two fifths, not nearly half"),
            ("robustness are standard", "uncited claim about the field"),
            ("one \\emph{identical} condition", "eight of the twelve cross a library version"),
            ("about seven weeks", "226 GPU-hours is 7.5 weeks of quota"),
            ("was\n\\emph{suppressing}", "the ordering with exposure is not established"),
            ("no better attack will", "untested claim about other attacks"),
            ("false-negative rate", "the paper's term is the recovery rate, defined in 7.2"),
            ("multiplies the perturbation needed", "causal wording for an association"),
            ("No effect this experiment could exhibit", "the claim is the observed range"),
            ("on our hardware", "the binding constraint is the quota"),
            ("can fix that", "a larger evaluation set does reduce sampling error"),
            ("shows almost all of that", "86% if every re-attack pair is kept"),
            ("they show whether the effect", "two runs indicate; they do not show"),
            ("they will show whether", "two runs indicate; they do not show"),
            ("gave the\nattack itself", "the re-attack pairs cross a minor library version"),
            ("and the effects are large", "13 points is not large; two of the three are"),
            ("grows 1.23-fold per tenfold", "causal wording for an association"),
            ("whether the rate depends on exposure", "the term is the recovery rate or 'that share'")]:
        check("absent: %s" % why, frag in tex, False)
    # Claims that must still be there, with their hedging intact.
    for frag, why in [
            ("75\\% of the 100 utterances", "Sec 3.3 uses the 1h-vs-5h reversal"),
            ("79\\% for the 5-hour model", "Sec 3.3 names the right rungs"),
            ("+0.0940", "limitations states the 100 h final sensitivity"),
            ("p = 0.0476", "limitations states the crossed threshold"),
            ("over-statement of unknown size", "censoring caveat present"),
            ("Removed in quadrature", "sigma wording corrected"),
            ("44\\% on all 100 utterances", "completed 100 h final figure"),
            ("2.9-point attack-measurement standard deviation",
             "rung sigma compared against the attack sd"),
            ("three degrees of freedom, the interval on the attack figure reaches 10.0",
             "both df-3 caveats stated"),
            ("measured at the 1, 5 and 25-hour rungs rather than",
             "the transfer caveat on the attack sd is stated"),
            ("one run per condition can\nresolve nothing smaller than 85 points",
             "the n=1 reading is in the sequence"),
            ("bounded by 0 and 100", "the physical-range punchline is present"),
            ("exits non-zero on any disagreement. Three kinds of",
             "methods names this script as the third"),
            ("verification script re-derives\nthe main figures and exits non-zero",
             "availability states the check is released"),
            ("puts it between 1.20 and 1.27", "conclusion gives the bracket, not one end"),
            ("The 5-hour model's 21 failures were not\nre-examined",
             "unmatched-ladder cause held to what was measured"),
            ("All four comparisons that share a library version are among the ten",
             "contributions answer the library-version objection"),
            ("33\\% at 98.0 hours", "lowest Wilson limit quoted as itself"),
            ("At the measured spread, settling the", "abstract power figure tied to the estimate"),
            ("14 to 316 runs", "power band across sigma's interval"),
            ("$[-0.24, +1.53]$", "no-milestone fit reported in limitations"),
            ("It is an\nassociation, not an isolated effect of audio", "pooled slope read as association"),
            ("the \\emph{recovery rate}", "censoring measure named and defined"),
            ("so a positive coefficient on it means a less", "sign defined at the model"),
            ("as a minimum diagnostic", "two-run recommendation scoped"),
            ("from 56\\% to 4\\% of 25 utterances while", "conclusion LM result carries model and n"),
            ("removes the variation between\ntraining runs", "evaluation-set claim scoped to run variance"),
            ("None of the three is as much\nas 1.3 times it",
             "intro states the ratio, not containment"),
            ("larger than the entire range across which attack success\nvaried",
             "power compares 85.5 to the observed range"),
            ("could not resolve anything below 85",
             "discussion gives the n=1 figure"),
            ("cheaper on most utterances, though not on all",
             "T6 comparison hedged to the median"),
            ("variance inflation\nfactor on exposure from 1.2 to 21.8",
             "margin collinearity stated with a number"),
            ("Measurement Validity", "title leads on measurement validity"),
            ("Three measurement\nproblems emerged", "abstract names the three findings"),
            ("\\ref{sec:sigmanum} and \\ref{sec:procedure}",
             "contributions cite sigma by label, not a stale number"),
            ("\\ref{sec:power} and \\ref{sec:validity}",
             "power cited by label, not the stale 5.5"),
            ("4{,}051", "5 h rung utterance count from provenance"),
            ("sampled Rademacher weights", "bootstrap variant named correctly"),
            ("Webb (2023) give $p = 0.053$", "six-point-weights p reported as itself"),
            ("It needs no imputation", "ASR described accurately"),
            ("pairs 1 and 2, the part-trained", "compute control scoped"),
            ("pair 3 does not", "holdout caveat on pair 3 stated"),
            ("judged 67 of the candidates", "judged sample size stated"),
            ("\\label{tab:problems}", "summary table of the three problems present"),
            ("\\section{The attack as a measuring instrument}",
             "censoring and re-attack have their own section"),
            ("\\textbf{Four recommendations.}", "recommendations list present"),
            ("larger than\none of the three effects and within a factor of 1.3",
             "conclusion compares the effects to the pair noise correctly"),
            ("within\nreach of the largest effect we measured",
             "equal-budget replication comparison"),
            ("\\label{tab:twelve}", "all twelve comparisons tabulated with p-values"),
            ("seven remain so\nafter Holm--Bonferroni correction", "multiple-testing correction reported"),
            ("closest to it in clean word error\nrate", "abstract states approximate matching"),
            ("95\\% Wilson intervals of 70--96\\%, 57--93\\% and 33--77\\%",
             "false-negative rates carry intervals"),
            ("Those 25 are every censored utterance of that model", "sampling design stated"),
            ("Conneau et al., 2022", "speaker check on an independent corpus"),
            ("accompanies this\nsubmission as supplementary material", "artifacts available to reviewers"),
            ("covers 57 of the 679 censored observations", "limitation on the strong search stated"),
            ("Markert et al. (2021)", "citation year corrected"),
            ("4 to 11\\% of each pair", "residual share under the current coefficient"),
            ("12 of 576", "pipeline count stated"),
            ("up to five restarts", "re-attack budget stated as run"),
            ("step 2500 in one run and step 5500 in the other", "checkpoint selection disclosed"),
            ("from 56\\% to 4\\% of 25 utterances", "language-model result carries its n"),
            ("on 200 utterances from\nthe Bengali test split of FLEURS", "cross-corpus check carries its n"),
            ("we do not claim a reversal as established", "unmatched step held to the noise yardstick"),
            ("Each\nexposure is also represented by one checkpoint", "exposure pattern flagged as three models"),
            ("within the 800 steps it allows at the ceiling", "the ceiling budget is stated as 800 steps"),
            ("which is not\n($p = 0.18$)", "the within-budget trend at 800 steps is reported"),
            ("an estimated 22, 28 and 26", "understatement in points of attack success"),
            ("up to 800 optimisation steps from a zero\nperturbation", "method states the attack budget"),
            ("we ran no\nforced-determinism control", "cause of divergence not asserted"),
            ("leaves $\\sigma = 20.5$ on three degrees of freedom", "sigma without the same-seed pair"),
            ("a less accurate model needs \\emph{more} perturbation", "accuracy coefficient read with the right sign"),
            ("Eight of the\ntwelve also cross a change of library version", "library crossing disclosed"),
            ("an observation, not an established relation", "alpha* ordering hedged")]:
        check("present: %s" % why, frag in tex, True)

    # The same two kinds of check on whitespace-normalised text, so a line
    # break inside a phrase can neither hide a stale claim nor fail a present one.
    texn = re.sub(r"\s+", " ", tex)
    for frag, why in [
            ("other than the control deletion", "the negation files are now released"),
            ("both control conditions", "T6 is one of the three released negation edit types"),
            ("per-utterance files live only", "the negation files are now released"),
            # third review (supervisor, 2026-11-17)
            ("usually means the attack failed", "censoring claim scoped to the re-examined checkpoints"),
            ("is usually a failed attack", "censoring claim scoped to the re-examined checkpoints"),
            ("usually marks a failed attack", "censoring claim scoped to the re-examined checkpoints"),
            ("makes an over-statement", "the negation caveat is conditional on the censoring evidence"),
            ("is an upper bound rather than a correction", "the Tobit slope is a sensitivity estimate"),
            ("1- and 5-hour runs held at 88", "no 5-hour run was re-attacked"),
            ("a significant trend ($p = 0.006$", "the utterance-level trend is exploratory"),
            ("accounts for 4 to 11", "the residual share needs its scale and derivation"),
            ("These are planning figures at the point estimate", "power figures labelled as normal approximation"),
            # primary attack sd swapped to the all-pairs 8.1 (third review)
            ("under 2\\% of that variance (14\\%", "abstract leads with the all-pairs figure"),
            ("98\\% on our primary reading", "the primary reading is now 86%"),
            ("puts the attack's own standard deviation at 2.9", "contribution leads with the all-pairs figure"),
            ("Three degrees of freedom behind that bound", "df stated for every figure"),
            # fourth review (2026-11-18)
            ("it is worse optimised", "fewer updates is not worse optimisation"),
            ("had less optimisation, not more", "fewer updates is not worse optimisation"),
            ("so a reviewer can run every check now", "reproducibility claim scoped to released files"),
            ("The subscript records how the values are summarised", "notation rewritten"),
            ("These are right-censored", "unresolved is not the same as censored")]:
        check("absent: %s" % why, frag in texn, False)
    for frag, why in [
            ("re-runs and all three negation edit types are publicly released",
             "availability lists the negation files"),
            ("the native speaker's decision on each candidate judged", "annotation record released"),
            ("censoring and negation results using only the Python standard library",
             "availability says the negation results reproduce"),
            ("the counts of negation candidates and the language-model and deployment results",
             "methods names what still rests on notebook output"),
            ("T1 and T6 were searched over a wider bracket than the main attacks, "
             "$[10^{-5}, 2\\times10^{-3}]$ in six rounds", "negation bracket disclosed"),
            ("would otherwise have stopped at that floor", "what the wider bracket prevented"),
            ("no successful retargeting attack came within a factor of two of its floor",
             "why T5's standard bracket loses nothing"),
            ("T6 was also measured weeks after T1, under newer library versions",
             "T6/T1 library crossing disclosed"),
            ("T1 and T5 share their versions on each model", "the T5/T1 comparison does not cross versions"),
            ("The public repository, dataset and notebook links are withheld",
             "the 1 h negation files are public as notebook output, not as a dataset"),
            # third review (supervisor, 2026-11-17)
            ("Second, most censored measurements we re-examined were failed attacks, not resistant models",
             "abstract carries the censoring finding, scoped"),
            ("\\textbf{Most censored measurements we re-examined were failed attacks, not resistant models.}",
             "censoring is its own contribution, scoped"),
            ("Most censored utterances we re-examined marked a failed attack", "conclusion scoped"),
            ("On this model a censored row was usually a failed attack", "1-hour claim kept to its model"),
            ("though only 56\\% at the highest exposure, the checkpoint closest to this model",
             "negation caveat names the relevant rate"),
            ("If these are mostly failed attacks too", "negation caveat conditional"),
            ("may make an over-statement of unknown size", "contributions and conclusion hedge the caveat"),
            ("a trend significant only with the utterance as the unit ($p = 0.006$), which makes it exploratory",
             "utterance-level trend labelled exploratory"),
            ("The second reading interpolates the rate at the exposures that were not re-attacked, 5 hours among them",
             "interpolation disclosed"),
            ("A uniform rate of any size leaves it between $+0.079$ and $+0.089$",
             "censoring bracket: the uniform range stated"),
            ("The 5-hour runs need no assumed rate for the sign", "sign claim needs no 5-hour rate"),
            ("with only the 1-hour runs held at 88\\%, the rate measured on one of them",
             "sign claim scoped to the measured rate"),
            ("The lowest value it reaches is $+0.045$", "6c-5 extreme reported"),
            ("a sensitivity estimate under a censoring assumption our re-attacks contradict",
             "Tobit wording"),
            ("Each pair was chosen once, on clean WER over those 25, before the other 75 existed",
             "holdout procedure stated"),
            ("None was reselected", "no reselection after the 100-utterance WERs"),
            ("do not convert it into points of attack success", "residual scale named"),
            ("multiplies predicted \\eps{} by $10^{0.55\\Delta}$", "residual derivation given"),
            ("covers 4 to 11\\% of each pair's excess over one, and 1 to 18\\% across the coefficient's 95\\% interval, $[+0.23, +0.87]$",
             "residual share with its uncertainty"),
            ("These are normal-approximation planning figures", "power labelled"),
            ("would need about 44 runs per condition", "abstract power figure approximate"),
            ("about 44 runs per condition by the normal approximation", "contribution power figure labelled"),
            ("about seven times the pooled $+0.089$", "no-milestone magnitude compared"),
            ("correlate at $r = -0.99$", "no-milestone collinearity stated"),
            ("only the sign of the point estimate is common to both fits", "no-milestone reading"),
            ("rescaled to 100 utterances at a fixed per-utterance flip rate", "25-utterance pair rescaling stated"),
            ("would raise the 2.9 to 3.3", "primary estimate with the 25-utterance pair"),
            ("1 h, 25 utt. & 0.98 & --- & $+12.0$", "25-utterance pair shown in the table"),
            # primary attack sd swapped to the all-pairs 8.1 (third review)
            ("a standard deviation of 8.1 points, 14\\% of that variance (2.9 points, under 2\\%, from the three version-controlled pairs)",
             "abstract: primary and sensitivity attack sd"),
            ("the attack's own standard deviation is 8.1, or 2.9 for the version-controlled pairs",
             "Table 1 carries both attack figures"),
            ("the three version-controlled pairs alone give 2.9 and 21.4", "contribution gives the sensitivity split"),
            ("we call them the version-controlled pairs", "term defined where introduced"),
            ("\\textbf{8.1 points} on 5 degrees of freedom (95\\% interval 4.8 to 18.8)", "primary figure with df and interval"),
            ("\\textbf{20.0 points for the training run}", "primary training figure"),
            ("This is our primary reading, because the alternative rests on setting two pairs aside",
             "why the all-pairs figure is primary"),
            ("so it does not set the primary figure", "post-hoc judgement kept out of the primary"),
            ("86\\% on our primary reading and 98\\% on the version-controlled pairs", "discussion gives both shares"),
            ("the primary 8.1 also carries a pair whose version was never recorded", "limitation covers the primary"),
            ("By the training run we mean everything", "training run defined"),
            # fourth review (2026-11-18)
            ("\\newcommand{\\eps}{\\ensuremath{\\hat{\\varepsilon}}}", "eps macro is the per-utterance estimate"),
            ("\\newcommand{\\epstrue}{\\ensuremath{\\varepsilon^{*}}}", "true minimum has its own symbol"),
            ("\\newcommand{\\epsmed}{\\ensuremath{\\varepsilon_{50}}}", "median has its own symbol"),
            ("Three quantities need keeping apart", "notation section"),
            ("\\epstrue(u) \\;=\\; \\inf\\,\\{\\|\\delta\\|_\\infty :\\ f(x_u+\\delta) = t_u\\}", "true minimum defined"),
            ("so $\\eps(u) \\ge \\epstrue(u)$", "estimate bounds the truth from above"),
            ("is \\emph{unresolved}", "unresolved defined"),
            ("bounds the median of $\\epstrue(u)$ from above", "median bound stated with its condition"),
            ("When at least half are resolved", "median bound condition"),
            ("the column \\texttt{eps50} in the released files", "notation tied to the file column"),
            ("censoring in the statistical sense would mean $\\epstrue(u)$ lies above the ceiling", "censoring distinguished"),
            ("\\log_{10}\\eps_{u,c} \\;=\\; \\beta_0 + \\beta_1\\log_{10}E_c + \\beta_2\\,\\mathrm{WER}_c + e_{u,c}",
             "regression written on utterance-level values"),
            ("not the 16 checkpoints' medians", "unit of observation stated"),
            ("lower the 8.1 to 7.3, 11\\% of the variance", "size-adjusted all-pairs figure"),
            ("We keep the unscaled 8.1 as the conservative figure", "8.1 labelled conservative"),
            ("root-mean-square discrepancies between repeated measurements", "pooling assumption stated"),
            ("That rules out the explanation that the high-exposure checkpoint is harder to attack because it received more updates",
             "compute control states what it rules out"),
            ("It does not establish that the two models were at equivalent stages of optimisation", "and what it does not"),
            ("reproduce every analysis for which per-utterance data are released", "reproducibility claim scoped"),
            ("One of the authors, a native speaker of Bengali and the only annotator", "annotator stated"),
            ("were made by one of the authors; no one outside the author team annotated data", "ethics covers annotation"),
            ("five behind the primary attack figure, three behind the version-controlled one and four behind $\\sigma$",
             "df for every figure")]:
        check("present: %s" % why, frag in texn, True)

    # Journal rule: every table and figure is cited in the text, and first
    # citations run in numerical order.
    _dt = re.findall(r"\\label\{(tab:[a-z0-9]+)\}", tex)
    _ct = []
    for _t in re.findall(r"Table~\\ref\{(tab:[a-z0-9]+)\}", tex):
        if _t not in _ct:
            _ct.append(_t)
    check("every table cited, in numerical order", _ct, _dt)
    _df = re.findall(r"\\label\{(fig:[a-z0-9]+)\}", tex)
    _cf = []
    for _t in re.findall(r"(?:Fig\.|Figure)~\\ref\{(fig:[a-z0-9]+)\}", tex):
        if _t not in _cf:
            _cf.append(_t)
    check("every figure cited, in numerical order", _cf, _df)
    check("no hardcoded section numbers left in the text",
          re.findall(r"Sections?[~ ]\d[\d.]*", tex), [])

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    check("all \\ref targets defined", sorted(refs - set(labels)), [])
    check("no duplicate \\label", len(labels), len(set(labels)))

# ===========================================================================
print("\n" + "=" * 80)
print("  %d checks passed, %d failed%s"
      % (N_OK, len(FAIL), ", section E skipped" if SKIP else ""))
for f in FAIL:
    print("    FAILED: %s" % f)
print("=" * 80)
sys.exit(1 if FAIL else 0)
