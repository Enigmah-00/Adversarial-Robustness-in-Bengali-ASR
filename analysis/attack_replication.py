#!/usr/bin/env python3
"""Attack-measurement variance at fixed weights.

Four pairs re-attack byte-identical weights on the 100-utterance T5 spine. The
aggregate ASR gap is a poor summary -- symmetric per-utterance churn cancels in
it -- so this script reports, for every pair:

  * the MATCHED gap, computed on the utterances both members measured. One
    condition finished 96 of 100, and comparing its 96-utterance ASR against a
    100-utterance published figure mixes the re-attack effect with a change of
    denominator.
  * the discordance, against what statistical independence would predict.
  * Cohen's kappa with a bootstrap interval, plus Fisher's exact test of
    whether the two measurements are associated at all, and McNemar's exact
    test of whether the shift is one-directional.
  * how far eps50 moves where both members succeeded, in units of the
    bisection bracket. A pair that disagrees by one bracket has reproduced
    eps50 to the resolution of the procedure.

Then it pools the pairs into a measurement standard deviation. Within a pair of
single measurements the difference is sqrt(2) times the standard deviation
behind it, so each pair contributes diff^2 / 2.

Stdlib only: no numpy, no scipy. Run from anywhere.
"""
import csv, math, random, statistics as st, os, sys
from math import comb, lgamma, exp, log, log10, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(os.path.dirname(HERE), "data", "results")
SIGMA = 21.574          # between-run sd, c4-corrected, from reanalysis.py
BOOTS = 20000
SEED  = 1337

# Each pair: label, exposure hours, the two files, and the software history.
# published file first, re-attack second.
PAIRS = [
    ("1h-traj",      0.98, "eps50_rung1-trajfinal_T5_n100.csv",
                           "eps50_rung1-traj-SECOND-MEASUREMENT_T5_n100.csv",
                           "4.44.2 vs UNRECORDED"),
    ("rung1-s1338",  0.98, "eps50_rung1-s1338_T5_n100.csv",
                           "attackrep_rung1-s1338_a2026_T5_n100.csv",
                           "5.16.1 -> 5.20.2"),
    ("rung5-s1338",  4.90, "eps50_rung5-s1338_T5_n100.csv",
                           "attackrep_rung5-s1338_a2026_T5_n100.csv",
                           "5.16.1 -> 5.20.2"),
    ("rung25-s1338", 24.60, "eps50_rung25-s1338_T5_n100.csv",
                            "attackrep_rung25-s1338_a2026_T5_n100.csv",
                            "5.16.1 -> 5.20.2"),
]


# ---------------------------------------------------------------- loading ---
def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, newline='', encoding='utf-8') as f:
        d = {}
        for x in csv.DictReader(f):
            assert x['utt_id'] not in d, f"duplicate utt_id in {name}"
            d[x['utt_id']] = x
    return d

def hit(row):
    return row['censored'].strip().lower() == 'false'

def eps(row):
    v = row['eps50'].strip()
    return float(v) if v else None


# ------------------------------------------------------------- statistics ---
def c4(df):
    """Bias correction for a standard deviation. Project convention:
    c4(df) is the standard c4 evaluated at df+1 observations."""
    n = df + 1
    return sqrt(2.0/(n-1)) * exp(lgamma(n/2.0) - lgamma((n-1)/2.0))

def kappa(pairs):
    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    neither = sum(1 for a, b in pairs if not a and not b)
    p1 = sum(1 for a, b in pairs if a)/n
    p2 = sum(1 for a, b in pairs if b)/n
    po = (both + neither)/n
    pe = p1*p2 + (1-p1)*(1-p2)
    return (po - pe)/(1 - pe) if pe < 1 else float('nan')

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2*sum(comb(n, i) for i in range(k, n+1))/2**n)

def fisher_exact(a, b, c, d):
    n, r1, c1 = a+b+c+d, a+b, a+c
    pr = lambda k: comb(r1, k)*comb(n-r1, c1-k)/comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1-(n-r1)), min(r1, c1)
    return min(1.0, sum(pr(k) for k in range(lo, hi+1) if pr(k) <= p0 + 1e-12))

def _gammainc_reg(a, x):
    if x <= 0:
        return 0.0
    if x < a + 1:                                   # series
        s = t = 1.0/a
        for n in range(1, 10001):
            t *= x/(a+n); s += t
            if abs(t) < abs(s)*1e-16:
                break
        return s*exp(-x + a*log(x) - lgamma(a))
    tiny = 1e-300                                   # continued fraction
    b, c, d = x + 1 - a, 1/tiny, 1/(x + 1 - a)
    h = d
    for i in range(1, 10001):
        an = -i*(i-a); b += 2
        d = an*d + b
        if abs(d) < tiny: d = tiny
        c = b + an/c
        if abs(c) < tiny: c = tiny
        d = 1/d; de = d*c; h *= de
        if abs(de - 1) < 1e-16:
            break
    return 1.0 - exp(-x + a*log(x) - lgamma(a))*h

def chi2_quantile(p, df):
    lo, hi = 1e-12, 1e4
    for _ in range(300):
        mid = (lo + hi)/2
        if _gammainc_reg(df/2.0, mid/2.0) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


# ------------------------------------------------------------------ main ---
def main():
    random.seed(SEED)
    print("=" * 78)
    print("ATTACK-MEASUREMENT VARIANCE AT FIXED WEIGHTS")
    print("=" * 78)
    print()

    results, boots_by_pair = [], {}
    for label, hours, f1, f2, versions in PAIRS:
        A, B = load(f1), load(f2)
        if A is None or B is None:
            print(f"{label}: SKIPPED, missing {f1 if A is None else f2}")
            continue
        common = sorted(set(A) & set(B))
        n = len(common)
        pairs = [(hit(A[u]), hit(B[u])) for u in common]

        asr1 = 100*sum(1 for a, b in pairs if a)/n
        asr2 = 100*sum(1 for a, b in pairs if b)/n
        asr1_full = 100*sum(hit(x) for x in A.values())/len(A)

        both = sum(1 for a, b in pairs if a and b)
        only1 = sum(1 for a, b in pairs if a and not b)
        only2 = sum(1 for a, b in pairs if not a and b)
        neither = sum(1 for a, b in pairs if not a and not b)
        disc = only1 + only2

        k = kappa(pairs)
        bs = []
        for _ in range(BOOTS):
            samp = [pairs[random.randrange(n)] for _ in range(n)]
            v = kappa(samp)
            if v == v:
                bs.append(v)
        boots_by_pair[label] = bs            # DRAW ORDER: sorting here would
        sb = sorted(bs)                      # turn a later difference into a
        klo, khi = sb[int(.025*len(sb))], sb[int(.975*len(sb))]

        pbar = (asr1 + asr2)/200
        indep = 100*2*pbar*(1-pbar)

        succ_both = [u for u in common if hit(A[u]) and hit(B[u])]
        lr = [log10(eps(B[u])/eps(A[u])) for u in succ_both]
        ident = sum(1 for u in succ_both if eps(A[u]) == eps(B[u]))
        bw = float(A[common[0]]['bracket_width'])
        brackets = st.median(abs(v) for v in lr)/log10(bw)

        print(f"--- {label}   exposure {hours} h   transformers {versions}")
        print(f"    n measured in both {n}"
              + (f"  (published file has {len(A)}; {len(A)-n} not re-attacked)"
                 if n != len(A) else ""))
        print(f"    published ASR {asr1:5.1f}%   re-attack ASR {asr2:5.1f}%"
              f"   MATCHED gap {asr2-asr1:+5.2f} pp")
        if n != len(A):
            print(f"    (the unmatched gap, against the {len(A)}-utterance published"
                  f" figure, would be {asr2-asr1_full:+.2f} pp)")
        print(f"    2x2  both {both} | published-only {only1} | re-attack-only {only2}"
              f" | neither {neither}")
        print(f"    discordance {100*disc/n:5.1f}%   independence would give"
              f" {indep:5.1f}%   ({100*(disc/n)/(indep/100):3.0f}% of it)")
        print(f"    kappa {k:+.3f}  bootstrap 95% CI [{klo:+.3f}, {khi:+.3f}]")
        print(f"    Fisher exact (associated at all?)   p = "
              f"{fisher_exact(both, only1, only2, neither):.4f}")
        print(f"    McNemar exact (one-directional?)    p = "
              f"{mcnemar_exact(only1, only2):.4f}")
        print(f"    eps50 where both succeeded: n={len(succ_both)},"
              f" byte-identical {ident} ({100*ident/len(succ_both):.0f}%),"
              f" median disagreement {brackets:.2f} bisection brackets,"
              f" worst x{10**max(abs(v) for v in lr):.2f}")
        print()
        results.append((label, hours, abs(asr2-asr1), k, versions))

    # ---- pooling -----------------------------------------------------------
    print("=" * 78)
    print("POOLED MEASUREMENT STANDARD DEVIATION")
    print("=" * 78)
    print()
    print("A pair of single measurements differs by sqrt(2) times the sd behind")
    print("it, so each pair contributes diff^2 / 2.")
    print()

    controlled = [r for r in results if "UNRECORDED" not in r[4]]
    LEGACY_25 = 12.0        # NB-05c vs NB-07 RUNG=1, 64.0% vs 76.0% on n=25
    rescaled = LEGACY_25/sqrt(100/25)

    sets = [
        ("the version-controlled pairs, measured",
         [(r[0], r[2]) for r in controlled]),
        ("those, plus the 25-utterance legacy pair rescaled to n=100",
         [(r[0], r[2]) for r in controlled] + [("nb05c/nb07 rescaled", rescaled)]),
        ("every pair at face value, including both confounded ones",
         [(r[0], r[2]) for r in results] + [("nb05c/nb07 raw", LEGACY_25)]),
    ]
    for label, gaps in sets:
        df = len(gaps)
        raw = sqrt(st.mean(g*g/2 for _, g in gaps))
        corr = raw/c4(df)
        lo = raw*sqrt(df/chi2_quantile(0.975, df))
        hi = raw*sqrt(df/chi2_quantile(0.025, df))
        train = sqrt(SIGMA**2 - corr**2) if corr < SIGMA else float('nan')
        print(f"{label}")
        print(f"  gaps (pp) " + ", ".join(f"{t}={g:.2f}" for t, g in gaps))
        print(f"  df {df}   raw sd {raw:6.3f}   c4-corrected {corr:6.3f} pp"
              f"   chi-square 95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  sigma {SIGMA} pp total  ->  TRAINING {train:6.3f} pp,"
              f"  attack share of variance {100*corr**2/SIGMA**2:4.1f}%")
        print()

    print(f"The 25-utterance pair is 3 flipped utterances out of 25. A fixed")
    print(f"per-utterance flip rate makes the aggregate gap scale as 1/sqrt(n),")
    print(f"so 12 pp on 25 utterances corresponds to {rescaled:.1f} pp on 100.")
    print("Reported at face value it is the second-largest gap in the set purely")
    print("because its denominator is a quarter of the size.")
    print()

    # ---- does exposure order the agreement? --------------------------------
    print("=" * 78)
    print("DOES EXPOSURE ORDER THE AGREEMENT?")
    print("=" * 78)
    print()
    for label, hours, gap, k, versions in sorted(results, key=lambda r: r[1]):
        print(f"  {label:14s} {hours:6.2f} h   kappa {k:+.3f}   gap {gap:5.2f} pp"
              f"   {versions}")
    same = [r for r in results if abs(r[1] - 0.98) < 1e-9]
    if len(same) == 2:
        a, b = same
        ba, bb = boots_by_pair[a[0]], boots_by_pair[b[0]]
        m = min(len(ba), len(bb))
        d = [bb[t] - ba[t] for t in range(m)]
        ds = sorted(d)
        p = min(1.0, 2*min(sum(1 for v in d if v <= 0),
                           sum(1 for v in d if v >= 0))/m)
        print()
        print(f"  Two pairs sit at identical exposure ({a[1]} h).")
        print(f"  kappa({b[0]}) - kappa({a[0]}) = {st.mean(d):+.3f}"
              f"  95% CI [{ds[int(.025*m)]:+.3f}, {ds[int(.975*m)]:+.3f}]"
              f"  p = {p:.4f}")
        print("  Exposure cannot explain a difference between two pairs that")
        print("  share it. What separates them is the software history.")

if __name__ == "__main__":
    main()
