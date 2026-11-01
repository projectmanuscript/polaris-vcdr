"""
Fieller interval against Monte Carlo for the FINAL method (TK, 90%), on every RIGA test image (94, de-duplicated).
For each image: the exact Fieller interval for A/B, and the 5th / 95th percentiles of A/B over 200,000 draws of (A, B) from the
same bivariate normal (means a, b; covariance from Sigma = tau (diag e + Sigma_rater)). Reports the endpoint differences.
Run:  python3 mc_check.py 2>&1 | tee outputs/mc_check_console.log        (about 5 min; re-executes final_run.py up to its fits)
"""

import os

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
fr = fr.replace(a_, "'\"outputs\", \"mc_check\")'")
cut = "# ---------------- Chaksu test set"
assert cut in fr
exec(fr.split(cut)[0])
rng_mc = np.random.default_rng(2027)
diffs, bad = [], 0
for d in TST:
    v = fit["TK"] * d["V"]["K"]
    vaa, vbb, vab = v
    iv = ivq(d["a"], d["b"], d["V"]["K"], fit["TK"], Z90)
    if iv is None:
        bad += 1
        continue
    smp = rng_mc.multivariate_normal([d["a"], d["b"]], [[vaa, vab], [vab, vbb]], 200000)
    r = smp[:, 0] / smp[:, 1]
    lo, hi = np.percentile(r, [5, 95])
    diffs.append(max(abs(lo - iv[0]), abs(hi - iv[1])))
diffs = np.array(diffs)
say(
    f"\nMONTE CARLO CHECK, final method (TK, tau={fit['TK']:.4f}), 90%, {len(diffs)} RIGA test images ({bad} unbounded skipped), 200,000 draws each:"
)
say(
    f"   largest endpoint difference per image: median {np.median(diffs):.2e}, 95th percentile {np.percentile(diffs, 95):.2e}, max {diffs.max():.2e}"
)
say(
    f"   mean interval width for scale: {np.mean([ivq(d['a'], d['b'], d['V']['K'], fit['TK'], Z90)[1] - ivq(d['a'], d['b'], d['V']['K'], fit['TK'], Z90)[0] for d in TST]):.4f}"
)
