"""
Three numbers for the paper. Re-executes final_run.py up to its segmentation block (outputs/review_numbers/); nothing refitted.
  1. Centre-line (chord) vCDR vs vertical-extent vCDR on every expert outline: mean |difference|, 95th percentile, max,
     Pearson r, mean signed difference (chord minus extent). RIGA test (94), RIGA calibration (117), Chaksu test (336).
  2. Fieller vs Monte Carlo (200,000 draws per image, same bivariate normal), final method TK at 90%, on all 94 RIGA and
     all 336 Chaksu test images (identical to mc_check.py, extended to Chaksu).
  3. Chaksu coverage per expert (TK at 90%; CFu shown for reference), 95% CI by resampling images (5000).
Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 review_numbers.py 2>&1 | tee outputs/review_numbers_console.log
"""

import os

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
fr = fr.replace(a_, "'\"outputs\", \"review_numbers\")'")
cut = "# ---------------- 2. segmentation on both datasets"
assert cut in fr
exec(fr.split(cut)[0])
say("\n==================== REVIEW NUMBERS ====================")

# ---------- 1. chord vs extent
say("\n1. CENTRE-LINE (chord) vCDR vs VERTICAL-EXTENT vCDR, every expert outline (chord minus extent)")
for nm, ds in [("RIGA test", TST), ("RIGA calib", CAL), ("Chaksu test", CT_)]:
    rx = np.concatenate([d["rx"] for d in ds])
    rv = np.concatenate([d["rv"] for d in ds])
    df = rv - rx
    ad = np.abs(df)
    say(
        f"   {nm:11s} ({len(ds)} images, {len(rx)} outlines): mean |diff| {ad.mean():.4f}, median {np.median(ad):.4f}, "
        f"95th pct {np.percentile(ad, 95):.4f}, max {ad.max():.4f}; mean signed {df.mean():+.4f}; Pearson r {np.corrcoef(rx, rv)[0, 1]:.4f}"
    )
rx = np.concatenate([d["rx"] for d in TST + CT_])
rv = np.concatenate([d["rv"] for d in TST + CT_])
ad = np.abs(rv - rx)
say(
    f"   all test (430 images, {len(rx)} outlines): mean |diff| {ad.mean():.4f}, 95th pct {np.percentile(ad, 95):.4f}, max {ad.max():.4f}"
)

# ---------- 2. Monte Carlo
say(f"\n2. MONTE CARLO CHECK, final method (TK, tau={fit['TK']:.5f}), nominal 90%, 200,000 draws per image")
rng_mc = np.random.default_rng(2027)
ALL = []
for nm, ds in [("RIGA test", TST), ("Chaksu test", CT_)]:
    diffs, rel, bad = [], [], 0
    for d in ds:
        vaa, vbb, vab = fit["TK"] * d["V"]["K"]
        iv = ivq(d["a"], d["b"], d["V"]["K"], fit["TK"], Z90)
        if iv is None:
            bad += 1
            continue
        smp = rng_mc.multivariate_normal([d["a"], d["b"]], [[vaa, vab], [vab, vbb]], 200000)
        r = smp[:, 0] / smp[:, 1]
        lo, hi = np.percentile(r, [5, 95])
        m = max(abs(lo - iv[0]), abs(hi - iv[1]))
        diffs.append(m)
        rel.append(m / (iv[1] - iv[0]))
    diffs, rel = np.array(diffs), np.array(rel)
    ALL.append(diffs)
    say(
        f"   {nm:11s}: {len(diffs)} images ({bad} unbounded skipped); largest endpoint difference per image: median {np.median(diffs):.2e}, "
        f"95th pct {np.percentile(diffs, 95):.2e}, max {diffs.max():.2e}; as a share of interval width: median {np.median(rel)*100:.2f}%, max {rel.max()*100:.2f}%"
    )
A = np.concatenate(ALL)
say(
    f"   all test   : {len(A)} images; median {np.median(A):.2e}, 95th pct {np.percentile(A, 95):.2e}, max {A.max():.2e}"
)

# ---------- 3. coverage per Chaksu expert
say("\n3. CHAKSU COVERAGE PER EXPERT, nominal 90% (95% CI: images resampled, 5000)")
IV = {k: get_ivs(k, CT_) for k in ["TK", "CFu"]}
for k in IV:
    say(f"   check {k} overall: {cov(score(IV[k], CT_)[0])*100:.1f}%")
experts = sorted({e for d in CT_ for e in d["ids"]}, key=str)
rng_b = np.random.default_rng(7)
for e in experts:
    idx = [i for i, d in enumerate(CT_) if e in d["ids"]]
    out = []
    for k in ["TK", "CFu"]:
        h = []
        for i in idx:
            d, iv = CT_[i], IV[k][i]
            x = d["rx"][list(d["ids"]).index(e)]
            h.append(1 if iv is None else int(iv[0] <= x <= iv[1]))
        h = np.array(h)
        bs = h[rng_b.integers(0, len(h), (5000, len(h)))].mean(1)
        out.append(f"{k} {h.mean()*100:5.1f}% [{np.percentile(bs, 2.5)*100:5.1f}, {np.percentile(bs, 97.5)*100:5.1f}]")
    vx = np.array([CT_[i]["rx"][list(CT_[i]["ids"]).index(e)] for i in idx])
    mu = np.array([CT_[i]["a"] / CT_[i]["b"] for i in idx])
    say(
        f"   Expert {e}: {len(idx)} images; "
        + "; ".join(out)
        + f"; expert vCDR minus POLARIS vCDR, mean {np.mean(vx - mu):+.3f}"
    )
say("\nREVIEW NUMBERS DONE")
