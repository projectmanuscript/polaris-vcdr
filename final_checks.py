"""
Final checks (read-only on data; writes outputs/final_checks/).
  A. Duplicate images between RIGA test and train/val/calib, confirmed at pixel level:
     candidates = pHash distance <= 12; duplicate = Pearson correlation >= 0.98 between 256x256 grayscale versions.
     Calibration-vs-train duplicates are listed too.
  B. Sensitivity: segmentation table (from phase8_v3crops LCC per-image results) and coverage table (calib_v4 fits,
     re-scored) on test images that have NO duplicate outside test. No refitting, no retraining.
  C. Chaksu test images with extreme rater disagreement (SD of raw extent vCDR > 0.2): each rater's vCDR, disc and cup area,
     and whether the cup lies inside the disc, plus a downsized overlay of every rater's masks.
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
src = open("calib_v4.py").read()
exec(src.replace('"outputs", "calib_v4")', '"outputs", "final_checks")').split("RES, IV = {}, {}")[0])
say("\n==== Final checks ====")
RES_ = {}


# ---------------- A. duplicates
def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


part = {k: s for s in ["train", "val", "calib", "test"] for k in split[s] if k.startswith("RIGA/")}
PATH, PH, SM = {}, {}, {}
for k in part:
    ds, case = k.split("/", 1)
    ip = resolve(json.load(open(os.path.join(PROCESSED_DIR, ds, case + ".json")))["image_path"])
    g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE) if ip else None
    if g is None:
        continue
    PATH[k] = ip
    v = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()[1:]
    PH[k] = v > np.median(v)
    s_ = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    SM[k] = (s_ - s_.mean()) / (s_.std() + 1e-6)


def dups(A, B):
    out = []
    for a in A:
        for b in B:
            if a == b:
                continue
            h = int(np.sum(PH[a] != PH[b]))
            if h <= 2:
                r = float(np.mean(SM[a] * SM[b]))
                if r >= 0.998:
                    out.append((a, b, h, r))
    return out


T = [k for k in PH if part[k] == "test"]
O = [k for k in PH if part[k] != "test"]
dt = dups(T, O)
dc = dups([k for k in PH if part[k] == "calib"], [k for k in PH if part[k] in ("train", "val")])
dirty = sorted({a for a, *_ in dt})
say(
    f"A. DUPLICATES (pHash <= 12 and pixel correlation >= 0.98): {len(dt)} test-vs-other pairs, involving {len(dirty)} of {len(T)} test images; "
    f"calib-vs-train/val pairs: {len(dc)}"
)
for a, b, h, r in sorted(dt, key=lambda x: -x[3]):
    say(f"   {a} (test)  =  {b} ({part[b]})   pHash {h}, r {r:.4f}")
by_src = {}
for a in dirty:
    s_ = re.match(r"[A-Za-z]+", a.split("/", 1)[1]).group(0)
    by_src[s_] = by_src.get(s_, 0) + 1
say(f"   affected test images by source: {by_src}")
corr_all = sorted(float(np.mean(SM[a] * SM[b])) for a in T for b in O if int(np.sum(PH[a] != PH[b])) <= 12)
say(
    f"   all candidate pairs (pHash <= 12): {len(corr_all)}; pixel correlation quartiles {np.percentile(corr_all, [25, 50, 75]).round(3).tolist() if corr_all else []} "
    f"(the 0.98 cut separates exact or re-saved copies from different photographs)"
)
pdir = os.path.join(OUT, "duplicates")
os.makedirs(pdir, exist_ok=True)
for i, (a, b, h, r) in enumerate(sorted(dt, key=lambda x: x[3])[:6]):
    ia, ib = cv2.imread(PATH[a]), cv2.imread(PATH[b])
    hh = 300
    ia = cv2.resize(ia, (int(ia.shape[1] * hh / ia.shape[0]), hh))
    ib = cv2.resize(ib, (int(ib.shape[1] * hh / ib.shape[0]), hh))
    cv2.imwrite(os.path.join(pdir, f"lowest_r_{i + 1}_r{r:.3f}.jpg"), np.hstack([ia, ib]))
RES_["duplicates_test"] = [(a, b, h, r) for a, b, h, r in dt]
RES_["duplicates_calib"] = [(a, b, h, r) for a, b, h, r in dc]

# ---------------- B. sensitivity without duplicated test images
say("\nB. SENSITIVITY: test images WITHOUT a duplicate outside test")
seg = json.load(open(os.path.join(BASE, "outputs", "phase8_v3crops", "segmentation_v3crops_lcc.json")))["per_image"]
clean_cases = {a.split("/", 1)[1] for a in dirty}
say(
    f"   {'model':17s} {'disc all':>9s} {'disc clean':>11s} {'cup all':>8s} {'cup clean':>10s} {'vCDR MAE all':>13s} {'clean':>7s}"
)
for m in ["harmonic_ens", "plain_ens", "attention_ens", "deepres_ens", "polar_ens", "harmonic_single"]:
    rows = seg[m]
    cl = [r for r in rows if r["case"] not in clean_cases]
    f = lambda rs, k: np.nanmean([r[k] for r in rs])
    say(
        f"   {m:17s} {f(rows, 'disc')*100:9.2f} {f(cl, 'disc')*100:11.2f} {f(rows, 'cup')*100:8.2f} {f(cl, 'cup')*100:10.2f} {f(rows, 'vcdr_ae'):13.4f} {f(cl, 'vcdr_ae'):7.4f}"
        + (f"   (n {len(rows)} -> {len(cl)})" if m == "harmonic_ens" else "")
    )
TC = [d for d in TST if d["key"] not in set(dirty)]
say(f"   coverage (individual raters, nominal 90%), all {len(TST)} vs clean {len(TC)} calibration-test images:")
for k in ["E", "K", "B", "C", "T", "CFh", "CFu", "UE"]:
    a_, b_ = score(test_ivs(k, TST), TST), score(test_ivs(k, TC), TC)
    say(
        f"   {NAMES[k]:34s} {cov(a_[0])*100:5.1f}% / width {np.nanmean(a_[1]):.4f}   ->   {cov(b_[0])*100:5.1f}% / width {np.nanmean(b_[1]):.4f}"
    )

# ---------------- C. Chaksu extreme-disagreement images
say("\nC. CHAKSU TEST images with extreme rater disagreement (raw extent vCDR SD > 0.2)")
MAN = json.load(open(os.path.join(BASE, "outputs", "chaksu", "rater_manifest.json")))


def bm(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    m = m.max(axis=2) if m.ndim == 3 else m
    return m > (m.max() / 2 if m.max() else 0)


cdir = os.path.join(OUT, "chaksu_extreme")
os.makedirs(cdir, exist_ok=True)
n_ext = 0
for cam in ["Bosch", "Forus", "Remidio"]:
    for k, e in sorted(MAN["Test"][cam].items()):
        info = []
        for rid, r in sorted(e["raters"].items()):
            d, c = bm(r["disc"]), bm(r["cup"])
            rd, rc = np.where(d.any(1))[0], np.where(c.any(1))[0]
            v = np.nan if len(rd) == 0 else (np.ptp(rc) + 1 if len(rc) else 0) / (np.ptp(rd) + 1)
            ncomp_c = cv2.connectedComponents(c.astype(np.uint8))[0] - 1
            info.append(
                (rid, v, int(d.sum()), int(c.sum()), float(np.logical_and(c, d).sum() / max(c.sum(), 1)), ncomp_c, d, c)
            )
        vs = [x[1] for x in info if not np.isnan(x[1])]
        if len(vs) >= 2 and np.std(vs, ddof=1) > 0.2:
            n_ext += 1
            say(
                f"   {cam}/{k}: "
                + "; ".join(
                    f"E{x[0]} vCDR {x[1]:.2f}, disc {x[2]} px, cup {x[3]} px, cup inside disc {x[4]*100:.0f}%, cup pieces {x[5]}"
                    for x in info
                )
            )
            img = cv2.imread(e["image"])
            sc = 600 / img.shape[0]
            small = cv2.resize(img, (int(img.shape[1] * sc), 600))
            for x, col in zip(info, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]):
                for m_ in (x[6], x[7]):
                    mm = cv2.resize(
                        m_.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(small, cs, -1, col, 2)
            cv2.imwrite(os.path.join(cdir, f"{cam}_{k}.jpg"), small)
say(
    f"   {n_ext} Chaksu test images with raw-vCDR SD > 0.2 (overlays in {cdir}; E1 red, E2 green, E3 blue, E4 yellow, E5 magenta)"
)
json.dump(RES_, open(os.path.join(OUT, "final_checks.json"), "w"), indent=1, default=float)
say("Saved", OUT)
