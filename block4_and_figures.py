"""
Block 4 checks + the two figures. Read-only on all data; writes outputs/block4/ and outputs/figures/.
  1. Filtered rater pairs (extraction-unstable registry): are they the OUTLYING raters? Their vCDR deviation from the other
     raters vs the same for unflagged raters (all RIGA splits), and test coverage of E with them added back.
  2. Count reconciliation: images per split before/after the >=2-valid-rater rule (with reasons), RIGA totals
     (raw folders, processed json, split), misregistered images, and candidate inter-rater "ceiling" values on test.
  3. Perceptual-hash (DCT pHash) check for repeated images between RIGA test and train/val/calib; closest pairs saved
     side by side for a visual check.
  4. Chaksu: rater disagreement (SD of the raters' extent vCDR, from the full-resolution masks) in TRAIN vs TEST, per
     camera, and for the 99 images used for recalibration.
  Figures (black and white): (a) |correlation| of rater deviations, raw radii vs harmonic coefficients (TRAIN images);
     (b) reliability: nominal vs empirical individual-rater coverage on RIGA test for E, fixed-width conformal, per-coef temperature.
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
src = open("calib_v4.py").read()
exec(src.replace('"outputs", "calib_v4")', '"outputs", "block4")').split("RES, IV = {}, {}")[0])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
say("\n==== Block 4 checks ====")
OUTJ = {}


# ---------------- 1. filtered rater pairs
def rater_all(key):
    ds, case = key.split("/", 1)
    rec = json.load(open(os.path.join(PROCESSED_DIR, ds, case + ".json")))
    tr = json.load(open(os.path.join(PREPROC_DIR, ds, case + "_transform.json")))
    out = []
    for rid, data in rec["raters"].items():
        dig = re.sub(r"\D", "", str(rid))
        flag = norm(case.replace("_", "/", 1) + "_rater" + dig) in unstable
        u, rr = [], []
        for sk in ("disc_contour", "cup_contour"):
            c = data.get(sk)
            if c is None:
                break
            crop = map_to_crop_space(c, tr["x_min"], tr["y_min"], tr["scale"])
            if not valid_contour(crop):
                break
            r = contour_to_radial(crop)[1]
            rr.append(r)
            u.append(Phi.T @ r)
        ok = len(u) == 2
        out.append(
            dict(
                rid=dig,
                flag=flag,
                valid=ok,
                u=np.concatenate(u) if ok else None,
                r=np.concatenate(rr) if ok else None,
                rx=(ext(u[1]) / ext(u[0])) if ok else None,
            )
        )
    return out


ALL = {}
for s in ["train", "val", "calib", "test"]:
    for k in split[s]:
        if k.startswith("RIGA/"):
            ALL[k] = (s, rater_all(k))
dev_f, dev_u, n_flag, n_flag_valid = [], [], 0, 0
for k, (s, rs) in ALL.items():
    good = [r for r in rs if r["valid"] and not r["flag"]]
    for r in rs:
        if r["flag"]:
            n_flag += 1
        if not r["valid"]:
            continue
        others = [o["rx"] for o in good if o is not r]
        if len(others) < 2:
            continue
        d = abs(r["rx"] - np.mean(others))
        if r["flag"]:
            dev_f.append(d)
            n_flag_valid += 1
        else:
            dev_u.append(d)
dev_f, dev_u = np.array(dev_f), np.array(dev_u)
p = mannwhitneyu(dev_f, dev_u, alternative="greater").pvalue if len(dev_f) else float("nan")
say(
    f"1. FILTERED RATER PAIRS: {n_flag} flagged rater/image pairs in RIGA; {n_flag_valid} still give valid contours and >= 2 unflagged peers."
)
say(
    f"   |rater vCDR - mean of other raters| (extent): flagged median {np.median(dev_f):.4f} (90th {np.percentile(dev_f, 90):.4f}), "
    f"unflagged median {np.median(dev_u):.4f} (90th {np.percentile(dev_u, 90):.4f}); one-sided Mann-Whitney p (flagged larger) = {p:.3g}"
)
say(f"   share above 0.10: flagged {np.mean(dev_f > 0.10)*100:.1f}%, unflagged {np.mean(dev_u > 0.10)*100:.1f}%")
IVE = test_ivs("E", TST)
add = 0
for d in TST:
    extra = [r["rx"] for r in ALL[d["key"]][1] if r["flag"] and r["valid"]]
    add += len(extra)
    d["rx_all"] = np.concatenate([d["rx"], extra]) if extra else d["rx"]
c0, c1 = cov(score(IVE, TST)[0]), cov(score(IVE, TST, "rx_all")[0])
say(
    f"   E on RIGA test: individual-rater coverage {c0*100:.1f}% without the flagged raters, {c1*100:.1f}% with {add} flagged-but-valid raters added back"
)
OUTJ["filtered"] = dict(
    n_flag=n_flag,
    n_valid=n_flag_valid,
    med_flag=float(np.median(dev_f)),
    med_unflag=float(np.median(dev_u)),
    p=float(p),
    cov_without=c0,
    cov_with=c1,
    added=add,
)

# ---------------- 2. counts
say("\n2. COUNT RECONCILIATION")
for s in ["train", "val", "calib", "test"]:
    keys = [k for k in split[s] if k.startswith("RIGA/")]
    kept = [d["key"] for d in D[s]] if s in D else []
    reasons = {}
    for k in keys:
        if k in kept:
            continue
        rs = ALL[k][1]
        v = sum(r["valid"] and not r["flag"] for r in rs)
        f = sum(r["flag"] for r in rs)
        bad = sum(not r["valid"] for r in rs)
        reasons[k.split("/", 1)[1]] = f"{v} valid unflagged, {f} flagged, {bad} invalid contour (of {len(rs)} raters)"
    say(
        f"   {s:6s}: {len(keys)} RIGA images in split, {len(kept)} with >= 2 valid unflagged raters, {len(keys) - len(kept)} excluded"
    )
    for k, r in list(reasons.items())[:12]:
        say(f"      {k}: {r}")
riga_json = sorted(
    os.path.splitext(f)[0] for f in os.listdir(os.path.join(PROCESSED_DIR, "RIGA")) if f.endswith(".json")
)
in_split = sorted(k.split("/", 1)[1] for s in split for k in split[s] if k.startswith("RIGA/"))
raw = {}
for root, _, fs in os.walk(os.path.join(BASE, "RIGA")):
    n = sum(1 for f in fs if re.search(r"prime\.(jpe?g|tiff?)$", f, re.I))
    if n:
        raw[os.path.relpath(root, BASE)] = n
say(f"   raw RIGA 'prime' images on disk: {sum(raw.values())} " + str(raw))
say(
    f"   processed RIGA records: {len(riga_json)}; RIGA keys across all splits: {len(in_split)} (unique {len(set(in_split))})"
)
say(
    f"   in processed but not in any split: {sorted(set(riga_json) - set(in_split))[:10]}; in a split but not processed: {sorted(set(in_split) - set(riga_json))[:10]}"
)
say(f"   misregistered (removed): {reg.get('misregistered_images')}")


def pair_dice(us, part):
    ms = [coeffs_to_mask(u[part], Phi) > 0 for u in us]
    pw = [dice(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    lo = [
        dice(ms[i], coeffs_to_mask(np.mean([us[j][part] for j in range(len(us)) if j != i], 0), Phi) > 0)
        for i in range(len(ms))
    ]
    return np.mean(pw), np.mean(lo)


cd = [pair_dice(d["U"], slice(25, 50)) for d in TST]
dd_ = [pair_dice(d["U"], slice(0, 25)) for d in TST]
say(
    f"   inter-rater ceiling candidates on RIGA test ({len(TST)} images, valid unflagged raters, M=12 contours): "
    f"cup pairwise Dice {np.mean([c[0] for c in cd])*100:.2f}, cup rater vs mean-of-others {np.mean([c[1] for c in cd])*100:.2f}; "
    f"disc pairwise {np.mean([c[0] for c in dd_])*100:.2f}, disc vs mean-of-others {np.mean([c[1] for c in dd_])*100:.2f}"
)

# ---------------- 3. perceptual hash
say("\n3. PERCEPTUAL HASH (DCT pHash, 63 bits) between RIGA test and train/val/calib")


def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


def phash(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    v = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()[1:]
    return v > np.median(v)


H, P = {}, {}
for k, (s, _) in ALL.items():
    ds, case = k.split("/", 1)
    ip = resolve(json.load(open(os.path.join(PROCESSED_DIR, ds, case + ".json")))["image_path"])
    if ip:
        H[k], P[k] = phash(ip), ip
tk = [k for k in H if ALL[k][0] == "test"]
ok_ = [k for k in H if ALL[k][0] != "test"]
pairs = sorted(((int(np.sum(H[a] != H[b])), a, b) for a in tk for b in ok_), key=lambda x: x[0])
mins = [min(int(np.sum(H[a] != H[b])) for b in ok_) for a in tk]
say(
    f"   hashed {len(H)} RIGA images; test {len(tk)} vs other splits {len(ok_)}. Nearest-neighbour distance per test image: "
    f"min {min(mins)}, 5th pct {np.percentile(mins, 5):.0f}, median {np.median(mins):.0f}; test images with a neighbour at distance <= 6: {sum(m <= 6 for m in mins)}, <= 10: {sum(m <= 10 for m in mins)}"
)
pd = os.path.join(OUT, "phash_closest_pairs")
os.makedirs(pd, exist_ok=True)
for i, (dist_, a, b) in enumerate(pairs[:10]):
    say(f"   pair {i + 1}: distance {dist_}  {a} (test)  <->  {b} ({ALL[b][0]})")
    ia, ib = cv2.imread(P[a]), cv2.imread(P[b])
    h = 400
    ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
    ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
    cv2.imwrite(os.path.join(pd, f"pair{i + 1:02d}_d{dist_}.jpg"), np.hstack([ia, ib]))
OUTJ["phash"] = dict(
    min=int(min(mins)),
    le6=int(sum(m <= 6 for m in mins)),
    le10=int(sum(m <= 10 for m in mins)),
    closest=[(int(x), a, b) for x, a, b in pairs[:10]],
)

# ---------------- 4. Chaksu train vs test disagreement
say("\n4. CHAKSU rater disagreement, TRAIN vs TEST (SD across raters of the extent vCDR from full-resolution masks)")
MAN = json.load(open(os.path.join(BASE, "outputs", "chaksu", "rater_manifest.json")))


def bm(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    m = m.max(axis=2) if m.ndim == 3 else m
    return m > (m.max() / 2 if m.max() else 0)


def vx(dp, cp):
    d, c = bm(dp), bm(cp)
    rd, rc = np.where(d.any(1))[0], np.where(c.any(1))[0]
    return np.nan if len(rd) == 0 else (np.ptp(rc) + 1 if len(rc) else 0) / (np.ptp(rd) + 1)


rng_c = np.random.default_rng(2026)
drawn = set()
for cam in ["Bosch", "Forus", "Remidio"]:
    ks = sorted(MAN["Train"][cam])
    n = round(100 * len(ks) / sum(len(MAN["Train"][c]) for c in ["Bosch", "Forus", "Remidio"]))
    drawn |= {("Train", cam, k) for k in rng_c.choice(ks, n, replace=False)}
rowsC = []
rng_s = np.random.default_rng(11)
for sp_ in ["Train", "Test"]:
    for cam in ["Bosch", "Forus", "Remidio"]:
        keys = sorted(MAN[sp_][cam])
        if (
            sp_ == "Train"
        ):  # all 99 recalibration images + a random third of the rest (reading every full-size mask takes ~1 h)
            rest = [k for k in keys if ("Train", cam, k) not in drawn]
            keys = [k for k in keys if ("Train", cam, k) in drawn] + list(
                rng_s.choice(rest, len(rest) // 3, replace=False)
            )
        for k in keys:
            e = MAN[sp_][cam][k]
            v = [vx(r["disc"], r["cup"]) for r in e["raters"].values()]
            v = [x for x in v if not np.isnan(x)]
            if len(v) >= 2:
                rowsC.append(
                    dict(
                        split=sp_,
                        cam=cam,
                        k=k,
                        n=len(v),
                        sd=float(np.std(v, ddof=1)),
                        mean=float(np.mean(v)),
                        drawn=(sp_, cam, k) in drawn,
                    )
                )
say(
    f"   {'group':28s} {'images':>6s} {'raters/img':>10s} {'median SD':>9s} {'mean SD':>8s} {'SD > 0.0731':>11s} {'mean vCDR':>9s}"
)


def line(name, rs):
    if rs:
        say(
            f"   {name:28s} {len(rs):6d} {np.mean([r['n'] for r in rs]):10.2f} {np.median([r['sd'] for r in rs]):9.4f} {np.mean([r['sd'] for r in rs]):8.4f} "
            f"{np.mean([r['sd'] > 0.0731 for r in rs])*100:10.1f}% {np.mean([r['mean'] for r in rs]):9.3f}"
        )


for sp_ in ["Train", "Test"]:
    line(f"{sp_} all", [r for r in rowsC if r["split"] == sp_])
    for cam in ["Bosch", "Forus", "Remidio"]:
        line(f"{sp_} {cam}", [r for r in rowsC if r["split"] == sp_ and r["cam"] == cam])
line("Train, 99 recalibration images", [r for r in rowsC if r["drawn"]])
line("Train Remidio, 4 raters", [r for r in rowsC if r["split"] == "Train" and r["cam"] == "Remidio" and r["n"] == 4])
line("Train Remidio, 5 raters", [r for r in rowsC if r["split"] == "Train" and r["cam"] == "Remidio" and r["n"] == 5])
say("   (0.0731 = upper tertile cut-point of rater spread on RIGA calibration)")
OUTJ["chaksu_disagreement"] = rowsC

# ---------------- figures
say("\nFIGURES")
rawdev = []
for d in D["train"]:
    rs = [r for r in ALL[d["key"]][1] if r["valid"] and not r["flag"]]
    if len(rs) >= 2:
        R_ = np.array([r["r"] for r in rs])
        rawdev.append(R_ - R_.mean(0))
rawdev = np.concatenate(rawdev)
hdev = np.concatenate([d["U"] - d["ubar"] for d in D["train"]])
Craw, Ch = np.corrcoef(rawdev, rowvar=False), np.corrcoef(hdev, rowvar=False)
off = lambda C: float(np.abs(C[~np.eye(len(C), dtype=bool)]).mean())
say(
    f"   mean |off-diagonal correlation| of rater deviations on TRAIN: raw radii (720) {off(Craw):.4f}, harmonic coefficients (50) {off(Ch):.4f}"
)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
im0 = ax[0].imshow(np.abs(Craw[::4, ::4]), cmap="Greys", vmin=0, vmax=1)
ax[0].set_title(f"Radii (360 disc + 360 cup)\nmean |r| off-diagonal = {off(Craw):.3f}")
ax[0].set_xticks([0, 45, 90, 135, 179])
ax[0].set_xticklabels(["disc 0°", "", "cup 0°", "", "359°"])
ax[0].set_yticks([])
im1 = ax[1].imshow(np.abs(Ch), cmap="Greys", vmin=0, vmax=1)
ax[1].set_title(f"Harmonic coefficients (25 + 25)\nmean |r| off-diagonal = {off(Ch):.3f}")
ax[1].set_xticks([0, 24, 25, 49])
ax[1].set_xticklabels(["disc", "", "cup", ""])
ax[1].set_yticks([])
fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.02, label="|correlation|")
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig1_rater_deviation_correlation.{ext_}"), dpi=300, bbox_inches="tight")
plt.close(fig)
rel = json.load(open(os.path.join(BASE, "outputs", "calib_v4", "calib_v4_results.json")))["reliability"]
L = np.array(rel["levels"]) * 100
fig, a = plt.subplots(figsize=(3.6, 3.4))
a.plot([45, 100], [45, 100], color="0.6", lw=0.8, ls=(0, (4, 3)), label="ideal")
a.plot(L, np.array(rel["E"]) * 100, "k-o", ms=4, lw=1.2, label="POLARIS (E)")
a.plot(L, np.array(rel["CFh"]) * 100, "k--s", ms=4, lw=1.0, mfc="white", label="fixed-width conformal")
a.plot(L, np.array(rel["T"]) * 100, "k:^", ms=4, lw=1.0, label="per-coefficient temperature")
a.set_xlim(45, 100)
a.set_ylim(40, 100)
a.set_xlabel("Nominal coverage (%)")
a.set_ylabel("Individual-rater coverage, RIGA test (%)")
a.legend(frameon=False, loc="upper left")
a.grid(color="0.9", lw=0.5)
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig2_reliability.{ext_}"), dpi=300, bbox_inches="tight")
plt.close(fig)
OUTJ["figure1"] = dict(raw=off(Craw), harmonic=off(Ch))
json.dump(OUTJ, open(os.path.join(OUT, "block4_results.json"), "w"), indent=1, default=float)
say("Saved", OUT, "and", FIG)
