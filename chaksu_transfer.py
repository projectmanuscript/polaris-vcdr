"""
Chaksu zero-shot transfer test. Settings fixed before the run; each Chaksu TEST number is computed once.
  - Models, dispersion head, rater style and EVERY scalar / quantile come from RIGA unchanged (calib_v4 re-run up to its
    calibration fits; those are deterministic and identical to the calib_v4 run). Nothing is fitted on Chaksu test.
  - Chaksu images are cropped exactly like RIGA: YOLO disc_detector_v2split (trained on RIGA+DRIONS v2 train only),
    ROI = 2.6 x detected disc diameter, 512 px, CLAHE on green, per-image z-score. Images read with cv2 (the masks were
    verified against cv2-read images in the probe); channels converted to RGB as for RIGA.
  - Raters: outputs/chaksu/rater_manifest.json (Forus Expert 1 rebuilt from drawings, duplicate rule applied).
    Each rater mask -> largest outer contour -> crop space -> the same validity filters as RIGA -> M=12 coefficients.
    Images with fewer than 2 valid raters are excluded and counted.
  - Primary: individual-rater coverage (extent vCDR) at nominal 90% for all eight configurations of calib_v4,
    cluster bootstrap CIs over images. "Coverage holds" = the 95% CI of POLARIS (E) contains 90%.
  - Secondary, always reported and labelled: the same configurations recalibrated on 100 Chaksu TRAIN images
    (seeded, proportional by camera), then scored on the same Chaksu test images.
Outputs: outputs/chaksu_transfer/ (log, results json, crops, one overlay per camera for a hand check)
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
src = open("calib_v4.py").read()
assert src.count('"outputs", "calib_v4")') >= 1 and "RES, IV = {}, {}" in src
exec(src.replace('"outputs", "calib_v4")', '"outputs", "chaksu_transfer")').split("RES, IV = {}, {}")[0])
from ultralytics import YOLO

say(
    "\n==== Chaksu zero-shot transfer ====  RIGA fits in use: "
    + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 5)) for k, v in fit.items()})
)
RIGA_CUTS = np.percentile([np.std(d["rx"], ddof=1) for d in CAL], [100 / 3, 200 / 3])

MAN = json.load(open(os.path.join(BASE, "outputs", "chaksu", "rater_manifest.json")))
yolo = YOLO(os.path.join(BASE, "outputs", "yolo_runs", "disc_detector_v2split", "weights", "best.pt"))
CH_ROOT = os.path.join(BASE, "outputs", "chaksu", "preprocessed")
CH_DIR = os.path.join(CH_ROOT, "CHAKSU")
os.makedirs(CH_DIR, exist_ok=True)


def crop_and_normalize(img, center_x, center_y, disc_diameter, target_size=512):  # identical to initial.ipynb cell 56
    roi_side = disc_diameter * 2.6
    half = roi_side / 2
    x_min, y_min, x_max, y_max = int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)
    h, w = img.shape[:2]
    pl, pt, pr, pb = max(0, -x_min), max(0, -y_min), max(0, x_max - w), max(0, y_max - h)
    if any([pl, pt, pr, pb]):
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        x_min += pl
        x_max += pl
        y_min += pt
        y_max += pt
    crop = cv2.resize(img[y_min:y_max, x_min:x_max], (target_size, target_size))
    enh = crop.copy()
    enh[:, :, 1] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop[:, :, 1])
    f = enh.astype(np.float32)
    return (f - f.mean(axis=(0, 1), keepdims=True)) / (f.std(axis=(0, 1), keepdims=True) + 1e-6), crop


def binmask(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    m = m.max(axis=2) if m.ndim == 3 else m
    return m > (m.max() / 2 if m.max() else 0)


def outer(b):
    cs, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) if cs else None


stats = {
    "no_detection": [],
    "low_conf": [],
    "shape_mismatch": 0,
    "invalid_contours": 0,
    "lt2_raters": [],
    "centre_err": [],
    "n_raters": [],
}
overlay_done = set()


def process(split, cam, k, entry):
    img = cv2.imread(entry["image"])
    if img is None:
        stats["no_detection"].append(f"{split}/{cam}/{k} (unreadable)")
        return None
    r = yolo(img, conf=0.01, verbose=False)[0]
    if len(r.boxes) == 0:
        stats["no_detection"].append(f"{split}/{cam}/{k}")
        return None
    x, y, w, h = r.boxes[0].xywh[0].tolist()
    dia = max(w, h)
    conf = float(r.boxes[0].conf[0])
    if conf < 0.25:
        stats["low_conf"].append((f"{split}/{cam}/{k}", round(conf, 2)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nrm, view = crop_and_normalize(rgb, x, y, dia)
    case = f"{split}_{cam}_{k}"
    np.save(os.path.join(CH_DIR, case + "_normalized.npy"), nrm)
    x_min, y_min, sc = x - 1.3 * dia, y - 1.3 * dia, 512 / (2.6 * dia)
    rows, ids, boxes, crops = [], [], [], []
    for e, rr in sorted(entry["raters"].items()):
        u, cc = [], []
        for sk in ("disc", "cup"):
            b = binmask(rr[sk])
            if b.shape != img.shape[:2]:
                stats["shape_mismatch"] += 1
                break
            pts = outer(b)
            if pts is None:
                break
            if sk == "disc":
                boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            crop = map_to_crop_space(pts, x_min, y_min, sc)
            if not valid_contour(crop):
                stats["invalid_contours"] += 1
                break
            u.append(Phi.T @ contour_to_radial(crop)[1])
            cc.append(crop)
        if len(u) == 2:
            rows.append(np.concatenate(u))
            ids.append(e)
            crops.append(cc)
    if boxes:
        bb = np.mean(boxes, 0)
        stats["centre_err"].append(float(np.hypot(x - (bb[0] + bb[2]) / 2, y - (bb[1] + bb[3]) / 2) * sc))
    if len(rows) < 2:
        stats["lt2_raters"].append(f"{split}/{cam}/{k}")
        return None
    stats["n_raters"].append(len(rows))
    if split == "Test" and cam not in overlay_done:
        ov = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        for cc, col in zip(crops, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]):
            for c in cc:
                cv2.polylines(ov, [c.astype(np.int32).reshape(-1, 1, 2)], True, col, 1)
        cv2.drawMarker(ov, (256, 256), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.imwrite(os.path.join(OUT, f"overlay_{cam}_{k}.png"), ov)
        overlay_done.add(cam)
    return dict(key="CHAKSU/" + case, U=np.array(rows), ids=ids, src=cam)


def build(items):
    out = []
    for split, cam, k, entry in items:
        d = process(split, cam, k, entry)
        if d is not None:
            out.append(d)
    PREV = globals()["PREPROC_DIR"]
    globals()["PREPROC_DIR"] = CH_ROOT  # feat_mu / npy / ens_pred / unet_vcdr read from here
    for d in out:
        f, _ = feat_mu(d["key"])
        d["feat"] = f
        d["ubar"] = d["U"].mean(0)
        P = ens_pred(d["key"])
        d["mu"] = P.mean(0)
        d["e"] = np.maximum(P.var(0, ddof=1), floor)
        with torch.no_grad():
            d["v0"] = np.exp(head(torch.tensor(f[None])).numpy()[0]) * resid_std**2
        d["S"] = np.diag(d["v0"]) + style_cov
        d["rx"] = np.array([ext(u[25:]) / ext(u[:25]) for u in d["U"]])
        d["rv"] = (d["U"][:, 25:] @ phi_v) / (d["U"][:, :25] @ phi_v)
        d["a"], d["b"] = phi_v @ d["mu"][25:], phi_v @ d["mu"][:25]
        d["V"] = {"E": vq(np.diag(d["e"]) + d["S"]), "B": vq(np.diag(d["e"])), "K": vq(np.diag(d["e"]) + Sconst)}
        d["u_ens"], d["u_seeds"] = unet_vcdr(d["key"])
        d["u_sd"] = max(np.std(d["u_seeds"], ddof=1), sfloor)
    globals()["PREPROC_DIR"] = PREV
    return out


test_items = [("Test", cam, k, e) for cam in ["Bosch", "Forus", "Remidio"] for k, e in sorted(MAN["Test"][cam].items())]
rng_c = np.random.default_rng(2026)
cal_items = []
for cam in ["Bosch", "Forus", "Remidio"]:
    ks = sorted(MAN["Train"][cam])
    n = round(100 * len(ks) / sum(len(MAN["Train"][c]) for c in ["Bosch", "Forus", "Remidio"]))
    cal_items += [("Train", cam, k, MAN["Train"][cam][k]) for k in rng_c.choice(ks, n, replace=False)]
CT = build(test_items)
s_test = {k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()}
ce = np.array(stats["centre_err"])
say(
    f"Chaksu TEST: {len(test_items)} images, usable {len(CT)} ({sum(len(d['rx']) for d in CT)} rater pairs). "
    f"No detection {len(stats['no_detection'])}, low confidence {len(stats['low_conf'])}, <2 valid raters {len(stats['lt2_raters'])}, "
    f"mask/image size mismatches {stats['shape_mismatch']}, invalid contours {stats['invalid_contours']}"
)
say(
    f"  raters per usable image: { {n: stats['n_raters'].count(n) for n in sorted(set(stats['n_raters']))} };  "
    f"YOLO centre error (crop px): median {np.median(ce):.1f}, 90th {np.percentile(ce, 90):.1f}, max {ce.max():.1f}, >10 px {np.mean(ce > 10)*100:.1f}%"
)
if stats["no_detection"] or stats["lt2_raters"]:
    say(f"  excluded: {stats['no_detection'][:10]} {stats['lt2_raters'][:10]}")
for k in stats:
    stats[k] = [] if isinstance(stats[k], list) else 0
CC = build(cal_items)
say(f"Chaksu recalibration split (TRAIN, proportional by camera): {len(cal_items)} drawn, usable {len(CC)}")


def report(ds, title):
    say(f"\n{title}")
    say(
        f"  {'config':34s} {'rater cov':>9s} {'95% CI':>15s} {'width':>7s} {'95% CI':>15s} {'chord-def cov':>13s} {'mean-target':>11s} {'unbounded':>9s}"
    )
    R = {}
    for k in NAMES:
        iv = test_ivs(k, ds)
        hn, w = score(iv, ds)
        hc, _ = score(iv, ds, "rv")
        ci_c, ci_w = boot(hn, w)
        R[k] = dict(
            cov=cov(hn),
            cov_ci=ci_c.tolist(),
            width=float(np.nanmean(w)),
            width_ci=ci_w.tolist(),
            cov_chord=cov(hc),
            cov_mean=float(mean_hits(iv, ds).mean()),
            unbounded=int(sum(x is None for x in iv)),
            widths=w.tolist(),
            hn=hn.tolist(),
            iv=iv,
        )
        r = R[k]
        say(
            f"  {NAMES[k]:34s} {r['cov']*100:8.1f}% [{r['cov_ci'][0]*100:5.1f},{r['cov_ci'][1]*100:5.1f}] {r['width']:7.4f} [{r['width_ci'][0]:.4f},{r['width_ci'][1]:.4f}] "
            f"{r['cov_chord']*100:12.1f}% {r['cov_mean']*100:10.1f}% {r['unbounded']:9d}"
        )
    wE = np.array(R["E"]["widths"])
    say("  paired width, E minus each (bootstrap 95% CI over images):")
    for k in NAMES:
        if k == "E":
            continue
        wk = np.array(R[k]["widths"])
        ok = ~(np.isnan(wE) | np.isnan(wk))
        dl = wE[ok] - wk[ok]
        bt = np.array([dl[BR.integers(0, len(dl), len(dl))].mean() for _ in range(5000)])
        say(f"    E - {k:4s}: {dl.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]")
    sp = lambda d: np.std(d["rx"], ddof=1)
    groups = {
        "spread low": lambda d: sp(d) <= RIGA_CUTS[0],
        "spread mid": lambda d: RIGA_CUTS[0] < sp(d) <= RIGA_CUTS[1],
        "spread high": lambda d: sp(d) > RIGA_CUTS[1],
        "vCDR <0.5": lambda d: np.mean(d["rx"]) < 0.5,
        "vCDR 0.5-0.7": lambda d: 0.5 <= np.mean(d["rx"]) <= 0.7,
        "vCDR >0.7": lambda d: np.mean(d["rx"]) > 0.7,
        "camera Bosch": lambda d: d["src"] == "Bosch",
        "camera Forus": lambda d: d["src"] == "Forus",
        "camera Remidio": lambda d: d["src"] == "Remidio",
    }
    say(f"  conditional individual-rater coverage (spread cut-points from RIGA calib {RIGA_CUTS.round(4).tolist()}):")
    say("    " + f"{'group':16s}{'n':>4s}" + "".join(f"{k:>8s}" for k in NAMES))
    cond = {}
    for g, fn in groups.items():
        idx = [i for i, d in enumerate(ds) if fn(d)]
        if not idx:
            continue
        cond[g] = {"n": len(idx), **{k: cov(np.array(R[k]["hn"])[idx]) for k in NAMES}}
        say("    " + f"{g:16s}{len(idx):4d}" + "".join(f"{cond[g][k]*100:7.1f}%" for k in NAMES))
    A = np.array([iv is None or (iv[0] < 0.6 < iv[1]) for iv in R["E"]["iv"]])
    Bv = np.array([d["rx"].min() < 0.6 <= d["rx"].max() for d in ds])
    po = np.mean(A == Bv)
    pe = A.mean() * Bv.mean() + (1 - A.mean()) * (1 - Bv.mean())
    kp = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    say(
        f"  0.6 crossing (E): interval crosses on {A.sum()}, raters split on {Bv.sum()}, both {np.sum(A & Bv)}, interval only {np.sum(A & ~Bv)}, raters only {np.sum(~A & Bv)}; kappa {kp:.3f}"
    )
    for k in R:
        R[k].pop("iv")
    return R, cond, kp


R0, C0, K0 = report(
    CT,
    f"ZERO-SHOT on Chaksu TEST ({len(CT)} images), RIGA settings unchanged, nominal 90%, individual-rater coverage (extent)",
)
holds = R0["E"]["cov_ci"][0] <= 0.90 <= R0["E"]["cov_ci"][1]
say(
    f"  -> coverage {'HOLDS' if holds else 'DOES NOT HOLD'} for POLARIS (E): 95% CI {np.round(np.array(R0['E']['cov_ci'])*100, 1).tolist()} vs nominal 90%"
)
rx = np.concatenate([d["rx"] for d in CT])
rv = np.concatenate([d["rv"] for d in CT])
df = rv - rx
say(
    f"  chord vs extent on Chaksu raters: r {np.corrcoef(rx, rv)[0, 1]:.4f}, mean diff {df.mean():+.4f}, limits [{df.mean() - 1.96*df.std():+.4f}, {df.mean() + 1.96*df.std():+.4f}]"
)

# secondary: recalibrated on Chaksu TRAIN images
riga_fit = dict(fit)
fit["E"] = fit_tau(CC, "E", LVL, Z90)
fit["B"] = fit_tau(CC, "B", LVL, Z90)
fit["K"] = fit_tau(CC, "K", LVL, Z90)
fit["C"] = fit_C(CC, LVL, Z90)
fit["T"] = coef_tau(CC, LVL, Z90, lambda d: d["e"] + np.diag(d["S"]))
fit["CFh"] = conformal_q(CC, hpt, LVL)
fit["CFu"] = conformal_q(CC, lambda d: d["u_ens"], LVL)
fit["UE"] = float(
    pick(
        UTAU,
        [
            cov(
                score(
                    [
                        (d["u_seeds"].mean() - t * Z90 * d["u_sd"], d["u_seeds"].mean() + t * Z90 * d["u_sd"])
                        for d in CC
                    ],
                    CC,
                )[0]
            )
            for t in UTAU
        ],
        LVL,
    )
)
say(
    "\nRecalibrated fits on Chaksu TRAIN split: "
    + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 5)) for k, v in fit.items()})
)
R1, C1, K1 = report(
    CT, f"SECONDARY: RECALIBRATED on {len(CC)} Chaksu TRAIN images, scored on the same Chaksu TEST images"
)
json.dump(
    {
        "riga_fit": {k: (list(v) if isinstance(v, tuple) else v) for k, v in riga_fit.items()},
        "chaksu_fit": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fit.items()},
        "test_stats": s_test,
        "centre_err": ce.tolist(),
        "zero_shot": {k: {kk: vv for kk, vv in v.items() if kk not in ("widths", "hn")} for k, v in R0.items()},
        "zero_shot_conditional": C0,
        "zero_shot_kappa": K0,
        "holds": holds,
        "recalibrated": {k: {kk: vv for kk, vv in v.items() if kk not in ("widths", "hn")} for k, v in R1.items()},
        "recalibrated_conditional": C1,
    },
    open(os.path.join(OUT, "chaksu_transfer_results.json"), "w"),
    indent=1,
    default=float,
)
say("\nSaved", OUT)
