"""
FINAL RUN. Every number in the paper comes from this script's log and json (outputs/final_run/).
Settings fixed before the run. Nothing is tuned on Chaksu test images.
  1. RIGA de-duplicated: the 8 test images with copies in train/val and the 6 calibration images with copies in train/val are
     removed (lists from outputs/final_checks). Every scalar and quantile is refitted on the cleaned calibration split.
     Configurations: E, B, K, C, T, CFh, CFu, UE (as in calib_v4) plus TK = the final method, fixed before this run:
     per-coefficient temperature on ensemble variance + a CONSTANT rater covariance (no per-image head), exact Fieller.
  2. Segmentation, RIGA (103) and Chaksu (336): harmonic ensemble and the four U-Net ensembles (5 seeds each, largest
     connected component). Disc Dice, cup Dice, vCDR MAE, signed vCDR bias (prediction minus reference), cup outside disc.
     Reference = mask of the mean rater contour (mean of rater radii), vCDR = vertical extent ratio. Holm-corrected paired
     sign-flip tests, harmonic ensemble vs each U-Net, overall and per Chaksu camera.
  3. U-Net drop audit: identical-input check (same .npy array to every model), 10 worst polar U-Net vCDR errors on Remidio
     drawn as overlays (reference white, polar U-Net red, harmonic green), worst-10 lists for every U-Net.
  4. Width-matched controls: fixed-width bands of 0.282, 0.297 and of TK's own Chaksu mean width around the polar U-Net
     vCDR and around the harmonic vCDR, on both datasets.
  5. Paired cluster bootstrap (images resampled, 5000): T and TK vs CFu, E, CFh on coverage and width, overall and per camera.
  6. Signed vCDR bias by reference vCDR level (<0.5, 0.5-0.7, >0.7), both datasets.
  7. Decision conditions printed.
 10/11. Figure 1 (reliability, RIGA and Chaksu side by side: TK, E, CFu, T), Figure 2 (copied off-diagonal figure),
     Table 1 (segmentation) and Table 2 (intervals) written as markdown.
"""

import os, json, shutil

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
cv4 = open("calib_v4.py").read().replace('"outputs", "calib_v4")', '"outputs", "final_run")')
M_FIT = "# ---------------- main fit on CALIB at 90%, single report on TEST"
assert M_FIT in cv4 and "RES, IV = {}, {}" in cv4
exec(cv4.split(M_FIT)[0])
from ultralytics import YOLO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

say("\n==================== FINAL RUN ====================")
J = {}
# ---------------- 1. de-duplication, then refit
FC = json.load(open(os.path.join(BASE, "outputs", "final_checks", "final_checks.json")))
DUP_T = sorted({a for a, *_ in FC["duplicates_test"]})
DUP_C = sorted({a for a, *_ in FC["duplicates_calib"]})
say(f"1. Removed {len(DUP_T)} test images: {DUP_T}\n   Removed {len(DUP_C)} calibration images: {DUP_C}")
n0c, n0t = len(CAL), len(TST)
D["calib"] = [d for d in D["calib"] if d["key"] not in DUP_C]
D["test"] = [d for d in D["test"] if d["key"] not in DUP_T]
CAL, TST = D["calib"], D["test"]
Sconst = np.diag(np.mean([d["v0"] for d in CAL], 0)) + style_cov
sfloor = max(1e-4, 1e-3 * np.median([np.std(d["u_seeds"], ddof=1) for d in CAL]))
for d in CAL + TST:
    d["V"]["K"] = vq(np.diag(d["e"]) + Sconst)
    d["u_sd"] = max(np.std(d["u_seeds"], ddof=1), sfloor)
say(
    f"   calibration {n0c} -> {len(CAL)} images ({sum(len(d['rx']) for d in CAL)} rater pairs); test {n0t} -> {len(TST)} images ({sum(len(d['rx']) for d in TST)} pairs)"
)
exec(cv4.split(M_FIT)[1].split("RES, IV = {}, {}")[0])
fit["TK"] = coef_tau(CAL, LVL, Z90, lambda d: d["e"] + np.diag(Sconst))
NAMES["TK"] = "FINAL: per-coef temp, const rater cov (TK)"


def get_ivs(name, ds, z=Z90):
    if name == "TK":
        return ivs_of(ds, "K", fit["TK"], z)
    if name.startswith("W:"):
        _, base, w = name.split(":")
        w = float(w)
        pt = (lambda d: d["u_ens"]) if base == "CFu" else hpt
        return [(pt(d) - w / 2, pt(d) + w / 2) for d in ds]
    return test_ivs(name, ds, z)


say(
    "   refitted on cleaned CALIB: "
    + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 5)) for k, v in fit.items()})
)
J["fit"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in fit.items()}
J["removed"] = dict(test=DUP_T, calib=DUP_C)

# ---------------- Chaksu test set (same pipeline as chaksu_transfer.py; recalibration split not built)
csrc = open("chaksu_transfer.py").read()
chunk = "RIGA_CUTS = " + csrc.split("RIGA_CUTS = ")[1].split("def report(ds, title):")[0]
assert "CC = build(cal_items)" in chunk
exec(chunk.replace("CC = build(cal_items)", "CC = []"))
CT_ = CT
say(f"   Chaksu test images in use: {len(CT_)} ({sum(len(d['rx']) for d in CT_)} rater pairs)")

# ---------------- 2. segmentation on both datasets
UN = {}
for m in ["plain", "attention", "deepres", "polar"]:
    ms = []
    for k in range(1, 6):
        mm = CL[m]().to(DEVICE)
        mm.load_state_dict(
            torch.load(
                os.path.join(BASE, "outputs", "phase8_v2", m, f"seed{k}_best.pt"),
                map_location=DEVICE,
                weights_only=True,
            )
        )
        mm.eval()
        ms.append(mm)
    UN[m] = ms


@torch.no_grad()
def unet_masks(m, imgs):
    out = []
    for im in imgs:
        x = np.stack([to_polar(np.ascontiguousarray(im[:, :, c])) for c in range(3)], -1) if m == "polar" else im
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)[None].to(DEVICE)
        p = np.mean([torch.sigmoid(mm(t).float()).cpu().numpy()[0] for mm in UN[m]], 0)
        if m == "polar":
            p = np.stack([to_cartesian(np.ascontiguousarray(p[c])) for c in range(2)])
        out.append((lcc(p[0] > 0.5), lcc(p[1] > 0.5)))
    return out


@torch.no_grad()
def harm_masks(imgs):
    out = []
    for im in imgs:
        x = torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32)).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, size=256, mode="bilinear", align_corners=False).to(DEVICE)
        u = np.mean([mm(x).cpu().numpy()[0] * s_ + t_ for mm, t_, s_ in members], 0)
        out.append((coeffs_to_mask(u[:25], Phi) > 0, coeffs_to_mask(u[25:], Phi) > 0))
    return out


MODELS = ["harmonic_ens", "plain_ens", "attention_ens", "deepres_ens", "polar_ens"]
RS = [s for s in test_samples if "RIGA/" + s["case"] not in DUP_T]
# Reference = mask of the mean VALIDATED rater contour (same definition as Chaksu), on the 94 test images with >= 2 valid raters.
# Diagnostic: compare with the older fused reference masks (eval_testset_all_models.py), which use a weaker rater filter.
old_ref = {
    s["case"]: vext(
        s["disc_mask"] > 0, (s["cup_mask"] > 0) if s["cup_mask"] is not None else np.zeros_like(s["disc_mask"] > 0)
    )
    for s in RS
}
riga_in = [
    (
        npy(d["key"]),
        d["key"].split("/", 1)[1],
        coeffs_to_mask(d["ubar"][:25], Phi) > 0,
        coeffs_to_mask(d["ubar"][25:], Phi) > 0,
        "RIGA",
    )
    for d in TST
]
dif = [
    (c, old_ref[c], vext(gd, gc))
    for _, c, gd, gc, _ in riga_in
    if c in old_ref and abs(old_ref[c] - vext(gd, gc)) > 0.1
]
say(
    f"   RIGA reference check: {len(dif)} of {len(riga_in)} images differ by > 0.1 in reference vCDR between the old fused masks and the validated rater mean: "
    + "; ".join(f"{c} old {o:.2f} new {n:.2f}" for c, o, n in dif)
)
ch_in = [
    (
        os.path.join(CH_DIR, d["key"].split("/", 1)[1] + "_normalized.npy"),
        d["key"].split("/", 1)[1],
        coeffs_to_mask(d["ubar"][:25], Phi) > 0,
        coeffs_to_mask(d["ubar"][25:], Phi) > 0,
        d["src"],
    )
    for d in CT_
]
SEG, PRED = {}, {}
for dname, items in [("RIGA", riga_in), ("Chaksu", ch_in)]:
    imgs = [np.load(p).astype(np.float32) for p, *_ in items]
    say(
        f"\n2. SEGMENTATION {dname}: {len(items)} images; input check: every model reads the same .npy array; "
        f"input mean {np.mean([im.mean() for im in imgs]):+.3f}, sd {np.mean([im.std() for im in imgs]):.3f} (z-scored per image and channel)"
    )
    P = {"harmonic_ens": harm_masks(imgs)}
    for m in ["plain", "attention", "deepres", "polar"]:
        P[m + "_ens"] = unet_masks(m, imgs)
    rows = {mn: [] for mn in MODELS}
    for i, (_, case, gd, gc, grp) in enumerate(items):
        gv = vext(gd, gc)
        for mn in MODELS:
            d_, c_ = P[mn][i]
            pv = vext(d_, c_)
            out = np.logical_and(c_, ~d_).sum()
            rows[mn].append(
                dict(
                    case=case,
                    group=grp,
                    disc=float(dice(d_, gd)),
                    cup=float(dice(c_, gc)),
                    v_pred=float(pv),
                    v_ref=float(gv),
                    vcdr_ae=float(abs(pv - gv)),
                    bias=float(pv - gv),
                    out_any=float(out > 0),
                    out_1=float(out / max(c_.sum(), 1) > 0.01),
                )
            )
    SEG[dname], PRED[dname] = rows, (P, items, imgs)


def holm_tests(rows, idx, metrics=("disc", "cup", "vcdr_ae")):
    rng_ = np.random.default_rng(2026)
    res = {}
    for met in metrics:
        R_ = []
        for u in MODELS[1:]:
            d = np.array([rows[u][i][met] - rows["harmonic_ens"][i][met] for i in idx])
            null = np.abs((rng_.choice([-1.0, 1.0], (10000, len(d))) * d).mean(1))
            p = (np.sum(null >= abs(d.mean())) + 1) / 10001
            R_.append(dict(u=u, diff=float(d.mean()), p=float(p)))
        order = np.argsort([r["p"] for r in R_])
        run = 0.0
        for rk, i in enumerate(order):
            run = max(run, min(1.0, (len(R_) - rk) * R_[i]["p"]))
            R_[i]["p_holm"] = run
        res[met] = R_
    return res


TAB1 = {}
for dname in ["RIGA", "Chaksu"]:
    rows = SEG[dname]
    groups = {"all": list(range(len(rows["harmonic_ens"])))}
    if dname == "Chaksu":
        for cam in ["Bosch", "Forus", "Remidio"]:
            groups[cam] = [i for i, r in enumerate(rows["harmonic_ens"]) if r["group"] == cam]
    for g, idx in groups.items():
        say(f"   {dname} / {g} (n={len(idx)}):  model  disc  cup  vCDR MAE  bias  cup-outside(any / >1%)")
        tests = holm_tests(rows, idx)
        TAB1[(dname, g)] = {}
        for mn in MODELS:
            f = lambda k: float(np.mean([rows[mn][i][k] for i in idx]))
            TAB1[(dname, g)][mn] = dict(
                n=len(idx),
                disc=f("disc"),
                cup=f("cup"),
                mae=f("vcdr_ae"),
                bias=f("bias"),
                out_any=f("out_any"),
                out_1=f("out_1"),
            )
            t = TAB1[(dname, g)][mn]
            say(
                f"     {mn:14s} {t['disc']*100:6.2f} {t['cup']*100:6.2f} {t['mae']:8.4f} {t['bias']:+8.4f}   {t['out_any']*100:4.1f}% / {t['out_1']*100:4.1f}%"
            )
        for met in ["disc", "cup", "vcdr_ae"]:
            say(
                f"     {met:8s} U-Net minus harmonic: "
                + "; ".join(
                    f"{r['u']} {r['diff']*(100 if met != 'vcdr_ae' else 1):+.{2 if met != 'vcdr_ae' else 4}f} (Holm p {r['p_holm']:.3f}{'*' if r['p_holm'] < 0.05 else ''})"
                    for r in tests[met]
                )
            )
        TAB1[(dname, g)]["tests"] = tests
J["table1"] = {f"{a}|{b}": v for (a, b), v in TAB1.items()}
J["seg_per_image"] = SEG

# ---------------- 3. U-Net drop audit
say("\n3. U-NET DROP AUDIT (Chaksu Remidio)")
P, items, imgs = PRED["Chaksu"]
rows = SEG["Chaksu"]
rem = [i for i, r in enumerate(rows["polar_ens"]) if r["group"] == "Remidio"]
for mn in MODELS[1:]:
    w10 = sorted(rem, key=lambda i: -rows[mn][i]["vcdr_ae"])[:10]
    say(
        f"   {mn:14s} worst 10 vCDR errors: "
        + ", ".join(
            f"{rows[mn][i]['case']} ({rows[mn][i]['bias']:+.2f}; harmonic {rows['harmonic_ens'][i]['bias']:+.2f})"
            for i in w10
        )
    )
odir = os.path.join(OUT, "worst_polar_unet_remidio")
os.makedirs(odir, exist_ok=True)
for rank, i in enumerate(sorted(rem, key=lambda i: -rows["polar_ens"][i]["vcdr_ae"])[:10]):
    im = imgs[i]
    v = np.stack([(im[:, :, c] - im[:, :, c].min()) / (np.ptp(im[:, :, c]) + 1e-6) * 255 for c in range(3)], -1).astype(
        np.uint8
    )
    v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
    _, _, gd, gc, _ = items[i]
    for mask, col in [
        (gd, (255, 255, 255)),
        (gc, (255, 255, 255)),
        (P["polar_ens"][i][0], (0, 0, 255)),
        (P["polar_ens"][i][1], (0, 0, 255)),
        (P["harmonic_ens"][i][0], (0, 255, 0)),
        (P["harmonic_ens"][i][1], (0, 255, 0)),
    ]:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cs, -1, col, 1)
    r = rows["polar_ens"][i]
    cv2.imwrite(
        os.path.join(
            odir,
            f"{rank + 1:02d}_{r['case']}_ref{r['v_ref']:.2f}_unet{r['v_pred']:.2f}_harm{rows['harmonic_ens'][i]['v_pred']:.2f}.png",
        ),
        v,
    )
say(f"   overlays: {odir}  (reference white, polar U-Net red, harmonic green; crops contain no patient text)")
say(
    f"   mean input statistics, RIGA vs Chaksu crops: mean {np.mean([x.mean() for x in PRED['RIGA'][2]]):+.3f} vs {np.mean([x.mean() for x in imgs]):+.3f}; "
    f"per-channel sd {np.mean([x.std(axis=(0, 1)).mean() for x in PRED['RIGA'][2]]):.3f} vs {np.mean([x.std(axis=(0, 1)).mean() for x in imgs]):.3f}"
)

# ---------------- 4/5. intervals on both datasets, width-matched controls, paired tests
CONF = ["TK", "E", "T", "K", "B", "C", "CFh", "CFu", "UE"]
tk_w = float(np.nanmean(score(get_ivs("TK", CT_), CT_)[1]))
CTRL = [f"W:{b}:{w}" for b in ("CFu", "CFh") for w in ("0.282", "0.297", f"{tk_w:.4f}")]
RES2 = {}
for dname, ds in [("RIGA", TST), ("Chaksu", CT_)]:
    say(
        f"\n4. INTERVALS {dname} ({len(ds)} images, {sum(len(d['rx']) for d in ds)} rater pairs), nominal 90%, individual-rater coverage"
    )
    RES2[dname] = {}
    for k in CONF + CTRL:
        iv = get_ivs(k, ds)
        hn, w = score(iv, ds)
        ci_c, ci_w = boot(hn, w)
        RES2[dname][k] = dict(
            hn=hn,
            w=w,
            cov=cov(hn),
            cov_ci=ci_c.tolist(),
            width=float(np.nanmean(w)),
            width_ci=ci_w.tolist(),
            unb=int(sum(x is None for x in iv)),
            iv=iv,
        )
        nm = (
            NAMES[k]
            if k in NAMES
            else f"fixed width {k.split(':')[2]} around {'polar U-Net' if 'CFu' in k else 'harmonic'} vCDR"
        )
        r = RES2[dname][k]
        say(
            f"   {nm:46s} {r['cov']*100:5.1f}% [{r['cov_ci'][0]*100:5.1f}, {r['cov_ci'][1]*100:5.1f}]  width {r['width']:.4f} [{r['width_ci'][0]:.4f}, {r['width_ci'][1]:.4f}]"
            + (f"  unbounded {r['unb']}" if r['unb'] else "")
        )
    if dname == "Chaksu":
        for cam in ["Bosch", "Forus", "Remidio"]:
            idx = [i for i, d in enumerate(ds) if d["src"] == cam]
            say(
                f"   {cam:8s} (n={len(idx)}): "
                + ", ".join(f"{k} {cov(RES2[dname][k]['hn'][idx])*100:.1f}%" for k in CONF + CTRL[:3])
            )


def paired(dname, a, b, idx, B=5000):
    A_, B_ = RES2[dname][a], RES2[dname][b]
    rng_ = np.random.default_rng(99)
    idx = np.array(idx)
    I = idx[rng_.integers(0, len(idx), (B, len(idx)))]
    dc = A_["hn"][I, 0].sum(1) / A_["hn"][I, 1].sum(1) - B_["hn"][I, 0].sum(1) / B_["hn"][I, 1].sum(1)
    wd = np.array([np.nanmean(A_["w"][i] - B_["w"][i]) for i in I])
    pv = lambda x: float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))
    return dict(
        dcov=float(cov(A_["hn"][idx]) - cov(B_["hn"][idx])),
        dcov_ci=np.percentile(dc, [2.5, 97.5]).tolist(),
        p_cov=pv(dc),
        dw=float(np.nanmean(A_["w"][idx] - B_["w"][idx])),
        dw_ci=np.percentile(wd, [2.5, 97.5]).tolist(),
        p_w=pv(wd),
    )


say("\n5. PAIRED CLUSTER BOOTSTRAP (A minus B; coverage in points, width in vCDR units)")
PT = {}
for dname, ds in [("RIGA", TST), ("Chaksu", CT_)]:
    groups = {"all": list(range(len(ds)))}
    if dname == "Chaksu":
        for cam in ["Bosch", "Forus", "Remidio"]:
            groups[cam] = [i for i, d in enumerate(ds) if d["src"] == cam]
    for g, idx in groups.items():
        for a in ["TK", "T"]:
            for b in ["CFu", "E", "CFh"]:
                r = paired(dname, a, b, idx)
                PT[f"{dname}|{g}|{a}-{b}"] = r
                say(
                    f"   {dname:6s} {g:7s} {a:2s} - {b:3s}: coverage {r['dcov']*100:+5.1f} [{r['dcov_ci'][0]*100:+5.1f}, {r['dcov_ci'][1]*100:+5.1f}] p={r['p_cov']:.4f}; "
                    f"width {r['dw']:+.4f} [{r['dw_ci'][0]:+.4f}, {r['dw_ci'][1]:+.4f}] p={r['p_w']:.4f}"
                )
J["paired"] = PT
J["intervals"] = {
    dn: {k: {kk: vv for kk, vv in v.items() if kk not in ("hn", "w", "iv")} for k, v in R.items()}
    for dn, R in RES2.items()
}

# ---------------- 6. signed bias by vCDR level
say("\n6. SIGNED vCDR BIAS (prediction minus reference) BY REFERENCE vCDR LEVEL")
BIAS = {}
for dname in ["RIGA", "Chaksu"]:
    rows = SEG[dname]
    for lo, hi, lab in [(0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 9, ">0.7")]:
        idx = [i for i, r in enumerate(rows["harmonic_ens"]) if lo <= r["v_ref"] < hi]
        if not idx:
            say(f"   {dname:6s} {lab:8s} n=0")
            continue
        BIAS[f"{dname}|{lab}"] = {mn: float(np.mean([rows[mn][i]["bias"] for i in idx])) for mn in MODELS}
        say(
            f"   {dname:6s} {lab:8s} n={len(idx):3d}: "
            + ", ".join(f"{mn} {BIAS[f'{dname}|{lab}'][mn]:+.3f}" for mn in MODELS)
        )
J["bias"] = BIAS

# ---------------- reliability + figures
say("\nRELIABILITY (each method refitted on cleaned RIGA calibration at every level)")
LEVELS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
REL = {dn: {k: [] for k in ["TK", "E", "CFu", "T"]} for dn in ["RIGA", "Chaksu"]}
for lv in LEVELS:
    z = _N.ppf(0.5 + lv / 2)
    tE = fit_tau(CAL, "E", lv, z)
    tT = coef_tau(CAL, lv, z, lambda d: d["e"] + np.diag(d["S"]))
    tK = coef_tau(CAL, lv, z, lambda d: d["e"] + np.diag(Sconst))
    q = conformal_q(CAL, lambda d: d["u_ens"], lv)
    for dn, ds in [("RIGA", TST), ("Chaksu", CT_)]:
        REL[dn]["E"].append(cov(score(ivs_of(ds, "E", tE, z), ds)[0]))
        REL[dn]["T"].append(cov(score(ivs_of(ds, "E", tT, z), ds)[0]))
        REL[dn]["TK"].append(cov(score(ivs_of(ds, "K", tK, z), ds)[0]))
        REL[dn]["CFu"].append(cov(score([(d["u_ens"] - q, d["u_ens"] + q) for d in ds], ds)[0]))
for dn in REL:
    say(f"   {dn}: nominal " + " ".join(f"{l*100:4.0f}" for l in LEVELS))
    for k, v in REL[dn].items():
        say(f"     {k:4s} " + " ".join(f"{x*100:4.1f}" for x in v))
J["reliability"] = {"levels": LEVELS, **REL}
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
for a, dn in zip(axs, ["RIGA", "Chaksu"]):
    L = np.array(LEVELS) * 100
    a.plot([45, 100], [45, 100], color="0.6", lw=0.8, ls=(0, (4, 3)), label="ideal")
    a.plot(L, np.array(REL[dn]["TK"]) * 100, "k-o", ms=4, lw=1.2, label="per-coefficient temperature (final)")
    a.plot(
        L, np.array(REL[dn]["E"]) * 100, "k--s", ms=4, lw=1.0, mfc="white", label="vCDR-level scalar (pre-specified)"
    )
    a.plot(L, np.array(REL[dn]["CFu"]) * 100, "k:^", ms=4, lw=1.0, label="conformal, U-Net vCDR")
    a.set_xlim(45, 100)
    a.set_ylim(35, 100)
    a.set_xlabel("Nominal coverage (%)")
    a.grid(color="0.9", lw=0.5)
    a.set_title(f"{'RIGA test' if dn == 'RIGA' else 'Chákṣu test (zero-shot)'}")
axs[0].set_ylabel("Individual-rater coverage (%)")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
FD = os.path.join(OUT, "figures")
os.makedirs(FD, exist_ok=True)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(FD, f"fig1_reliability_riga_chaksu.{e}"), dpi=300, bbox_inches="tight")
    src_ = os.path.join(BASE, "outputs", "figures", f"fig1_rater_deviation_correlation.{e}")
    if os.path.exists(src_):
        shutil.copy(src_, os.path.join(FD, f"fig2_rater_deviation_correlation.{e}"))
plt.close(fig)

# ---------------- tables
T1 = [
    "| Dataset | Group | Model | Disc Dice | Cup Dice | vCDR MAE | vCDR bias | Cup outside disc |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for (dn, g), v in TAB1.items():
    for mn in MODELS:
        t = v[mn]
        T1.append(
            f"| {dn} | {g} (n={t['n']}) | {mn} | {t['disc']*100:.2f} | {t['cup']*100:.2f} | {t['mae']:.4f} | {t['bias']:+.4f} | {t['out_any']*100:.1f}% |"
        )
T2 = [
    "| Configuration | RIGA coverage [95% CI] | RIGA width | Chákṣu coverage [95% CI] | Chákṣu width |",
    "| --- | --- | --- | --- | --- |",
]
for k in CONF + CTRL:
    nm = (
        NAMES[k] if k in NAMES else f"Fixed width {k.split(':')[2]}, {'polar U-Net' if 'CFu' in k else 'harmonic'} vCDR"
    )
    a, b = RES2["RIGA"][k], RES2["Chaksu"][k]
    T2.append(
        f"| {nm} | {a['cov']*100:.1f}% [{a['cov_ci'][0]*100:.1f}, {a['cov_ci'][1]*100:.1f}] | {a['width']:.3f} | {b['cov']*100:.1f}% [{b['cov_ci'][0]*100:.1f}, {b['cov_ci'][1]*100:.1f}] | {b['width']:.3f} |"
    )
open(os.path.join(OUT, "table1_segmentation.md"), "w").write("\n".join(T1))
open(os.path.join(OUT, "table2_intervals.md"), "w").write("\n".join(T2))

# ---------------- 7. decision
say("\n7. DECISION CONDITIONS")
ch = TAB1[("Chaksu", "all")]
c2 = all(r["diff"] >= 0 or r["p_holm"] >= 0.05 for r in ch["tests"]["vcdr_ae"])
say(
    f"   (2) Chaksu vCDR MAE, harmonic ensemble {ch['harmonic_ens']['mae']:.4f} vs U-Nets "
    + ", ".join(f"{m} {ch[m]['mae']:.4f}" for m in MODELS[1:])
    + f" -> harmonic at least as good (better, or not significantly worse after Holm): {'YES' if c2 else 'NO'}"
)
for a in ["TK", "T"]:
    r = PT[f"Chaksu|all|{a}-CFu"]
    say(
        f"   (3) {a} minus CFu coverage on Chaksu: {r['dcov']*100:+.1f} points [{r['dcov_ci'][0]*100:+.1f}, {r['dcov_ci'][1]*100:+.1f}], p={r['p_cov']:.4f} -> "
        f"{'YES' if r['dcov_ci'][0] > 0 else 'NO'}"
    )
say(f"   (1) U-Net drop survives the audit: decided by inspecting the overlays in {odir}")
json.dump(
    J,
    open(os.path.join(OUT, "final_run_results.json"), "w"),
    indent=1,
    default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o),
)
say("\nSaved", OUT)
