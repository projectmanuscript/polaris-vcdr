"""
1. Evaluates the pretrained, augmented ResNet-34 U-Net (outputs/phase8_rn34, used when >= 3 seeds exist) exactly like the
   other U-Nets: RIGA test and Chaksu segmentation (Table 1 row), its fixed-width conformal band fitted on the same RIGA
   calibration images (Table 2 row), width-matched bands, paired tests, Holm family extended to five U-Nets.
2. Distance from nominal 90% for every Table 2 configuration.
3. Figure 2(a): Chaksu reliability, POLARIS against conformal polar U-Net, with bootstrap band (vector PDF).
4. Figure 2(b-e): four example crops (lowest / highest expert spread, RIGA and Chaksu), colour crops, 600 dpi PNG.
5. Figure 1 assets in the same colour code (fundus + box, crop, coefficient bars, output, expert outlines, r(theta),
   target vector, the two correlation heatmaps).
Re-executes final_run.py in memory (outputs to outputs/figs_v2/); nothing is refitted beyond final_run.
Run:  python3 eval_rn34_and_figs.py 2>&1 | tee outputs/figs_v2_console.log        (CASE=RIGA/<image> picks the Fig. 1 image)
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
exec(fr.replace(a_, "'\"outputs\", \"figs_v2\")'"))
assert OUT.endswith("figs_v2"), OUT
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

BLUE, GREEN, VERM, ORANGE = "#0072B2", "#009E73", "#D55E00", "#E69F00"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)
say(
    f"\n==================== FIGS v2 ====================\nfont in use: {matplotlib.font_manager.findfont(plt.rcParams['font.sans-serif'][0])}"
)
FD = os.path.join(OUT, "figures_v2")
os.makedirs(FD, exist_ok=True)
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)
CM = 1 / 2.54
sp = lambda d: np.std(d["rx"], ddof=1)
DS_IDX = {
    "RIGA": {"all": list(range(len(TST)))},
    "Chaksu": {
        "all": list(range(len(CT_))),
        **{c: [i for i, d in enumerate(CT_) if d["src"] == c] for c in ["Bosch", "Forus", "Remidio"]},
    },
}
X = {}

# ---------------- 1. pretrained ResNet-34 U-Net
from rn34_unet import ResUNet34

RNDIR = os.path.join(BASE, "outputs", "phase8_rn34")
RSEEDS = [k for k in range(1, 6) if os.path.exists(os.path.join(RNDIR, f"seed{k}_best.pt"))]
RN = len(RSEEDS) >= 3 and os.environ.get("RN", "1") != "0"
say(
    f"\n1. PRETRAINED RESNET-34 U-NET: {'seeds ' + str(RSEEDS) if RN else 'not evaluated (needs >= 3 seeds in ' + RNDIR + ')'}"
)
if RN:
    RM = []
    for k in RSEEDS:
        m_ = ResUNet34().to(DEVICE)
        m_.load_state_dict(torch.load(os.path.join(RNDIR, f"seed{k}_best.pt"), map_location=DEVICE, weights_only=True))
        m_.eval()
        RM.append(m_)

    @torch.no_grad()
    def rn_masks(imgs, bs=8):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(np.stack(imgs[i : i + bs]).astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
            p = np.mean([torch.sigmoid(m_(x).float()).cpu().numpy() for m_ in RM], 0)
            out += [(lcc(q[0] > 0.5), lcc(q[1] > 0.5)) for q in p]
        return out

    for dname in ["RIGA", "Chaksu"]:
        P, items, imgs = PRED[dname]
        P["rn34_ens"] = rn_masks(imgs)
        rows = []
        for i, (_, case, gd, gc, grp) in enumerate(items):
            gv = vext(gd, gc)
            d_, c_ = P["rn34_ens"][i]
            pv = vext(d_, c_)
            o = np.logical_and(c_, ~d_).sum()
            rows.append(
                dict(
                    case=case,
                    group=grp,
                    disc=float(dice(d_, gd)),
                    cup=float(dice(c_, gc)),
                    v_pred=float(pv),
                    v_ref=float(gv),
                    vcdr_ae=float(abs(pv - gv)),
                    bias=float(pv - gv),
                    out_any=float(o > 0),
                    out_1=float(o / max(c_.sum(), 1) > 0.01),
                )
            )
        SEG[dname]["rn34_ens"] = rows
    MODELS.append("rn34_ens")
    X["table1"] = {}
    for dname in ["RIGA", "Chaksu"]:
        rows = SEG[dname]
        groups = {"all": list(range(len(rows["harmonic_ens"])))}
        if dname == "Chaksu":
            groups.update(
                {
                    c: [i for i, r in enumerate(rows["harmonic_ens"]) if r["group"] == c]
                    for c in ["Bosch", "Forus", "Remidio"]
                }
            )
        for g, idx in groups.items():
            f = lambda mn, k: float(np.mean([rows[mn][i][k] for i in idx]))
            T = holm_tests(rows, idx)  # harmonic vs each of the five U-Nets, Holm over five
            X["table1"][f"{dname}|{g}"] = {
                mn: dict(disc=f(mn, "disc"), cup=f(mn, "cup"), mae=f(mn, "vcdr_ae"), bias=f(mn, "bias"))
                for mn in MODELS
            }
            say(
                f"   {dname} / {g} (n={len(idx)}): rn34 U-Net disc {f('rn34_ens', 'disc')*100:.2f}, cup {f('rn34_ens', 'cup')*100:.2f}, vCDR MAE {f('rn34_ens', 'vcdr_ae'):.4f}, "
                f"bias {f('rn34_ens', 'bias'):+.4f}, cup outside disc {f('rn34_ens', 'out_any')*100:.1f}%   (POLARIS {f('harmonic_ens', 'disc')*100:.2f} / {f('harmonic_ens', 'cup')*100:.2f} / {f('harmonic_ens', 'vcdr_ae'):.4f})"
            )
            for met in ["disc", "cup", "vcdr_ae"]:
                say(
                    f"      {met:8s} U-Net minus POLARIS, Holm over 5 U-Nets: "
                    + "; ".join(
                        f"{r['u'].replace('_ens', '')} {r['diff']*(100 if met != 'vcdr_ae' else 1):+.{2 if met != 'vcdr_ae' else 4}f} (p {r['p_holm']:.3f}{'*' if r['p_holm'] < 0.05 else ''})"
                        for r in T[met]
                    )
                )

    # vCDR of the rn34 ensemble on calibration, RIGA test and Chaksu test; conformal band fitted on RIGA calibration
    def rn_vcdr(paths):
        res = []
        for i in range(0, len(paths), 16):
            res += [vext(*m) for m in rn_masks([np.load(p).astype(np.float32) for p in paths[i : i + 16]])]
        return res

    for ds_, getp in [
        (CAL + TST, lambda d: npy(d["key"])),
        (CT_, lambda d: os.path.join(CH_DIR, d["key"].split("/", 1)[1] + "_normalized.npy")),
    ]:
        for d, v in zip(ds_, rn_vcdr([getp(d) for d in ds_])):
            d["r_ens"] = v
    fit["CFr"] = conformal_q(CAL, lambda d: d["r_ens"], LVL)
    NAMES["CFr"] = "conformal, pretrained ResNet-34 U-Net (CFr)"
    say(
        f"   CFr half-width (RIGA calibration): {fit['CFr']:.5f}; empty-disc predictions: {sum(np.isnan(d['r_ens']) for d in CAL + TST + CT_)}"
    )

    def gi(name, ds):
        if name == "CFr":
            return [(d["r_ens"] - fit["CFr"], d["r_ens"] + fit["CFr"]) for d in ds]
        if name.startswith("W:CFr:"):
            w = float(name.split(":")[2])
            return [(d["r_ens"] - w / 2, d["r_ens"] + w / 2) for d in ds]
        return get_ivs(name, ds)

    RW = ["CFr", "W:CFr:0.282", "W:CFr:0.297", f"W:CFr:{tk_w:.4f}"]
    for dname, ds in [("RIGA", TST), ("Chaksu", CT_)]:
        for k in RW:
            iv = gi(k, ds)
            hn, w = score(iv, ds)
            c_, w_ = boot(hn, w)
            RES2[dname][k] = dict(
                hn=hn,
                w=w,
                cov=cov(hn),
                cov_ci=c_.tolist(),
                width=float(np.nanmean(w)),
                width_ci=w_.tolist(),
                unb=0,
                iv=iv,
            )
            say(
                f"   {dname:6s} {k:14s} coverage {cov(hn)*100:5.1f}% [{c_[0]*100:.1f}, {c_[1]*100:.1f}]  width {np.nanmean(w):.4f}"
            )
    say("   paired cluster bootstrap (A minus B):")
    X["paired"] = {}
    for dname in ["Chaksu", "RIGA"]:
        for g, idx in DS_IDX[dname].items():
            for a, b in [
                ("TK", "CFr"),
                ("CFr", "CFh"),
                ("CFr", "CFu"),
                ("W:CFh:0.282", "W:CFr:0.282"),
                ("W:CFh:0.297", "W:CFr:0.297"),
            ]:
                r = paired(dname, a, b, idx)
                X["paired"][f"{dname}|{g}|{a}-{b}"] = r
                say(
                    f"   {dname:6s} {g:8s} {a:12s} - {b:12s}: coverage {r['dcov']*100:+5.1f} [{r['dcov_ci'][0]*100:+.1f}, {r['dcov_ci'][1]*100:+.1f}] p={r['p_cov']:.4f}; "
                    f"width {r['dw']:+.4f} p={r['p_w']:.4f}"
                )
    rel = []
    for lv in LEVELS:
        q_ = conformal_q(CAL, lambda d: d["r_ens"], lv)
        rel.append(cov(score([(d["r_ens"] - q_, d["r_ens"] + q_) for d in CT_], CT_)[0]))
    say(
        "   CFr reliability on Chaksu, nominal "
        + " ".join(f"{l*100:.0f}" for l in LEVELS)
        + ": "
        + " ".join(f"{x*100:.1f}" for x in rel)
    )
    X["rn34_rel_chaksu"] = rel
    t1 = X["table1"]
    say(
        f"\n   TABLE 1 ROW  | U-Net, ImageNet ResNet-34, aug. | {t1['RIGA|all']['rn34_ens']['disc']*100:.2f} | {t1['RIGA|all']['rn34_ens']['cup']*100:.2f} | "
        f"{t1['RIGA|all']['rn34_ens']['mae']:.3f} | {t1['Chaksu|all']['rn34_ens']['disc']*100:.2f} | {t1['Chaksu|all']['rn34_ens']['cup']*100:.2f} | {t1['Chaksu|all']['rn34_ens']['mae']:.3f} |"
    )
    a, b = RES2["RIGA"]["CFr"], RES2["Chaksu"]["CFr"]
    say(
        f"   TABLE 2 ROW  | Conformal, ResNet-34 U-Net | {a['cov']*100:.1f} | {a['width']:.3f} | {b['cov']*100:.1f} | [{b['cov_ci'][0]*100:.1f}, {b['cov_ci'][1]*100:.1f}] | {b['width']:.3f} |"
    )

# ---------------- 2. distance from nominal 90%
say("\n2. DISTANCE FROM NOMINAL 90% (|coverage - 90|, points) and whether the Chaksu 95% CI contains 90%")
for k in ["CFu", "UE", "CFh"] + (["CFr"] if RN else []) + ["B", "T", "E", "K", "C", "TK"]:
    a, b = RES2["RIGA"][k], RES2["Chaksu"][k]
    say(
        f"   {k:4s} RIGA {abs(a['cov'] - 0.9)*100:4.1f}   Chaksu {abs(b['cov'] - 0.9)*100:4.1f}   Chaksu CI [{b['cov_ci'][0]*100:.1f}, {b['cov_ci'][1]*100:.1f}] "
        f"contains 90%: {'yes' if b['cov_ci'][0] <= 0.9 <= b['cov_ci'][1] else 'no'}"
    )

# ---------------- 3. Figure 2(a): Chaksu reliability
TKc, CFc = np.array(REL["Chaksu"]["TK"]), np.array(REL["Chaksu"]["CFu"])
L = np.array(LEVELS) * 100
lo_, hi_ = [], []
for i, lv in enumerate(LEVELS):
    z = _N.ppf(0.5 + lv / 2)
    tK = coef_tau(CAL, lv, z, lambda d: d["e"] + np.diag(Sconst))
    hn = score(ivs_of(CT_, "K", tK, z), CT_)[0]
    assert abs(cov(hn) - TKc[i]) < 1e-12
    I = np.random.default_rng(7).integers(0, len(hn), (5000, len(hn)))
    cs = hn[I, 0].sum(1) / hn[I, 1].sum(1)
    lo_.append(np.percentile(cs, 2.5) * 100)
    hi_.append(np.percentile(cs, 97.5) * 100)
fig, a = plt.subplots(figsize=(2.2, 1.4))
a.axvline(90, color="0.85", lw=0.6, zorder=0)
a.plot([45, 100], [45, 100], color="0.55", lw=0.8, ls=(0, (3, 2)), label="ideal", zorder=1)
a.fill_between(L, lo_, hi_, color=BLUE, alpha=0.18, lw=0, zorder=2)
a.plot(L, CFc * 100, ls=(0, (4, 1.5)), marker="s", ms=2.6, color=ORANGE, lw=1.2, label="Conformal U-Net", zorder=3)
a.plot(L, TKc * 100, ls="-", marker="o", ms=2.8, color=BLUE, lw=1.2, label="POLARIS", zorder=4)
i9 = LEVELS.index(0.9)
a.annotate(
    f"{TKc[i9]*100:.1f}%",
    (90, TKc[i9] * 100),
    xytext=(-3, 3),
    textcoords="offset points",
    ha="right",
    va="bottom",
    color=BLUE,
)
a.annotate(
    f"{CFc[i9]*100:.1f}%",
    (90, CFc[i9] * 100),
    xytext=(3, -2),
    textcoords="offset points",
    ha="left",
    va="top",
    color=ORANGE,
)
a.set_xlim(47, 97)
a.set_ylim(30, 100)
a.set_xticks([50, 60, 70, 80, 90])
a.set_yticks([40, 60, 80, 100])
a.set_xlabel("Nominal coverage (%)", labelpad=1)
a.set_ylabel("Expert coverage (%)", labelpad=1)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
fig.canvas.draw()
p0, p1 = a.transData.transform([(55, 55), (65, 65)])
a.text(
    57,
    59.5,
    "ideal",
    color="0.45",
    rotation=np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])),
    rotation_mode="anchor",
    ha="left",
    va="bottom",
)
h_, l_ = a.get_legend_handles_labels()
keep = [i for i, t in enumerate(l_) if t != "ideal"]
a.legend(
    [h_[i] for i in keep][::-1],
    [l_[i] for i in keep][::-1],
    loc="lower right",
    frameon=False,
    handlelength=1.6,
    borderaxespad=0.0,
    labelspacing=0.2,
)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(FD, f"fig2a_reliability_chaksu.{e}"), dpi=600, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
X["fig2a"] = dict(
    levels=LEVELS, polaris=TKc.tolist(), conformal_unet=CFc.tolist(), polaris_ci_lo=lo_, polaris_ci_hi=hi_
)
say(
    f"\n3. FIG 2(a) saved; POLARIS 90% point {TKc[i9]*100:.1f} [{lo_[i9]:.1f}, {hi_[i9]:.1f}], conformal U-Net {CFc[i9]*100:.1f}"
)


# ---------------- shared drawing helpers
def xy(u25, f=1.0):
    r = (Phi @ u25) * f
    x, y = 256 + r * np.cos(GRID), 256 + r * np.sin(GRID)
    return np.append(x, x[0]), np.append(y, y[0])


def rgb_crop(d):
    ds_, case = d["key"].split("/", 1)
    if ds_ == "RIGA":
        return cv2.cvtColor(cv2.imread(os.path.join(PREPROC_DIR, ds_, case + "_viewable.jpg")), cv2.COLOR_BGR2RGB)
    _, cam, k = case.split("_", 2)
    img = cv2.imread(MAN["Test"][cam][k]["image"])
    r = yolo(img, conf=0.01, verbose=False)[0]
    x, y, w, h = r.boxes[0].xywh[0].tolist()
    nrm, view = crop_and_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), x, y, max(w, h))
    dev = float(np.abs(nrm - np.load(os.path.join(CH_DIR, case + "_normalized.npy"))).max())
    say(f"   {d['key']}: re-cropped colour image matches the stored network input (max abs difference {dev:.2e})")
    return view


def annulus(a, u_cup, f_lo, f_hi):
    xo, yo = xy(u_cup, f_hi)
    xi, yi = xy(u_cup, f_lo)
    verts = np.concatenate([np.c_[xo, yo], np.c_[xi[::-1], yi[::-1]]])
    codes = (
        [MPath.MOVETO]
        + [MPath.LINETO] * (len(xo) - 2)
        + [MPath.CLOSEPOLY]
        + [MPath.MOVETO]
        + [MPath.LINETO] * (len(xi) - 2)
        + [MPath.CLOSEPOLY]
    )
    a.add_patch(PathPatch(MPath(verts, codes), facecolor=BLUE, alpha=0.25, edgecolor="none", zorder=2))
    for f_ in (f_lo, f_hi):
        a.plot(*xy(u_cup, f_), color=BLUE, lw=0.8, ls=(0, (2.5, 1.5)), zorder=3)


def panel(d, path, size_cm=2.9, label=True, experts=True):
    img = rgb_crop(d)
    iv = get_ivs("TK", [d])[0]
    er = ext(d["mu"][25:]) / ext(d["mu"][:25])
    pt = d["a"] / d["b"]
    s = size_cm * CM
    fig = plt.figure(figsize=(s, s))
    a = fig.add_axes([0, 0, 1, 1])
    a.imshow(img)
    a.set_axis_off()
    xd, yd = xy(d["mu"][:25])
    cx, cy = (xd.min() + xd.max()) / 2, (yd.min() + yd.max()) / 2
    half = 0.8 * max(np.ptp(xd), np.ptp(yd))
    a.set_xlim(cx - half, cx + half)
    a.set_ylim(cy + half, cy - half)
    if experts:
        for u in d["U"]:
            a.plot(*xy(u[25:]), color=VERM, lw=0.6, zorder=4)
    if iv is not None:
        annulus(a, d["mu"][25:], max(iv[0], 0) / er, iv[1] / er)
    a.plot(xd, yd, color=GREEN, lw=1.2, zorder=5)
    a.plot(*xy(d["mu"][25:]), color=BLUE, lw=1.2, zorder=6)
    k_in = int(np.sum((d["rx"] >= iv[0]) & (d["rx"] <= iv[1]))) if iv is not None else len(d["rx"])
    txt1 = f"vCDR {pt:.2f} [{iv[0]:.2f}, {iv[1]:.2f}]" if iv is not None else f"vCDR {pt:.2f} [unbounded]"
    if label:
        t = a.text(
            0.03,
            0.97,
            txt1 + f"\n{k_in}/{len(d['rx'])} experts inside",
            transform=a.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.6),
            zorder=10,
        )
        fig.canvas.draw()
        if (
            t.get_window_extent().width > 0.95 * fig.bbox.width
        ):  # too wide for the frame: break after the point estimate
            t.set_text(txt1.replace(" [", "\n[") + f"\n{k_in}/{len(d['rx'])} experts inside")
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return dict(
        key=d["key"],
        spread=float(sp(d)),
        vcdr=float(pt),
        interval=None if iv is None else [float(iv[0]), float(iv[1])],
        inside=k_in,
        n=len(d["rx"]),
    )


# ---------------- 4. Figure 2(b-e)
say("\n4. FIG 2(b-e): lowest and highest expert spread, RIGA test and Chaksu test")
X["fig2be"] = {}
for tag, (dn, ds) in zip(["b", "c", "d", "e"], [("RIGA", TST), ("RIGA", TST), ("Chaksu", CT_), ("Chaksu", CT_)]):
    d = min(ds, key=sp) if tag in ("b", "d") else max(ds, key=sp)
    info = panel(d, os.path.join(FD, f"fig2{tag}_{dn}_{'low' if tag in ('b', 'd') else 'high'}_spread.png"))
    X["fig2be"][tag] = info
    say(
        f"   ({tag}) {info['key']}: expert SD {info['spread']:.3f}, vCDR {info['vcdr']:.2f} {info['interval']}, {info['inside']}/{info['n']} inside"
    )

# ---------------- 5. Figure 1 assets
case = os.environ.get("CASE")
d1 = [x for x in TST if x["key"] == case][0] if case else sorted(TST, key=sp)[len(TST) // 2]
ds1, cs1 = d1["key"].split("/", 1)
say(f"\n5. FIG 1 ASSETS for {d1['key']}")
rec = json.load(open(os.path.join(PROCESSED_DIR, ds1, cs1 + ".json")))
tr = json.load(open(os.path.join(PREPROC_DIR, ds1, cs1 + "_transform.json")))


def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


full = cv2.imread(resolve(rec["image_path"]))
dia, cx_, cy_ = tr["disc_diameter"], tr["center_x"], tr["center_y"]
lw_ = max(3, full.shape[1] // 250)
cv2.rectangle(
    full, (int(cx_ - dia / 2), int(cy_ - dia / 2)), (int(cx_ + dia / 2), int(cy_ + dia / 2)), (255, 255, 255), lw_
)
cv2.imwrite(os.path.join(FD, "f1_fundus_box.png"), full)
crop = rgb_crop(d1)
enh = crop.copy()
enh[:, :, 1] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop[:, :, 1])
cv2.imwrite(os.path.join(FD, "f1_crop_clahe.png"), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))
# coefficient bars (ensemble mean), disc green, cup blue; the two DC terms are clipped at the top
mu = ens_pred(d1["key"]).mean(0)
fig, a = plt.subplots(figsize=(1.2, 0.55))
xs = np.r_[np.arange(25), np.arange(25) + 27]
cols = [GREEN] * 25 + [BLUE] * 25
a.bar(xs, mu, width=0.8, color=cols, lw=0)
lim = 1.25 * np.abs(np.r_[mu[1:25], mu[26:]]).max()
a.set_ylim(-lim, lim)
a.axhline(0, color="0.3", lw=0.5)
a.set_axis_off()
a.text(12, -lim * 1.05, "disc", ha="center", va="top", color=GREEN)
a.text(39, -lim * 1.05, "cup", ha="center", va="top", color=BLUE)
for e in ("pdf", "svg"):
    fig.savefig(os.path.join(FD, f"f1_coef_bars.{e}"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
X["fig1_output"] = panel(d1, os.path.join(FD, "f1_output.png"), size_cm=2.2, label=False, experts=False)
# expert outlines on the crop
fig = plt.figure(figsize=(2.2 * CM, 2.2 * CM))
a = fig.add_axes([0, 0, 1, 1])
a.imshow(crop)
a.set_axis_off()
xd, yd = xy(d1["ubar"][:25])
c0x, c0y = (xd.min() + xd.max()) / 2, (yd.min() + yd.max()) / 2
h0 = 0.8 * max(np.ptp(xd), np.ptp(yd))
for u in d1["U"]:
    a.plot(*xy(u[:25]), color=VERM, lw=0.6)
    a.plot(*xy(u[25:]), color=VERM, lw=0.6)
a.set_xlim(c0x - h0, c0x + h0)
a.set_ylim(c0y + h0, c0y - h0)
fig.savefig(os.path.join(FD, "f1_experts.png"), dpi=600)
plt.close(fig)


# r(theta): raw expert-mean radius against the 12-harmonic fit
def raw_radii(key):
    ds_, case_ = key.split("/", 1)
    rc = json.load(open(os.path.join(PROCESSED_DIR, ds_, case_ + ".json")))
    t_ = json.load(open(os.path.join(PREPROC_DIR, ds_, case_ + "_transform.json")))
    out = []
    for rid, data in rc["raters"].items():
        if norm(case_.replace("_", "/", 1) + "_rater" + re.sub(r"\D", "", str(rid))) in unstable:
            continue
        rr = []
        for sk in ("disc_contour", "cup_contour"):
            c = data.get(sk)
            if c is None:
                break
            cr = map_to_crop_space(c, t_["x_min"], t_["y_min"], t_["scale"])
            if not valid_contour(cr):
                break
            rr.append(contour_to_radial(cr)[1])
        if len(rr) == 2:
            out.append(np.concatenate(rr))
    return out


R1 = np.mean(raw_radii(d1["key"]), 0)
deg = np.degrees(GRID) % 360
o = np.argsort(deg)
fig, a = plt.subplots(figsize=(1.35, 0.85))
for sl, col in ((slice(0, 360), GREEN), (slice(360, 720), BLUE)):
    r_ = R1[sl]
    fitv = Phi @ (Phi.T @ r_)
    a.plot(deg[o][::6], r_[o][::6], ls="none", marker="o", ms=1.2, color="0.45")
    a.plot(deg[o], fitv[o], color=col, lw=1.1)
a.set_xticks([0, 180, 360])
a.set_xlabel("θ (°)", labelpad=1)
a.set_ylabel("r (px)", labelpad=1)
a.set_xlim(0, 360)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
for e in ("pdf", "svg"):
    fig.savefig(os.path.join(FD, f"f1_rtheta.{e}"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
# training-target vector (standardised), disc block outlined green, cup block blue
tm, ts = members[0][1], members[0][2]
zt = (d1["ubar"] - tm) / ts
fig, a = plt.subplots(figsize=(0.32, 1.2))
a.imshow(zt[:, None], cmap="Greys", aspect="auto", vmin=-2.5, vmax=2.5, interpolation="nearest")
from matplotlib.patches import Rectangle

a.add_patch(Rectangle((-0.5, -0.5), 1, 25, fill=False, ec=GREEN, lw=1.2))
a.add_patch(Rectangle((-0.5, 24.5), 1, 25, fill=False, ec=BLUE, lw=1.2))
a.set_axis_off()
for e in ("pdf", "svg"):
    fig.savefig(os.path.join(FD, f"f1_target_vector.{e}"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
# correlation heatmaps of expert deviations (training images): radii against harmonic coefficients
rawdev, hdev = [], []
for d in D["train"]:
    rs = raw_radii(d["key"])
    if len(rs) >= 2:
        Rm = np.array(rs)
        rawdev.append(Rm - Rm.mean(0))
    hdev.append(d["U"] - d["ubar"])
rawdev, hdev = np.concatenate(rawdev), np.concatenate(hdev)
Craw, Ch = np.abs(np.corrcoef(rawdev, rowvar=False)), np.abs(np.corrcoef(hdev, rowvar=False))
offm = lambda C: float(C[~np.eye(len(C), dtype=bool)].mean())
for nm, C in [("f1_corr_radii", Craw[::4, ::4]), ("f1_corr_coefficients", Ch)]:
    fig = plt.figure(figsize=(0.8, 0.8))
    a = fig.add_axes([0, 0, 1, 1])
    a.imshow(C, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    a.set_axis_off()
    fig.savefig(os.path.join(FD, nm + ".pdf"), pad_inches=0)
    fig.savefig(os.path.join(FD, nm + ".png"), dpi=600, pad_inches=0)
    plt.close(fig)
say(
    f"   mean |off-diagonal correlation|: radii {offm(Craw):.3f}, coefficients {offm(Ch):.3f}  (paper: 0.270 and 0.030)"
)
X["fig1"] = dict(case=d1["key"], corr_radii=offm(Craw), corr_coef=offm(Ch), tau=fit["TK"], **X["fig1_output"])
say(
    f"   Fig. 1 output label: vCDR {X['fig1_output']['vcdr']:.2f} {np.round(X['fig1_output']['interval'], 2).tolist()}; tau = {fit['TK']:.2f}"
)
json.dump(
    X,
    open(os.path.join(OUT, "figs_v2_results.json"), "w"),
    indent=1,
    default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o),
)
say(f"\nSaved {FD}\nFIGS v2 DONE")
