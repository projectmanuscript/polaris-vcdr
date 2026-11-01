"""
EXTRAS RUN: additional analyses and supplementary material. Read-only on data; everything goes to outputs/extras_run/.
Step 0 re-executes final_run.py in memory, unchanged except that its outputs go to outputs/extras_run/ (so final_run/ is
not overwritten). Same fits, same Table 1 and Table 2 numbers. Nothing below is fitted on Chaksu test images.
  A.  Paired cluster bootstrap (images resampled, 5000), Chaksu overall and per camera (RIGA for reference):
        TK minus the fixed harmonic band of width 0.282; fixed harmonic band minus fixed polar U-Net band, both width 0.297.
  B.  Recomputed on the cleaned split: inter-rater agreement (94 test images), detector centre error (test without the
      8 duplicates, and on the 94 analysis images), Spearman of predicted against observed rater spread (117 calibration).
      Each is also recomputed on the pre-dedup set, to show the code reproduces the old value.
  C.  Fair polar U-Net (outputs/phase8_fair/polar; used when >= 3 seeds exist, FAIR=0 skips): segmentation on RIGA test and
      Chaksu, paired tests against the harmonic ensemble, its fixed-width conformal band (CFf) fitted on the same 117
      calibration images, width-matched bands at 0.282 / 0.297, paired bootstrap against TK and the harmonic band.
  S1. Coverage and width of every configuration (Table 2) and reliability 50-95% for every method, both datasets.
  S2. Conditional coverage by rater-spread tertile (cut-points from cleaned calibration) and by vCDR level.
  S3. Signed bias and MAE of vCDR by reference vCDR level, all models.
  S4. Centre-shift curve, 0-20 px in 4 directions, harmonic ensemble and every U-Net, 94 RIGA test images (SKIP_SHIFT=1 skips).
  S5. Negative results: BW transport (incl. 5-fold cross-fitting on calibration), per-image dispersion head (Spearman),
      the pre-specified E, the 0.6-crossing kappa (E and TK, both datasets).
  S6. Secondary: every configuration recalibrated on the Chaksu training images (same 100 drawn as before), scored on Chaksu test.
  S7. Duplicate sensitivity: RIGA test results with and without the 8 duplicated images (cleaned fits, same models).
  Q1. Interval bands drawn on ROI crops, easy and hard cases, RIGA and Chaksu (only crops whose window lies inside the photograph).
  Q2. Worst polar U-Net vCDR errors on Remidio and the polar seam.
  I.  Implementation facts read from configs, checkpoints and training logs.
Run:  python3 extras_run.py 2>&1 | tee outputs/extras_run_console.log
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
for a, b in [
    ("'\"outputs\", \"final_run\")'", "'\"outputs\", \"extras_run\")'"),
    (
        "n0c, n0t = len(CAL), len(TST)",
        "n0c, n0t = len(CAL), len(TST); TST_FULL, CAL_FULL = list(D['test']), list(D['calib'])",
    ),
]:
    assert fr.count(a) == 1, a
    fr = fr.replace(a, b)
exec(fr)
assert OUT.endswith("extras_run"), OUT
from scipy.stats import spearmanr

X = {}
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)
SEG_MODELS = list(MODELS)


def jd(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    try:
        return float(o)
    except Exception:
        return str(o)


def ci_txt(r, s=100, f=1):
    return f"{r[0]*s:+.{f}f}, {r[1]*s:+.{f}f}"


def cams(ds):
    return {
        "all": list(range(len(ds))),
        **{c: [i for i, d in enumerate(ds) if d["src"] == c] for c in ["Bosch", "Forus", "Remidio"]},
    }


def add_res(dname, ds, k, iv):
    hn, w = score(iv, ds)
    c_, w_ = boot(hn, w)
    RES2[dname][k] = dict(
        hn=hn,
        w=w,
        cov=cov(hn),
        cov_ci=c_.tolist(),
        width=float(np.nanmean(w)),
        width_ci=w_.tolist(),
        unb=int(sum(x is None for x in iv)),
        iv=iv,
    )


def show_pair(dname, g, a, b, lab):
    r = paired(dname, a, b, DS_IDX[dname][g])
    PT[f"{dname}|{g}|{a}-{b}"] = r
    say(
        f"   {dname:6s} {g:8s} {lab}: coverage {r['dcov']*100:+5.1f} [{ci_txt(r['dcov_ci'])}] p={r['p_cov']:.4f}; "
        f"width {r['dw']:+.4f} [{r['dw_ci'][0]:+.4f}, {r['dw_ci'][1]:+.4f}] p={r['p_w']:.4f}"
    )
    return r


DS_IDX = {"RIGA": {"all": list(range(len(TST)))}, "Chaksu": cams(CT_)}

say("\n\n==================== EXTRAS ====================")
# ---------------- A. paired tests on the existing outputs
say("\nA. PAIRED CLUSTER BOOTSTRAP (A minus B; coverage in points, width in vCDR units; images resampled, 5000)")
for dname in ["Chaksu", "RIGA"]:
    for g in DS_IDX[dname]:
        show_pair(dname, g, "TK", "W:CFh:0.282", "TK - fixed harmonic band 0.282         ")
    for g in DS_IDX[dname]:
        show_pair(dname, g, "W:CFh:0.297", "W:CFu:0.297", "harmonic band 0.297 - U-Net band 0.297  ")
    for g in DS_IDX[dname]:
        show_pair(dname, g, "TK", f"W:CFh:{tk_w:.4f}", f"TK - harmonic band at TK width {tk_w:.4f}")
X["A"] = {k: v for k, v in PT.items() if "W:" in k}

# ---------------- B. three numbers recomputed on the cleaned split
say("\nB. RECOMPUTED ON THE CLEANED SPLIT")


def pair_dice(us, part):
    ms = [coeffs_to_mask(u[part], Phi) > 0 for u in us]
    pw = [dice(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    lo = [
        dice(ms[i], coeffs_to_mask(np.mean([us[j][part] for j in range(len(us)) if j != i], 0), Phi) > 0)
        for i in range(len(ms))
    ]
    return np.mean(pw), np.mean(lo)


X["B"] = {}
for lab, ds in [("cleaned test", TST), ("pre-dedup test (old value 79.28 / 83.38)", TST_FULL)]:
    cd = [pair_dice(d["U"], slice(25, 50)) for d in ds]
    dd_ = [pair_dice(d["U"], slice(0, 25)) for d in ds]
    v = dict(
        n=len(ds),
        cup_pairwise=float(np.mean([c[0] for c in cd])),
        cup_vs_others=float(np.mean([c[1] for c in cd])),
        disc_pairwise=float(np.mean([c[0] for c in dd_])),
        disc_vs_others=float(np.mean([c[1] for c in dd_])),
    )
    X["B"]["agreement|" + lab] = v
    say(
        f"   inter-rater agreement, {lab} ({v['n']} images): cup pairwise {v['cup_pairwise']*100:.2f}, cup vs mean of others {v['cup_vs_others']*100:.2f}; "
        f"disc pairwise {v['disc_pairwise']*100:.2f}, disc vs mean of others {v['disc_vs_others']*100:.2f}"
    )
CE = json.load(open(os.path.join(BASE, "outputs", "centre_v3", "centre_errors.json")))["rows"]
te = [r for r in CE if r["part"] == "test" and r.get("new_v2split")]
TKEYS = {d["key"] for d in TST}
for lab, rs in [
    ("all test images, pre-dedup (old value 3.4)", te),
    ("test without the 8 duplicates", [r for r in te if r["key"] not in DUP_T]),
    ("the 94 RIGA analysis images", [r for r in te if r["key"] in TKEYS]),
]:
    cp = np.array([r["new_v2split"]["px"] * r["new_v2split"]["scale"] for r in rs])
    srcs = {s: sum(r["key"].startswith(s + "/") for r in rs) for s in sorted({r["key"].split("/")[0] for r in rs})}
    X["B"]["centre|" + lab] = dict(
        n=len(cp),
        median=float(np.median(cp)),
        p90=float(np.percentile(cp, 90)),
        max=float(cp.max()),
        gt10=float(np.mean(cp > 10)),
    )
    say(
        f"   detector centre error, {lab}: n={len(cp)} {srcs}, crop px median {np.median(cp):.1f}, 90th {np.percentile(cp, 90):.1f}, max {cp.max():.1f}, >10 px {np.mean(cp > 10)*100:.1f}%"
    )


def sd_v(d, Sig):
    g = np.concatenate([-d["a"] / d["b"] ** 2 * phi_v, phi_v / d["b"]])
    return float(np.sqrt(g @ Sig @ g))


for lab, ds in [("cleaned calibration", CAL), ("pre-dedup calibration (old value -0.007)", CAL_FULL)]:
    obs = [np.std(d["rx"], ddof=1) for d in ds]
    sde = np.array([sd_v(d, np.diag(d["e"])) for d in ds])
    sds = np.array([sd_v(d, d["S"]) for d in ds])
    sdk = np.array([sd_v(d, Sconst) for d in ds])
    v = dict(
        n=len(ds),
        model_only=spearmanr(sde, obs)[0],
        head_only=spearmanr(sds, obs)[0],
        summed=spearmanr(np.sqrt(sde**2 + sds**2), obs)[0],
        const_summed=spearmanr(np.sqrt(sde**2 + sdk**2), obs)[0],
    )
    X["B"]["spearman|" + lab] = v
    say(
        f"   Spearman with observed rater spread (SD of rater extent vCDR), {lab} ({v['n']} images): ensemble-only sd {v['model_only']:+.3f}; "
        f"dispersion-head sd {v['head_only']:+.3f}; ensemble + head sd {v['summed']:+.3f} (the -0.007 quantity); ensemble + constant rater cov {v['const_summed']:+.3f}"
    )

# ---------------- C. fair polar U-Net
FAIRDIR = os.path.join(BASE, "outputs", "phase8_fair", "polar")
FS = [k for k in range(1, 6) if os.path.exists(os.path.join(FAIRDIR, f"seed{k}_best.pt"))]
FAIR = os.environ.get("FAIR", "1") != "0" and len(FS) >= 3
say(
    f"\nC. FAIR POLAR U-NET: {'seeds ' + str(FS) if FAIR else 'not evaluated (needs >= 3 finished seeds in ' + FAIRDIR + ')'}"
)
FIXW = ["0.282", "0.297", f"{tk_w:.4f}"]
if FAIR:
    FM = []
    for k in FS:
        mm = PlainUNet().to(DEVICE)
        mm.load_state_dict(
            torch.load(os.path.join(FAIRDIR, f"seed{k}_best.pt"), map_location=DEVICE, weights_only=True)
        )
        mm.eval()
        FM.append(mm)

    @torch.no_grad()
    def fair_probs(im):
        x = np.stack([to_polar(np.ascontiguousarray(im[:, :, c])) for c in range(3)], -1)
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)[None].to(DEVICE)
        p = np.mean([torch.sigmoid(mm(t).float()).cpu().numpy()[0] for mm in FM], 0)
        return np.stack([to_cartesian(np.ascontiguousarray(p[c])) for c in range(2)]), p

    def fair_masks(imgs):
        out = []
        for im in imgs:
            pc, _ = fair_probs(im)
            out.append((lcc(pc[0] > 0.5), lcc(pc[1] > 0.5)))
        return out

    def fair_v(path):
        return vext(*fair_masks([np.load(path).astype(np.float32)])[0])

    chp = lambda d: os.path.join(CH_DIR, d["key"].split("/", 1)[1] + "_normalized.npy")
    for d in CAL + TST_FULL:
        d["f_ens"] = fair_v(npy(d["key"]))
    for d in CT_:
        d["f_ens"] = fair_v(chp(d))
    nn_ = sum(np.isnan(d["f_ens"]) for d in CAL + TST_FULL + CT_)
    say(
        f"   fair model vCDR computed on {len(CAL)} calibration, {len(TST_FULL)} RIGA test, {len(CT_)} Chaksu images; empty disc predictions: {nn_}"
    )
    MODELS.append("polarfair_ens")
    for dname in ["RIGA", "Chaksu"]:
        P, items, imgs = PRED[dname]
        P["polarfair_ens"] = fair_masks(imgs)
        rows = []
        for i, (_, case, gd, gc, grp) in enumerate(items):
            gv = vext(gd, gc)
            d_, c_ = P["polarfair_ens"][i]
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
        SEG[dname]["polarfair_ens"] = rows
    X["C_seg"] = {}
    rng_f = np.random.default_rng(2026)
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
            say(
                f"   {dname} / {g} (n={len(idx)}): fair polar U-Net disc {f('polarfair_ens', 'disc')*100:.2f}, cup {f('polarfair_ens', 'cup')*100:.2f}, "
                f"vCDR MAE {f('polarfair_ens', 'vcdr_ae'):.4f}, bias {f('polarfair_ens', 'bias'):+.4f}, cup outside disc {f('polarfair_ens', 'out_any')*100:.1f}%  "
                f"(harmonic {f('harmonic_ens', 'disc')*100:.2f} / {f('harmonic_ens', 'cup')*100:.2f} / {f('harmonic_ens', 'vcdr_ae'):.4f}; "
                f"original polar U-Net {f('polar_ens', 'disc')*100:.2f} / {f('polar_ens', 'cup')*100:.2f} / {f('polar_ens', 'vcdr_ae'):.4f})"
            )
            for other, tag in [
                ("harmonic_ens", "fair minus harmonic (requested test)"),
                ("polar_ens", "fair minus original polar (descriptive)"),
            ]:
                parts = []
                for met in ["disc", "cup", "vcdr_ae"]:
                    dl = np.array([rows["polarfair_ens"][i][met] - rows[other][i][met] for i in idx])
                    null = np.abs((rng_f.choice([-1.0, 1.0], (10000, len(dl))) * dl).mean(1))
                    p = (np.sum(null >= abs(dl.mean())) + 1) / 10001
                    bt = np.array([dl[rng_f.integers(0, len(dl), len(dl))].mean() for _ in range(5000)])
                    s_ = 1 if met == "vcdr_ae" else 100
                    X["C_seg"][f"{dname}|{g}|{other}|{met}"] = dict(
                        diff=float(dl.mean()), ci=np.percentile(bt, [2.5, 97.5]).tolist(), p=float(p)
                    )
                    parts.append(
                        f"{met} {dl.mean()*s_:+.{4 if s_ == 1 else 2}f} [{np.percentile(bt, 2.5)*s_:+.{4 if s_ == 1 else 2}f}, {np.percentile(bt, 97.5)*s_:+.{4 if s_ == 1 else 2}f}] p={p:.4f}"
                    )
                say(f"      {tag}: " + "; ".join(parts))
    fit["CFf"] = conformal_q(CAL, lambda d: d["f_ens"], LVL)
    NAMES["CFf"] = "conformal, fair polar U-Net (CFf)"
    say(f"   CFf half-width fitted on cleaned calibration: {fit['CFf']:.5f} (CFu {fit['CFu']:.5f})")


def gi(name, ds, z=Z90):
    if name == "CFf":
        return [(d["f_ens"] - fit["CFf"], d["f_ens"] + fit["CFf"]) for d in ds]
    if name.startswith("W:CFf:"):
        w = float(name.split(":")[2])
        return [(d["f_ens"] - w / 2, d["f_ens"] + w / 2) for d in ds]
    return get_ivs(name, ds, z)


FCONF = (["CFf"] + [f"W:CFf:{w}" for w in FIXW]) if FAIR else []
if FAIR:
    for dname, ds in [("RIGA", TST), ("Chaksu", CT_)]:
        for k in FCONF:
            add_res(dname, ds, k, gi(k, ds))
            r = RES2[dname][k]
            say(
                f"   {dname:6s} {k:12s} coverage {r['cov']*100:5.1f}% [{r['cov_ci'][0]*100:.1f}, {r['cov_ci'][1]*100:.1f}]  width {r['width']:.4f}"
            )
    say("   paired cluster bootstrap (A minus B):")
    for dname in ["Chaksu", "RIGA"]:
        for g in DS_IDX[dname]:
            for a, b in [("TK", "CFf"), ("CFf", "CFh"), ("CFf", "CFu")] + [
                (f"W:CFh:{w}", f"W:CFf:{w}") for w in FIXW[:2]
            ]:
                show_pair(dname, g, a, b, f"{a:12s} - {b:12s}")
    X["C_int"] = {
        dn: {k: {kk: vv for kk, vv in RES2[dn][k].items() if kk not in ("hn", "w", "iv")} for k in FCONF} for dn in RES2
    }


# ---------------- tables with every model (fair rows included when present)
def nm_of(k):
    if k in NAMES:
        return NAMES[k]
    _, b, w = k.split(":")
    return (
        f"Fixed width {w}, {'polar U-Net' if b == 'CFu' else ('fair polar U-Net' if b == 'CFf' else 'harmonic')} vCDR"
    )


T1 = [
    "| Dataset | Group | Model | Disc Dice | Cup Dice | vCDR MAE | vCDR bias | Cup outside disc |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
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
        for mn in MODELS:
            f = lambda k: float(np.mean([rows[mn][i][k] for i in idx]))
            T1.append(
                f"| {dname} | {g} (n={len(idx)}) | {mn} | {f('disc')*100:.2f} | {f('cup')*100:.2f} | {f('vcdr_ae'):.4f} | {f('bias'):+.4f} | {f('out_any')*100:.1f}% |"
            )
CONF_ALL = CONF + (["CFf"] if FAIR else []) + CTRL + FCONF[1:]
T2 = [
    "| Configuration | RIGA coverage [95% CI] | RIGA width | Chákṣu coverage [95% CI] | Chákṣu width |",
    "| --- | --- | --- | --- | --- |",
]
for k in CONF_ALL:
    a, b = RES2["RIGA"][k], RES2["Chaksu"][k]
    T2.append(
        f"| {nm_of(k)} | {a['cov']*100:.1f}% [{a['cov_ci'][0]*100:.1f}, {a['cov_ci'][1]*100:.1f}] | {a['width']:.3f} | {b['cov']*100:.1f}% [{b['cov_ci'][0]*100:.1f}, {b['cov_ci'][1]*100:.1f}] | {b['width']:.3f} |"
    )
open(os.path.join(OUT, "table1_segmentation_all.md"), "w").write("\n".join(T1))
open(os.path.join(OUT, "table2_intervals_all.md"), "w").write("\n".join(T2))
say(f"\n   tables with every model: {OUT}/table1_segmentation_all.md, table2_intervals_all.md")

# ---------------- S1. reliability for every method
say("\nS1. RELIABILITY, every method refitted on cleaned RIGA calibration at each level (individual-rater coverage, %)")


def ue_iv(ds, t, z):
    return [(d["u_seeds"].mean() - t * z * d["u_sd"], d["u_seeds"].mean() + t * z * d["u_sd"]) for d in ds]


def iv_at(k, v, ds, z):
    if k in ("E", "B", "K"):
        return ivs_of(ds, k, v, z)
    if k == "T":
        return ivs_of(ds, "E", v, z)
    if k == "TK":
        return ivs_of(ds, "K", v, z)
    if k == "C":
        return ivs_C(ds, *v, z)
    if k == "UE":
        return ue_iv(ds, v, z)
    pt = {"CFh": hpt, "CFu": lambda d: d["u_ens"], "CFf": lambda d: d["f_ens"]}[k]
    return [(pt(d) - v, pt(d) + v) for d in ds]


def fit_all(ds, lv, z):
    f_ = {
        "TK": coef_tau(ds, lv, z, lambda d: d["e"] + np.diag(Sconst)),
        "T": coef_tau(ds, lv, z, lambda d: d["e"] + np.diag(d["S"])),
        "E": fit_tau(ds, "E", lv, z),
        "K": fit_tau(ds, "K", lv, z),
        "B": fit_tau(ds, "B", lv, z),
        "C": fit_C(ds, lv, z),
        "CFh": conformal_q(ds, hpt, lv),
        "CFu": conformal_q(ds, lambda d: d["u_ens"], lv),
        "UE": float(pick(UTAU, [cov(score(ue_iv(ds, t, z), ds)[0]) for t in UTAU], lv)),
    }
    if FAIR:
        f_["CFf"] = conformal_q(ds, lambda d: d["f_ens"], lv)
    return f_


METH = ["TK", "T", "E", "K", "B", "C", "CFh", "CFu", "UE"] + (["CFf"] if FAIR else [])
RELA = {dn: {k: [] for k in METH} for dn in ["RIGA", "Chaksu"]}
WIDA = {dn: {k: [] for k in METH} for dn in ["RIGA", "Chaksu"]}
for lv in LEVELS:
    z = _N.ppf(0.5 + lv / 2)
    f_ = fit_all(CAL, lv, z)
    for dn, ds in [("RIGA", TST), ("Chaksu", CT_)]:
        for k in METH:
            hn, w = score(iv_at(k, f_[k], ds, z), ds)
            RELA[dn][k].append(cov(hn))
            WIDA[dn][k].append(float(np.nanmean(w)))
for dn in RELA:
    say(f"   {dn}: nominal " + " ".join(f"{l*100:5.0f}" for l in LEVELS))
    for k in METH:
        say(
            f"     {k:4s}  cov "
            + " ".join(f"{x*100:5.1f}" for x in RELA[dn][k])
            + "   width at 90%: "
            + f"{WIDA[dn][k][LEVELS.index(0.9)]:.4f}"
        )
    say(
        f"     check at 90% against Table 2: "
        + ", ".join(f"{k} {RELA[dn][k][LEVELS.index(0.9)]*100:.1f}/{RES2[dn][k]['cov']*100:.1f}" for k in METH)
    )
X["S1"] = {"levels": LEVELS, "coverage": RELA, "width": WIDA}
SUP = os.path.join(OUT, "supplement")
os.makedirs(SUP, exist_ok=True)
STY = {
    "TK": ("k-", "o", "k"),
    "T": ("k--", "s", "w"),
    "E": ("k:", "^", "k"),
    "K": ("k-.", "v", "w"),
    "B": ("k--", "D", "k"),
    "C": ("k:", "x", "k"),
    "CFh": ("k--", "s", "w"),
    "CFu": ("k:", "^", "k"),
    "UE": ("k-.", "D", "w"),
    "CFf": ("k-", "x", "k"),
}
LAB = {
    "TK": "TK (final)",
    "T": "T",
    "E": "E (pre-specified)",
    "K": "K",
    "B": "B (model only)",
    "C": "C (BW transport)",
    "CFh": "conformal, harmonic",
    "CFu": "conformal, polar U-Net",
    "UE": "U-Net seed spread",
    "CFf": "conformal, fair polar U-Net",
}
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
for r_, dn in enumerate(["RIGA", "Chaksu"]):
    for c_, fam in enumerate([["TK", "T", "E", "K", "B", "C"], ["TK", "CFh", "CFu", "UE"] + (["CFf"] if FAIR else [])]):
        a = axs[r_, c_]
        L = np.array(LEVELS) * 100
        a.plot([45, 100], [45, 100], color="0.6", lw=0.8, ls=(0, (4, 3)))
        for k in fam:
            ls, mk, fc = STY[k]
            a.plot(L, np.array(RELA[dn][k]) * 100, ls, marker=mk, ms=3.5, lw=0.9, mfc=fc, mec="k", label=LAB[k])
        a.set_xlim(45, 100)
        a.set_ylim(30, 100)
        a.grid(color="0.9", lw=0.5)
        a.set_title(
            f"{'RIGA test' if dn == 'RIGA' else 'Chákṣu test (zero-shot)'}: {'harmonic covariances' if c_ == 0 else 'final method and baselines'}",
            fontsize=8,
        )
        if r_ == 1:
            a.set_xlabel("Nominal coverage (%)")
        if c_ == 0:
            a.set_ylabel("Individual-rater coverage (%)")
for c_ in range(2):
    h_, l_ = axs[0, c_].get_legend_handles_labels()
    axs[1, c_].legend(h_, l_, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.subplots_adjust(hspace=0.22, wspace=0.08, bottom=0.2)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(SUP, f"figS1_reliability_all_methods.{e}"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- S2. conditional coverage
say(
    "\nS2. CONDITIONAL individual-rater coverage at 90% (spread = SD of the raters' extent vCDR; level = mean rater extent vCDR)"
)
sp_ = lambda d: np.std(d["rx"], ddof=1)
CUTS = np.percentile([sp_(d) for d in CAL], [100 / 3, 200 / 3])
GR = {
    "spread low": lambda d: sp_(d) <= CUTS[0],
    "spread mid": lambda d: CUTS[0] < sp_(d) <= CUTS[1],
    "spread high": lambda d: sp_(d) > CUTS[1],
    "vCDR <0.5": lambda d: np.mean(d["rx"]) < 0.5,
    "vCDR 0.5-0.7": lambda d: 0.5 <= np.mean(d["rx"]) < 0.7,
    "vCDR >=0.7": lambda d: np.mean(d["rx"]) >= 0.7,
}
CK = CONF + (["CFf"] if FAIR else [])
say(f"   spread tertile cut-points from cleaned calibration: {CUTS.round(4).tolist()}")
S2 = {}
T3 = ["| Dataset | Group | n | " + " | ".join(CK) + " |", "| --- | --- | --- | " + " | ".join(["---"] * len(CK)) + " |"]
for dn, ds in [("RIGA", TST), ("Chaksu", CT_)]:
    say(f"   {dn}: " + f"{'group':14s}{'n':>4s}" + "".join(f"{k:>7s}" for k in CK))
    for g, fn in GR.items():
        idx = [i for i, d in enumerate(ds) if fn(d)]
        if not idx:
            say(f"     {g:14s}   0")
            continue
        row = {k: cov(RES2[dn][k]["hn"][idx]) for k in CK}
        S2[f"{dn}|{g}"] = dict(n=len(idx), **row)
        say(f"     {g:14s}{len(idx):4d}" + "".join(f"{row[k]*100:6.1f}%" for k in CK))
        T3.append(f"| {dn} | {g} | {len(idx)} | " + " | ".join(f"{row[k]*100:.1f}%" for k in CK) + " |")
open(os.path.join(SUP, "tableS2_conditional_coverage.md"), "w").write("\n".join(T3))
X["S2"] = dict(cuts=CUTS.tolist(), groups=S2)

# ---------------- S3. bias and MAE by level, all models
say(
    "\nS3. vCDR BIAS (prediction minus reference) and MAE BY REFERENCE vCDR LEVEL (reference = mask of the mean rater outline)"
)
T4 = [
    "| Dataset | Level | n | " + " | ".join(f"{m} bias / MAE" for m in MODELS) + " |",
    "| --- | --- | --- | " + " | ".join(["---"] * len(MODELS)) + " |",
]
S3 = {}
for dn in ["RIGA", "Chaksu"]:
    rows = SEG[dn]
    for lo, hi, lab in [(0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 9, ">=0.7")]:
        idx = [i for i, r in enumerate(rows["harmonic_ens"]) if lo <= r["v_ref"] < hi]
        if not idx:
            say(f"   {dn:6s} {lab:8s} n=0")
            T4.append(f"| {dn} | {lab} | 0 | " + " | ".join(["-"] * len(MODELS)) + " |")
            continue
        v = {
            m: (float(np.mean([rows[m][i]["bias"] for i in idx])), float(np.mean([rows[m][i]["vcdr_ae"] for i in idx])))
            for m in MODELS
        }
        S3[f"{dn}|{lab}"] = dict(n=len(idx), **v)
        say(f"   {dn:6s} {lab:8s} n={len(idx):3d}: " + ", ".join(f"{m} {v[m][0]:+.3f}/{v[m][1]:.3f}" for m in MODELS))
        T4.append(
            f"| {dn} | {lab} | {len(idx)} | " + " | ".join(f"{v[m][0]:+.3f} / {v[m][1]:.3f}" for m in MODELS) + " |"
        )
open(os.path.join(SUP, "tableS3_bias_by_level.md"), "w").write("\n".join(T4))
X["S3"] = S3


# ---------------- S4. centre-shift curve
def masks_for(mn, imgs):
    if mn == "harmonic_ens":
        return harm_masks(imgs)
    if mn == "polarfair_ens":
        return fair_masks(imgs)
    return unet_masks(mn.replace("_ens", ""), imgs)


if os.environ.get("SKIP_SHIFT") != "1":
    SHIFTS, DIRS = [0, 3, 5, 10, 15, 20], [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def shf(a, dx, dy, nearest=False):
        M_ = np.float32([[1, 0, dx], [0, 1, dy]])
        if nearest:
            return cv2.warpAffine(a.astype(np.uint8), M_, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        return cv2.warpAffine(a, M_, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    VAR = [(s, dx * s, dy * s) for s in SHIFTS for (dx, dy) in (DIRS if s else [(0, 0)])]
    _, items, imgs = PRED["RIGA"]
    SH = {mn: {s: {"disc": [], "cup": [], "ae": []} for s in SHIFTS} for mn in MODELS}
    for im, (_, _, gd, gc, _) in zip(imgs, items):
        vi = [shf(im, dx, dy) for _, dx, dy in VAR]
        vg = [(shf(gd, dx, dy, True), shf(gc, dx, dy, True)) for _, dx, dy in VAR]
        for mn in MODELS:
            for (s, _, _), (d_, c_), (sd_, sc_) in zip(VAR, masks_for(mn, vi), vg):
                SH[mn][s]["disc"].append(dice(d_, sd_))
                SH[mn][s]["cup"].append(dice(c_, sc_))
                SH[mn][s]["ae"].append(abs(vext(d_, c_) - vext(sd_, sc_)))
    say(
        f"\nS4. CENTRE-SHIFT, {len(imgs)} RIGA test images x 4 directions; disc Dice / cup Dice / vCDR MAE per shift (px in the 512 crop)"
    )
    for mn in MODELS:
        say(
            f"   {mn:14s} "
            + "  ".join(
                f"{s:2d}px {np.mean(SH[mn][s]['disc'])*100:5.2f}/{np.mean(SH[mn][s]['cup'])*100:5.2f}/{np.nanmean(SH[mn][s]['ae']):.4f}"
                for s in SHIFTS
            )
        )
    say(
        "   (0 px must equal Table 1 RIGA: "
        + ", ".join(
            f"{mn} {TAB1[('RIGA', 'all')][mn]['disc']*100:.2f}/{TAB1[('RIGA', 'all')][mn]['cup']*100:.2f}"
            for mn in SEG_MODELS
        )
        + ")"
    )
    X["S4"] = {mn: {str(s): {k: float(np.nanmean(v)) for k, v in SH[mn][s].items()} for s in SHIFTS} for mn in MODELS}
    MSTY = {
        "harmonic_ens": ("k-", "o", "k"),
        "plain_ens": ("k--", "s", "w"),
        "attention_ens": ("k:", "^", "k"),
        "deepres_ens": ("k-.", "v", "w"),
        "polar_ens": ("k--", "D", "k"),
        "polarfair_ens": ("k:", "x", "k"),
    }
    MLAB = {
        "harmonic_ens": "harmonic",
        "plain_ens": "plain U-Net",
        "attention_ens": "attention U-Net",
        "deepres_ens": "deep residual U-Net",
        "polar_ens": "polar U-Net",
        "polarfair_ens": "fair polar U-Net",
    }
    med_ce = X["B"]["centre|the 94 RIGA analysis images"]["median"]
    p90_ce = X["B"]["centre|the 94 RIGA analysis images"]["p90"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for a, (k, yl, s_) in zip(
        axs, [("disc", "Disc Dice (%)", 100), ("cup", "Cup Dice (%)", 100), ("ae", "vCDR MAE", 1)]
    ):
        for mn in MODELS:
            ls, mk, fc = MSTY[mn]
            a.plot(
                SHIFTS,
                [np.nanmean(SH[mn][s][k]) * s_ for s in SHIFTS],
                ls,
                marker=mk,
                ms=3.5,
                lw=0.9,
                mfc=fc,
                mec="k",
                label=MLAB[mn],
            )
        for xv, lsx in [(med_ce, ":"), (p90_ce, "--")]:
            a.axvline(xv, color="0.5", lw=0.7, ls=lsx)
        a.set_xlabel("Centre shift (px, 512 crop)")
        a.set_ylabel(yl)
        a.grid(color="0.9", lw=0.5)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=7, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(SUP, f"figS4_centre_shift.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(
        f"   vertical lines: measured detector error on the 94 images, median {med_ce:.1f} px (dotted), 90th percentile {p90_ce:.1f} px (dashed)"
    )

# ---------------- S5. negative results
say("\nS5. NEGATIVE RESULTS")
say(
    "   BW transport (C) and pre-specified E, 90%: "
    + "; ".join(
        f"{dn} C {RES2[dn]['C']['cov']*100:.1f}% w {RES2[dn]['C']['width']:.4f}, E {RES2[dn]['E']['cov']*100:.1f}% w {RES2[dn]['E']['width']:.4f}, "
        f"TK {RES2[dn]['TK']['cov']*100:.1f}% w {RES2[dn]['TK']['width']:.4f}"
        for dn in ["RIGA", "Chaksu"]
    )
)
perm = np.random.default_rng(7).permutation(len(CAL))
folds = np.array_split(perm, 5)
cf = {k: {"hn": [None] * len(CAL), "w": [None] * len(CAL)} for k in ["E", "C", "T", "TK"]}
for fo in folds:
    fs = set(fo.tolist())
    tr = [CAL[i] for i in range(len(CAL)) if i not in fs]
    te_ = [CAL[i] for i in fo]
    fv = {
        "E": fit_tau(tr, "E", LVL, Z90),
        "C": fit_C(tr, LVL, Z90),
        "T": coef_tau(tr, LVL, Z90, lambda d: d["e"] + np.diag(d["S"])),
        "TK": coef_tau(tr, LVL, Z90, lambda d: d["e"] + np.diag(Sconst)),
    }
    for k in cf:
        hn, w = score(iv_at(k, fv[k], te_, Z90), te_)
        for j, i in enumerate(fo):
            cf[k]["hn"][i], cf[k]["w"][i] = hn[j], w[j]
for k in cf:
    cf[k]["hn"] = np.array(cf[k]["hn"])
    cf[k]["w"] = np.array(cf[k]["w"], float)
say(
    f"   5-fold cross-fitting on the {len(CAL)} calibration images (fit on 4/5, score on 1/5; constant rater covariance of TK from all calibration images): "
    + ", ".join(f"{k} {cov(cf[k]['hn'])*100:.1f}% w {np.nanmean(cf[k]['w']):.4f}" for k in cf)
)
ok_ = ~(np.isnan(cf["C"]["w"]) | np.isnan(cf["E"]["w"]))
dl = cf["C"]["w"][ok_] - cf["E"]["w"][ok_]
null = np.abs((np.random.default_rng(1).choice([-1.0, 1.0], (10000, len(dl))) * dl).mean(1))
say(
    f"   paired width C - E in cross-fitting: {dl.mean():+.4f} (C narrower on {np.mean(dl < 0)*100:.0f}% of images), sign-flip p = {(np.sum(null >= abs(dl.mean())) + 1)/10001:.4f}"
)
sc_ = X["B"]["spearman|cleaned calibration"]
say(
    f"   per-image dispersion head: Spearman with observed spread {sc_['head_only']:+.3f} (head alone), {sc_['summed']:+.3f} (ensemble + head)"
)


def kappa06(iv, ds):
    A_ = np.array([x is None or (x[0] < 0.6 < x[1]) for x in iv])
    Bv = np.array([d["rx"].min() < 0.6 <= d["rx"].max() for d in ds])
    po = np.mean(A_ == Bv)
    pe = A_.mean() * Bv.mean() + (1 - A_.mean()) * (1 - Bv.mean())
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), int(A_.sum()), int(Bv.sum()), int(np.sum(A_ & Bv))


X["S5"] = {"crossfit": {k: dict(cov=cov(v["hn"]), width=float(np.nanmean(v["w"]))) for k, v in cf.items()}, "kappa": {}}
for dn, ds in [("RIGA", TST), ("Chaksu", CT_)]:
    for k in ["E", "TK"]:
        kp, na, nb, nab = kappa06(RES2[dn][k]["iv"], ds)
        X["S5"]["kappa"][f"{dn}|{k}"] = kp
        say(
            f"   0.6 crossing, {dn} {k}: interval crosses 0.6 on {na} images, raters split around 0.6 on {nb}, both {nab}; Cohen kappa {kp:.3f}"
        )

# ---------------- S7. duplicate sensitivity
EX8 = [d for d in TST_FULL if d["key"] in DUP_T]
for d in EX8:
    d["V"]["K"] = vq(np.diag(d["e"]) + Sconst)
    d["u_sd"] = max(np.std(d["u_seeds"], ddof=1), sfloor)
say(
    f"\nS7. DUPLICATE SENSITIVITY: RIGA test with the {len(EX8)} duplicated images ({len(TST_FULL)}) vs without ({len(TST)}); fits from cleaned calibration"
)
S7 = {}
T5 = ["| Configuration | Coverage, 94 | Width, 94 | Coverage, 102 | Width, 102 |", "| --- | --- | --- | --- | --- |"]
for k in CONF + (["CFf"] if FAIR else []):
    hn1, w1 = score(gi(k, TST_FULL), TST_FULL)
    a = RES2["RIGA"][k]
    S7[k] = dict(cov94=a["cov"], w94=a["width"], cov102=cov(hn1), w102=float(np.nanmean(w1)))
    say(f"   {nm_of(k):46s} {a['cov']*100:5.1f}% / {a['width']:.4f}   ->   {cov(hn1)*100:5.1f}% / {np.nanmean(w1):.4f}")
    T5.append(f"| {nm_of(k)} | {a['cov']*100:.1f}% | {a['width']:.3f} | {cov(hn1)*100:.1f}% | {np.nanmean(w1):.3f} |")
ex_items = [
    (
        npy(d["key"]),
        d["key"].split("/", 1)[1],
        coeffs_to_mask(d["ubar"][:25], Phi) > 0,
        coeffs_to_mask(d["ubar"][25:], Phi) > 0,
        "RIGA",
    )
    for d in EX8
]
ex_imgs = [np.load(p).astype(np.float32) for p, *_ in ex_items]
T5 += ["", "| Model | Disc / cup Dice / vCDR MAE, 94 | Same, 102 |", "| --- | --- | --- |"]
for mn in MODELS:
    ms = masks_for(mn, ex_imgs)
    ex_rows = [
        dict(disc=dice(dm, gd), cup=dice(cm, gc), ae=abs(vext(dm, cm) - vext(gd, gc)))
        for (dm, cm), (_, _, gd, gc, _) in zip(ms, ex_items)
    ]
    r94 = SEG["RIGA"][mn]
    f94 = lambda k: np.mean([r[k] for r in r94])
    f102 = lambda k, k2: np.mean([r[k] for r in r94] + [r[k2] for r in ex_rows])
    S7["seg|" + mn] = dict(
        disc94=f94("disc"),
        cup94=f94("cup"),
        mae94=f94("vcdr_ae"),
        disc102=f102("disc", "disc"),
        cup102=f102("cup", "cup"),
        mae102=f102("vcdr_ae", "ae"),
    )
    say(
        f"   {mn:14s} disc/cup/MAE {f94('disc')*100:.2f}/{f94('cup')*100:.2f}/{f94('vcdr_ae'):.4f}  ->  {f102('disc', 'disc')*100:.2f}/{f102('cup', 'cup')*100:.2f}/{f102('vcdr_ae', 'ae'):.4f}"
    )
    T5.append(
        f"| {mn} | {f94('disc')*100:.2f} / {f94('cup')*100:.2f} / {f94('vcdr_ae'):.4f} | {f102('disc', 'disc')*100:.2f} / {f102('cup', 'cup')*100:.2f} / {f102('vcdr_ae', 'ae'):.4f} |"
    )
open(os.path.join(SUP, "tableS7_duplicate_sensitivity.md"), "w").write("\n".join(T5))
X["S7"] = S7

# ---------------- Q1. interval bands on ROI crops
say(
    "\nQ1. INTERVAL BANDS ON ROI CROPS (TK at 90%); only crops whose 2.6x window lies inside the photograph (no border, no corner text)"
)


def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


def window_inside(dn, d, margin=5):
    if dn == "RIGA":
        ds_, case = d["key"].split("/", 1)
        tr = json.load(open(os.path.join(PREPROC_DIR, ds_, case + "_transform.json")))
        ip = resolve(json.load(open(os.path.join(PROCESSED_DIR, ds_, case + ".json")))["image_path"])
        im = cv2.imread(ip) if ip else None
        if im is None:
            return False
        x0, y0, side = tr["x_min"], tr["y_min"], 2.6 * tr["disc_diameter"]
    else:
        _, cam, k = d["key"].split("/", 1)[1].split("_", 2)
        im = cv2.imread(MAN["Test"][cam][k]["image"])
        r = yolo(im, conf=0.01, verbose=False)[0]
        x, y, w, h = r.boxes[0].xywh[0].tolist()
        dia = max(w, h)
        x0, y0, side = x - 1.3 * dia, y - 1.3 * dia, 2.6 * dia
    H_, W_ = im.shape[:2]
    return x0 >= margin and y0 >= margin and x0 + side <= W_ - margin and y0 + side <= H_ - margin


def cxy(u25, f=1.0):
    r = (Phi @ u25) * f
    x, y = 256 + r * np.cos(GRID), 256 + r * np.sin(GRID)
    return np.append(x, x[0]), np.append(y, y[0])


def draw_case(a, dn, d, iv, title):
    path = npy(d["key"]) if dn == "RIGA" else os.path.join(CH_DIR, d["key"].split("/", 1)[1] + "_normalized.npy")
    g = np.load(path)[:, :, 1]
    lo_, hi_ = np.percentile(g, [1, 99])
    a.imshow(np.clip((g - lo_) / (hi_ - lo_ + 1e-6), 0, 1), cmap="gray")
    for u in d["U"]:
        a.plot(*cxy(u[25:]), color="k", lw=0.5)
    a.plot(*cxy(d["mu"][:25]), color="w", lw=1.0)
    a.plot(*cxy(d["mu"][25:]), color="w", lw=1.2)
    er = ext(d["mu"][25:]) / ext(d["mu"][:25])
    if iv is not None:
        for bnd in iv:
            a.plot(*cxy(d["mu"][25:], max(bnd, 0) / er), color="w", lw=0.9, ls=(0, (3, 2)))
    R = float((Phi @ d["mu"][:25]).max()) * 1.45
    a.set_xlim(256 - R, 256 + R)
    a.set_ylim(256 + R, 256 - R)
    a.set_xticks([])
    a.set_yticks([])
    k_in = int(np.sum((d["rx"] >= iv[0]) & (d["rx"] <= iv[1]))) if iv is not None else len(d["rx"])
    a.set_title(
        title
        + f"\nraters {d['rx'].min():.2f} to {d['rx'].max():.2f}\ninterval "
        + (f"[{iv[0]:.2f}, {iv[1]:.2f}]" if iv is not None else "unbounded")
        + f", {k_in}/{len(d['rx'])} inside",
        fontsize=6,
    )


fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.9))
Q1 = {}
for r_, (dn, ds) in enumerate([("RIGA", TST), ("Chaksu", CT_)]):
    ivs = RES2[dn]["TK"]["iv"]
    order = sorted(range(len(ds)), key=lambda i: sp_(ds[i]))

    def full(i):
        iv = ivs[i]
        return iv is None or np.all((ds[i]["rx"] >= iv[0]) & (ds[i]["rx"] <= iv[1]))

    easy, hard = [], []
    for i in order:
        if len(easy) < 2 and full(i) and window_inside(dn, ds[i]):
            easy.append(i)
        if len(easy) == 2:
            break
    for i in order[::-1]:
        if len(hard) < 2 and window_inside(dn, ds[i]):
            hard.append(i)
        if len(hard) == 2:
            break
    if len(easy) < 2 or len(hard) < 2:
        say(
            f"   WARNING {dn}: fewer than 2 crops passed the border check; filling from the ranked list without it (check these for text)"
        )
        easy += [i for i in order if full(i) and i not in easy][: 2 - len(easy)]
        hard += [i for i in order[::-1] if i not in hard][: 2 - len(hard)]
    Q1[dn] = dict(easy=[ds[i]["key"] for i in easy], hard=[ds[i]["key"] for i in hard])
    say(
        f"   {dn}: easy (lowest rater spread, all raters inside) {Q1[dn]['easy']}; hard (highest rater spread) {Q1[dn]['hard']}"
    )
    for c_, (i, lab) in enumerate([(easy[0], "easy"), (easy[1], "easy"), (hard[0], "hard"), (hard[1], "hard")]):
        draw_case(axs[r_, c_], dn, ds[i], ivs[i], f"{'RIGA' if dn == 'RIGA' else 'Chákṣu ' + ds[i]['src']}, {lab}")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(SUP, f"figS8_interval_bands.{e}"), dpi=300, bbox_inches="tight")
plt.close(fig)
X["Q1"] = Q1
say(
    "   white solid: predicted disc and cup; white dashed: cup scaled to the interval's lower and upper vCDR; black: each rater's cup. "
    "Look at the figure once before use to confirm no text is visible."
)

# ---------------- Q2. worst polar U-Net cases on Remidio and the seam
say("\nQ2. WORST POLAR U-NET vCDR ERRORS ON REMIDIO, AND THE POLAR SEAM")
P, items, imgs = PRED["Chaksu"]
rows = SEG["Chaksu"]
rem = [i for i, r in enumerate(rows["polar_ens"]) if r["group"] == "Remidio"]
w4 = sorted(rem, key=lambda i: -rows["polar_ens"][i]["vcdr_ae"])[:4]


@torch.no_grad()
def polar_space(im, models):
    x = np.stack([to_polar(np.ascontiguousarray(im[:, :, c])) for c in range(3)], -1)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)[None].to(DEVICE)
    return np.mean([torch.sigmoid(mm(t).float()).cpu().numpy()[0] for mm in models], 0)


seam = []
for i in range(len(imgs)):
    pp = polar_space(imgs[i], UN["polar"])
    cm = (pp[1] > 0.5).any(0)
    rmax = int(np.where(cm)[0].max()) + 1 if cm.any() else 0
    seam.append(float(np.abs(pp[1, 0, : max(rmax, 1)] - pp[1, -1, : max(rmax, 1)]).mean()))
seam = np.array(seam)
si = int(np.argmax(seam))
say(
    f"   seam score (mean |cup probability at 0 deg - at 360 deg| along the radius inside the cup), polar U-Net, {len(seam)} Chaksu images: "
    f"median {np.median(seam):.3f}, 90th {np.percentile(seam, 90):.3f}, max {seam.max():.3f} ({items[si][1]})"
)
fig, axs = plt.subplots(1, 5, figsize=(7.4, 1.9))


def mcont(a, m, **kw):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        c = c.reshape(-1, 2)
        a.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), **kw)


for a, i in zip(axs[:4], w4):
    g = imgs[i][:, :, 1]
    lo_, hi_ = np.percentile(g, [1, 99])
    a.imshow(np.clip((g - lo_) / (hi_ - lo_ + 1e-6), 0, 1), cmap="gray")
    _, case, gd, gc, _ = items[i]
    for m_ in (gd, gc):
        mcont(a, m_, color="w", lw=0.9, ls=(0, (3, 2)))
    for m_ in P["polar_ens"][i]:
        mcont(a, m_, color="w", lw=1.0)
    for m_ in P["harmonic_ens"][i]:
        mcont(a, m_, color="k", lw=0.9)
    ys, xs = np.where(gd)
    c0 = (xs.mean(), ys.mean())
    R = 0.75 * max(np.ptp(xs), np.ptp(ys))
    a.set_xlim(c0[0] - R, c0[0] + R)
    a.set_ylim(c0[1] + R, c0[1] - R)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(
        f"ref {rows['polar_ens'][i]['v_ref']:.2f}, U-Net {rows['polar_ens'][i]['v_pred']:.2f},\nharmonic {rows['harmonic_ens'][i]['v_pred']:.2f}",
        fontsize=6.5,
    )
    say(
        f"   worst: {case}: reference {rows['polar_ens'][i]['v_ref']:.3f}, polar U-Net {rows['polar_ens'][i]['v_pred']:.3f}, harmonic {rows['harmonic_ens'][i]['v_pred']:.3f}, "
        f"window inside photograph: {window_inside('Chaksu', [d for d in CT_ if d['key'] == 'CHAKSU/' + case][0])}"
    )
pc = to_cartesian(np.ascontiguousarray(polar_space(imgs[si], UN["polar"])[1]))
axs[4].imshow(pc, cmap="gray", vmin=0, vmax=1)
mcont(axs[4], P["polar_ens"][si][1], color="w", lw=0.8)
axs[4].plot([256, 512], [256, 256], color="0.5", lw=0.7, ls=(0, (2, 2)))
axs[4].set_xlim(96, 416)
axs[4].set_ylim(416, 96)
axs[4].set_xticks([])
axs[4].set_yticks([])
axs[4].set_title(f"cup probability, polar U-Net;\ndashed: 0/360 deg seam", fontsize=6.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(SUP, f"figS9_worst_unet_remidio_and_seam.{e}"), dpi=300, bbox_inches="tight")
plt.close(fig)
say("   panels 1-4: reference dashed white, polar U-Net solid white, harmonic black; panel 5: seam image")
X["Q2"] = dict(
    worst=[items[i][1] for i in w4],
    seam_case=items[si][1],
    seam_median=float(np.median(seam)),
    seam_max=float(seam.max()),
)

# ---------------- I. implementation facts
say("\nI. IMPLEMENTATION FACTS (read from files)")
IMP = {}
for k in range(1, 6):
    p = os.path.join(BASE, "outputs", "phase2_ens", f"seed{k}", "best_dice.pt")
    c = torch.load(p, map_location="cpu", weights_only=False)
    IMP[f"harmonic seed{k}"] = dict(epoch=c.get("epoch"), config=c.get("config"))
    for fn in ["config.json", "history.json"]:
        q = os.path.join(os.path.dirname(p), fn)
        if os.path.exists(q) and fn == "config.json":
            IMP[f"harmonic seed{k}"]["config_json"] = json.load(open(q))
        if os.path.exists(q) and fn == "history.json":
            IMP[f"harmonic seed{k}"]["epochs_run"] = len(json.load(open(q)))
    say(
        f"   harmonic seed {k}: selected epoch {IMP[f'harmonic seed{k}']['epoch']}, epochs run {IMP[f'harmonic seed{k}'].get('epochs_run')}, "
        f"config {json.dumps(IMP[f'harmonic seed{k}'].get('config_json') or IMP[f'harmonic seed{k}']['config'], default=str)}"
    )
for m in ["plain", "attention", "deepres", "polar"]:
    ep = []
    for k in range(1, 6):
        lg = json.load(open(os.path.join(BASE, "outputs", "phase8_v2", m, f"seed{k}_log.json")))
        ep.append(1 + int(np.argmax([(r["val_disc"] + r["val_cup"]) / 2 for r in lg])))
    IMP[m] = ep
    say(
        f"   {m} U-Net: selected epochs per seed {ep} (of 30); parameters {sum(q.numel() for q in UN[m][0].parameters())/1e6:.1f}M"
    )
if FAIR:
    ep = [
        1
        + int(
            np.argmax(
                [
                    (r["val_disc"] + r["val_cup"]) / 2
                    for r in json.load(open(os.path.join(FAIRDIR, f"seed{k}_log.json")))
                ]
            )
        )
        for k in FS
        if os.path.exists(os.path.join(FAIRDIR, f"seed{k}_log.json"))
    ]
    IMP["polarfair"] = ep
    say(f"   fair polar U-Net: selected epochs per seed {ep} (of 30)")
say(f"   harmonic regressor parameters {sum(q.numel() for q in members[0][0].parameters())/1e6:.1f}M")
X["I"] = IMP

# ---------------- S6. recalibrated on Chaksu training images (last: it changes the fitted values, restored afterwards)
say(
    "\nS6. SECONDARY: every configuration recalibrated on Chaksu TRAIN images (proportional by camera, same draw as before), scored on Chaksu test"
)
for k_ in stats:
    stats[k_] = [] if isinstance(stats[k_], list) else 0
CC = build(cal_items)
if FAIR:
    for d in CC:
        d["f_ens"] = fair_v(chp(d))
say(
    f"   recalibration images: {len(cal_items)} drawn, {len(CC)} usable ({sum(len(d['rx']) for d in CC)} rater pairs); camera counts "
    + str({c: sum(d["src"] == c for d in CC) for c in ["Bosch", "Forus", "Remidio"]})
)
fit0 = dict(fit)
fc = fit_all(CC, LVL, Z90)
fit.update(fc)
say(
    "   fits on Chaksu train: "
    + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(float(v), 5)) for k, v in fc.items()})
)
S6 = {}
T6 = [
    "| Configuration | Zero-shot coverage | Zero-shot width | Recalibrated coverage [95% CI] | Recalibrated width |",
    "| --- | --- | --- | --- | --- |",
]
for k in METH:
    iv = iv_at(k, fc[k], CT_, Z90)
    hn, w = score(iv, CT_)
    c_, w_ = boot(hn, w)
    z0 = RES2["Chaksu"][k]
    S6[k] = dict(
        cov=cov(hn),
        cov_ci=c_.tolist(),
        width=float(np.nanmean(w)),
        per_cam={c: cov(hn[DS_IDX["Chaksu"][c]]) for c in ["Bosch", "Forus", "Remidio"]},
    )
    say(
        f"   {nm_of(k):46s} zero-shot {z0['cov']*100:5.1f}% / {z0['width']:.4f}  ->  recalibrated {cov(hn)*100:5.1f}% [{c_[0]*100:.1f}, {c_[1]*100:.1f}] / {np.nanmean(w):.4f};  "
        + ", ".join(f"{c} {v*100:.1f}%" for c, v in S6[k]["per_cam"].items())
    )
    T6.append(
        f"| {nm_of(k)} | {z0['cov']*100:.1f}% | {z0['width']:.3f} | {cov(hn)*100:.1f}% [{c_[0]*100:.1f}, {c_[1]*100:.1f}] | {np.nanmean(w):.3f} |"
    )
fit.clear()
fit.update(fit0)
open(os.path.join(SUP, "tableS6_chaksu_recalibrated.md"), "w").write("\n".join(T6))
X["S6"] = dict(fits=fc, results=S6, n=len(CC))

# ---------------- S1 table (written last so the json is complete)
T7 = [
    "| Configuration | Dataset | " + " | ".join(f"{int(l*100)}%" for l in LEVELS) + " | Width at 90% |",
    "| --- | --- | " + " | ".join(["---"] * len(LEVELS)) + " | --- |",
]
for k in METH:
    for dn in ["RIGA", "Chaksu"]:
        T7.append(
            f"| {nm_of(k)} | {dn} | "
            + " | ".join(f"{x*100:.1f}" for x in RELA[dn][k])
            + f" | {WIDA[dn][k][LEVELS.index(0.9)]:.3f} |"
        )
open(os.path.join(SUP, "tableS1_reliability_all_methods.md"), "w").write("\n".join(T7))
json.dump(X, open(os.path.join(OUT, "extras_results.json"), "w"), indent=1, default=jd)
json.dump({k: v for k, v in PT.items()}, open(os.path.join(OUT, "paired_all.json"), "w"), indent=1, default=jd)
say(
    f"\nSaved {OUT} (tables: table1_segmentation_all.md, table2_intervals_all.md; supplement/: figures S1, S4, S8, S9 and tables S1-S3, S6, S7)"
)
say("EXTRAS DONE")
