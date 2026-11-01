"""
Block 3: calibration analysis on the corrected crops (preprocessed_data_v3). ALL settings below are fixed before the run;
each test number is computed once.
Fixed settings
  - Primary target: INDIVIDUAL-RATER coverage at nominal 90%. A rater's vCDR is the vertical EXTENT ratio of that rater's
    own contour (max y - min y of cup / of disc, from the rater's M=12 contour), which does not depend on any centre.
    Secondary: the same hits with the chord definition, and rater-mean coverage.
  - Model: 5-seed harmonic ensemble (phase2_ens), mean prediction mu, model variance e = per-coefficient variance across seeds.
  - Rater dispersion S = diag(dispersion head) + rater-style covariance (Phase 3, fitted on TRAIN images only).
  - Every scalar / quantile is fitted on CALIB (123 images with >= 2 valid raters) and reported once on TEST.
  - Intervals for harmonic configs: Fieller interval for the chord vCDR of mu under covariance Sigma.
Configurations
  E   POLARIS: tau * (diag(e) + S), one scalar fitted at the vCDR level                  <- the method
  B   model only: tau * diag(e)
  K   E with a CONSTANT rater covariance: tau * (diag(e) + diag(mean calib head) + style_cov)
  C   BW transport ablation: tau fitted per coefficient on diag(e), lambda in [0, 3] fitted at the vCDR level
  T   per-coefficient temperature: E's covariance, tau fitted so per-COEFFICIENT rater coverage = nominal
  CFh fixed-width split conformal around the harmonic ensemble's vCDR (weights 1/raters per image)
  CFu fixed-width split conformal around the best U-Net ensemble's vCDR (best = highest mean VALIDATION Dice in training logs)
  UE  best U-Net 5-seed spread: mean seed vCDR +/- tau * z * sd across seeds
Reported: marginal coverage + mean width with cluster-bootstrap 95% CIs (images resampled, 5000), paired width differences,
conditional coverage (rater-spread tertile with calib cut-points; vCDR level <0.5 / 0.5-0.7 / >0.7; sub-source),
5-fold cross-fitting on CALIB of E vs C (and B, K), reliability 50-95% for E, CFh, T, kappa for the 0.6 crossing,
chord vs extent agreement on the rater contours, and the width diagnostic (why adding S narrows the interval).
Outputs: outputs/calib_v4/
"""

import os, re, json, numpy as np, torch, cv2
from scipy.stats import norm as _N

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
OLD, NEW = 'os.path.join(BASE, "preprocessed_data")', 'os.path.join(BASE, "preprocessed_data_v3")'
s34 = open("phase34_v3.py").read()
EV = "exec(open('eval_testset_all_models.py').read().split('results = {}')[0])"
assert EV in s34 and '"outputs", "phase34_v3")' in s34
s34 = s34.replace(
    EV, "exec(open('eval_testset_all_models.py').read().split('results = {}')[0].replace('%s', '%s'))" % (OLD, NEW)
)
s34 = s34.replace('"outputs", "phase34_v3")', '"outputs", "calib_v4")')
exec(s34.split("# ---------------------------------------------------------------- 3. Phase 4")[0])
assert PREPROC_DIR.endswith("preprocessed_data_v3"), PREPROC_DIR
say(f"\n==== calib_v4 on {PREPROC_DIR} ====")

# ---------------- ensemble predictions, targets
ENS = [os.path.join(BASE, "outputs", "phase2_ens", f"seed{k}", "best_dice.pt") for k in range(1, 6)]
members = []
for p in ENS:
    c = torch.load(p, map_location=DEVICE, weights_only=False)
    m = HarmonicRegressor().to(DEVICE)
    m.load_state_dict(c["model_state"])
    m.eval()
    members.append((m, np.asarray(c["target_mean"]), np.asarray(c["target_std"])))


def npy(key):
    ds, case = key.split("/", 1)
    return os.path.join(PREPROC_DIR, ds, case + "_normalized.npy")


@torch.no_grad()
def ens_pred(key):
    x = torch.from_numpy(np.load(npy(key)).astype(np.float32)).permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(x, size=256, mode="bilinear", align_corners=False).to(DEVICE)
    return np.stack([mm(x).cpu().numpy()[0] * s_ + t_ for mm, t_, s_ in members])


SIN = np.sin(np.linspace(-np.pi, np.pi, 360, endpoint=False))  # drawing grid: index n <-> angle -pi + n degrees


def ext(u25):
    y = (Phi @ u25) * SIN
    return y.max() - y.min()


for s in ["calib", "test"]:
    for d in D[s]:
        P = ens_pred(d["key"])
        d["mu"] = P.mean(0)
        d["e"] = P.var(0, ddof=1)
        d["S"] = np.diag(d["v0"]) + style_cov
        d["rx"] = np.array([ext(u[25:]) / ext(u[:25]) for u in d["U"]])  # rater extent vCDR (primary)
        d["rv"] = (d["U"][:, 25:] @ phi_v) / (d["U"][:, :25] @ phi_v)  # rater chord vCDR (secondary)
        d["src"] = (lambda c: "Magrabia" if c.startswith("Magrabi") else re.match(r"[A-Za-z]+", c).group(0))(
            d["key"].split("/", 1)[1]
        )
floor = 1e-3 * np.median(np.concatenate([d["e"] for s in ["calib", "test"] for d in D[s]]))
for s in ["calib", "test"]:
    for d in D[s]:
        d["e"] = np.maximum(d["e"], floor)
Sconst = np.diag(np.mean([d["v0"] for d in D["calib"]], 0)) + style_cov
CAL, TST = D["calib"], D["test"]
say(
    f"CALIB {len(CAL)} images / {sum(len(d['rx']) for d in CAL)} rater pairs; TEST {len(TST)} images / {sum(len(d['rx']) for d in TST)} pairs"
)


# ---------------- Fieller in closed form (checked against the project's fieller below)
def vq(Sig):
    return np.array([phi_v @ Sig[25:, 25:] @ phi_v, phi_v @ Sig[:25, :25] @ phi_v, phi_v @ Sig[25:, :25] @ phi_v])


def ivq(a, b, v, tau, z):
    vaa, vbb, vab = tau * v
    A = b * b - z * z * vbb
    if A <= 0:
        return None
    Bq = a * b - z * z * vab
    C = a * a - z * z * vaa
    disc = Bq * Bq - A * C
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return ((Bq - r) / A, (Bq + r) / A)


for d in CAL + TST:
    d["a"], d["b"] = phi_v @ d["mu"][25:], phi_v @ d["mu"][:25]
    d["V"] = {"E": vq(np.diag(d["e"]) + d["S"]), "B": vq(np.diag(d["e"])), "K": vq(np.diag(d["e"]) + Sconst)}
chk = []
for d in CAL[:10]:
    mine = ivq(d["a"], d["b"], d["V"]["E"], 1.0, Z)
    theirs = fieller(*dist(d["mu"], np.diag(d["e"]) + d["S"]))
    if mine and theirs:
        chk.append(max(abs(mine[0] - theirs[0]), abs(mine[1] - theirs[1])))
say(
    f"Fieller check vs project implementation (10 calib images): max abs difference {max(chk) if chk else float('nan'):.2e}"
)


# ---------------- scoring helpers
def score(ivs, ds, key="rx"):
    hn, w = [], []
    for iv, d in zip(ivs, ds):
        t = d[key]
        if iv is None:
            hn.append((len(t), len(t)))
            w.append(np.nan)
        else:
            hn.append((int(np.sum((t >= iv[0]) & (t <= iv[1]))), len(t)))
            w.append(iv[1] - iv[0])
    return np.array(hn), np.array(w)


def mean_hits(ivs, ds, key="rx"):
    return np.array([1 if iv is None else int(iv[0] <= np.mean(d[key]) <= iv[1]) for iv, d in zip(ivs, ds)])


def cov(hn):
    return hn[:, 0].sum() / hn[:, 1].sum()


BR = np.random.default_rng(2026)


def boot(hn, w, B=5000):
    n = len(hn)
    I = BR.integers(0, n, (B, n))
    c = hn[I, 0].sum(1) / hn[I, 1].sum(1)
    ww = np.array([np.nanmean(w[i]) for i in I])
    return np.percentile(c, [2.5, 97.5]), np.percentile(ww, [2.5, 97.5])


TAUS = np.logspace(-3, 6, 1801)
pick = lambda grid, covs, lvl: grid[int(np.argmin([abs(c - lvl) + 1e-9 * i for i, c in enumerate(covs)]))]


def fit_tau(ds, V, lvl, z, key="rx"):
    return float(
        pick(TAUS, [cov(score([ivq(d["a"], d["b"], d["V"][V], t, z) for d in ds], ds, key)[0]) for t in TAUS], lvl)
    )


def ivs_of(ds, V, tau, z):
    return [ivq(d["a"], d["b"], d["V"][V], tau, z) for d in ds]


def wq(vals, w, q):
    o = np.argsort(vals)
    v, c = vals[o], np.cumsum(w[o]) / w.sum()
    return float(v[min(np.searchsorted(c, q), len(v) - 1)])


def conformal_q(ds, point, lvl, key="rx"):
    vals = np.concatenate([np.abs(d[key] - point(d)) for d in ds])
    w = np.concatenate([np.full(len(d[key]), 1 / len(d[key])) for d in ds])
    return wq(vals, w, min(1.0, lvl * (1 + 1 / len(ds))))


def coef_tau(ds, lvl, z, var):
    fr = [np.mean([np.mean(np.abs(d["U"] - d["mu"][None]) <= z * np.sqrt(t * var(d))[None]) for d in ds]) for t in TAUS]
    return float(pick(TAUS, fr, lvl))


LAMS = [round(x, 3) for x in np.arange(0, 3.0001, 0.005)]


def fit_C(ds, lvl, z):
    tC = coef_tau(ds, lvl, z, lambda d: d["e"])
    for d in ds:
        d["_T"] = transport_T(tC * d["e"], d["S"])
    covs = [
        cov(score([ivq(d["a"], d["b"], vq(sigma_lam(tC * d["e"], d["_T"], l)), 1.0, z) for d in ds], ds)[0])
        for l in LAMS
    ]
    return tC, float(pick(LAMS, covs, lvl))


def ivs_C(ds, tC, lam, z):
    out = []
    for d in ds:
        T = transport_T(tC * d["e"], d["S"])
        out.append(ivq(d["a"], d["b"], vq(sigma_lam(tC * d["e"], T, lam)), 1.0, z))
    return out


# ---------------- best U-Net (chosen on VALIDATION from training logs), its vCDR on calib + test
CL = {"plain": PlainUNet, "attention": AttentionUNet, "deepres": DeepResUNet, "polar": PlainUNet}
vs = {}
for m in CL:
    v = []
    for k in range(1, 6):
        lg = json.load(open(os.path.join(BASE, "outputs", "phase8_v2", m, f"seed{k}_log.json")))
        v.append(max((r["val_disc"] + r["val_cup"]) / 2 for r in lg))
    vs[m] = float(np.mean(v))
BEST = max(vs, key=vs.get)
say(
    "Validation Dice (mean of seeds' best epochs): "
    + ", ".join(f"{m} {v*100:.2f}" for m, v in vs.items())
    + f"  -> best U-Net = {BEST}"
)


def lcc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return m if n <= 2 else lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def vext(dm, cm):
    rd = np.where(dm.any(1))[0]
    rc = np.where(cm.any(1))[0]
    return np.nan if len(rd) == 0 else (np.ptp(rc) + 1 if len(rc) else 0) / (np.ptp(rd) + 1)


seeds = []
for k in range(1, 6):
    mm = CL[BEST]().to(DEVICE)
    mm.load_state_dict(
        torch.load(
            os.path.join(BASE, "outputs", "phase8_v2", BEST, f"seed{k}_best.pt"), map_location=DEVICE, weights_only=True
        )
    )
    mm.eval()
    seeds.append(mm)


@torch.no_grad()
def unet_vcdr(key):
    im = np.load(npy(key)).astype(np.float32)
    if BEST == "polar":
        im = np.stack([to_polar(np.ascontiguousarray(im[:, :, c])) for c in range(3)], -1)
    x = torch.from_numpy(im).permute(2, 0, 1)[None].to(DEVICE)
    probs = [torch.sigmoid(m(x).float()).cpu().numpy()[0] for m in seeds]

    def masks(p):
        if BEST == "polar":
            p = np.stack([to_cartesian(np.ascontiguousarray(p[c])) for c in range(2)])
        return lcc(p[0] > 0.5), lcc(p[1] > 0.5)

    per = [vext(*masks(p)) for p in probs]
    ens = vext(*masks(np.mean(probs, 0)))
    return ens, np.array(per)


for d in CAL + TST:
    d["u_ens"], d["u_seeds"] = unet_vcdr(d["key"])
    d["u_sd"] = np.std(d["u_seeds"], ddof=1)
sfloor = max(1e-4, 1e-3 * np.median([d["u_sd"] for d in CAL]))
for d in CAL + TST:
    d["u_sd"] = max(d["u_sd"], sfloor)

# ---------------- main fit on CALIB at 90%, single report on TEST
LVL, Z90 = 0.90, _N.ppf(0.95)
fit = {}
fit["E"] = fit_tau(CAL, "E", LVL, Z90)
fit["B"] = fit_tau(CAL, "B", LVL, Z90)
fit["K"] = fit_tau(CAL, "K", LVL, Z90)
fit["C"] = fit_C(CAL, LVL, Z90)
fit["T"] = coef_tau(CAL, LVL, Z90, lambda d: d["e"] + np.diag(d["S"]))
hpt = lambda d: d["a"] / d["b"]
fit["CFh"] = conformal_q(CAL, hpt, LVL)
fit["CFu"] = conformal_q(CAL, lambda d: d["u_ens"], LVL)
UTAU = np.logspace(-2, 3, 2001)
fit["UE"] = float(
    pick(
        UTAU,
        [
            cov(
                score(
                    [
                        (d["u_seeds"].mean() - t * Z90 * d["u_sd"], d["u_seeds"].mean() + t * Z90 * d["u_sd"])
                        for d in CAL
                    ],
                    CAL,
                )[0]
            )
            for t in UTAU
        ],
        LVL,
    )
)
say(
    "\nCALIB fits at 90%: "
    + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 5)) for k, v in fit.items()})
)


def test_ivs(name, ds, z=Z90):
    if name in ("E", "B", "K"):
        return ivs_of(ds, name, fit[name], z)
    if name == "T":
        return ivs_of(ds, "E", fit["T"], z)
    if name == "C":
        return ivs_C(ds, *fit["C"], z)
    if name == "CFh":
        return [(hpt(d) - fit["CFh"], hpt(d) + fit["CFh"]) for d in ds]
    if name == "CFu":
        return [(d["u_ens"] - fit["CFu"], d["u_ens"] + fit["CFu"]) for d in ds]
    if name == "UE":
        return [
            (d["u_seeds"].mean() - fit["UE"] * Z90 * d["u_sd"], d["u_seeds"].mean() + fit["UE"] * Z90 * d["u_sd"])
            for d in ds
        ]


NAMES = {
    "E": "POLARIS (E)",
    "B": "model only (B)",
    "K": "E, constant rater cov (K)",
    "C": "BW transport (C)",
    "T": "per-coef temperature (T)",
    "CFh": "conformal, harmonic (CFh)",
    "CFu": f"conformal, {BEST} U-Net (CFu)",
    "UE": f"{BEST} U-Net seed spread (UE)",
}
RES, IV = {}, {}
say(
    f"\nTEST, nominal 90%, individual-rater coverage (extent definition), {len(TST)} images. CIs: cluster bootstrap over images."
)
say(
    f"  {'config':34s} {'rater cov':>9s} {'95% CI':>15s} {'width':>7s} {'95% CI':>15s} {'chord-def cov':>13s} {'mean-target':>11s} {'unbounded':>9s}"
)
for k in NAMES:
    iv = test_ivs(k, TST)
    IV[k] = iv
    hn, w = score(iv, TST)
    hc, _ = score(iv, TST, "rv")
    ci_c, ci_w = boot(hn, w)
    RES[k] = dict(
        cov=cov(hn),
        cov_ci=ci_c.tolist(),
        width=float(np.nanmean(w)),
        width_ci=ci_w.tolist(),
        cov_chord=cov(hc),
        cov_mean=float(mean_hits(iv, TST).mean()),
        unbounded=int(sum(x is None for x in iv)),
        widths=w.tolist(),
        hn=hn.tolist(),
    )
    r = RES[k]
    say(
        f"  {NAMES[k]:34s} {r['cov']*100:8.1f}% [{r['cov_ci'][0]*100:5.1f},{r['cov_ci'][1]*100:5.1f}] {r['width']:7.4f} [{r['width_ci'][0]:.4f},{r['width_ci'][1]:.4f}] "
        f"{r['cov_chord']*100:12.1f}% {r['cov_mean']*100:10.1f}% {r['unbounded']:9d}"
    )
say("\nPaired width difference, E minus each config (images where both bounded; bootstrap 95% CI over images):")
wE = np.array(RES["E"]["widths"])
for k in NAMES:
    if k == "E":
        continue
    wk = np.array(RES[k]["widths"])
    ok = ~(np.isnan(wE) | np.isnan(wk))
    dlt = wE[ok] - wk[ok]
    bt = np.array([dlt[BR.integers(0, len(dlt), len(dlt))].mean() for _ in range(5000)])
    say(
        f"  E - {k:4s}: {dlt.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]  (E narrower on {np.mean(dlt < 0)*100:.0f}% of images)"
    )

# ---------------- conditional coverage
sp = lambda d: np.std(d["rx"], ddof=1)
cuts = np.percentile([sp(d) for d in CAL], [100 / 3, 200 / 3])
groups = {
    "spread low": lambda d: sp(d) <= cuts[0],
    "spread mid": lambda d: cuts[0] < sp(d) <= cuts[1],
    "spread high": lambda d: sp(d) > cuts[1],
    "vCDR <0.5": lambda d: np.mean(d["rx"]) < 0.5,
    "vCDR 0.5-0.7": lambda d: 0.5 <= np.mean(d["rx"]) <= 0.7,
    "vCDR >0.7": lambda d: np.mean(d["rx"]) > 0.7,
}
for s in sorted({d["src"] for d in TST}):
    groups["source " + s] = (lambda s: lambda d: d["src"] == s)(s)
say(f"\nCONDITIONAL individual-rater coverage on TEST (spread tertile cut-points from CALIB: {cuts.round(4).tolist()})")
say("  " + f"{'group':18s}{'n':>4s}" + "".join(f"{k:>8s}" for k in NAMES))
COND = {}
for gname, g in groups.items():
    idx = [i for i, d in enumerate(TST) if g(d)]
    if not idx:
        continue
    row = {k: cov(np.array(RES[k]["hn"])[idx]) for k in NAMES}
    COND[gname] = dict(n=len(idx), **row)
    say("  " + f"{gname:18s}{len(idx):4d}" + "".join(f"{row[k]*100:7.1f}%" for k in NAMES))
say(
    "  (max minus min across groups, per config: "
    + ", ".join(f"{k} {(max(COND[g][k] for g in COND) - min(COND[g][k] for g in COND))*100:.1f}" for k in NAMES)
    + " points)"
)

# ---------------- cross-fitting on CALIB (5 folds), E vs C (B, K for reference)
say("\nCROSS-FITTING on CALIB, 5 folds (fit on 4/5, score on 1/5), nominal 90%:")
perm = np.random.default_rng(7).permutation(len(CAL))
folds = np.array_split(perm, 5)
cf = {k: {"hn": [None] * len(CAL), "w": [None] * len(CAL)} for k in ["E", "B", "K", "C"]}
for fo in folds:
    tr = [CAL[i] for i in range(len(CAL)) if i not in set(fo)]
    te = [CAL[i] for i in fo]
    for k in ["E", "B", "K"]:
        hn, w = score(ivs_of(te, k, fit_tau(tr, k, LVL, Z90), Z90), te)
        for j, i in enumerate(fo):
            cf[k]["hn"][i], cf[k]["w"][i] = hn[j], w[j]
    tC, lam = fit_C(tr, LVL, Z90)
    hn, w = score(ivs_C(te, tC, lam, Z90), te)
    for j, i in enumerate(fo):
        cf["C"]["hn"][i], cf["C"]["w"][i] = hn[j], w[j]
for k in cf:
    cf[k]["hn"] = np.array(cf[k]["hn"])
    cf[k]["w"] = np.array(cf[k]["w"], float)
for k in cf:
    say(f"  {k}: rater coverage {cov(cf[k]['hn'])*100:.1f}%, mean width {np.nanmean(cf[k]['w']):.4f}")
ok = ~(np.isnan(cf["C"]["w"]) | np.isnan(cf["E"]["w"]))
dlt = cf["C"]["w"][ok] - cf["E"]["w"][ok]
null = np.abs((np.random.default_rng(1).choice([-1.0, 1.0], (10000, len(dlt))) * dlt).mean(1))
say(
    f"  paired width C - E: {dlt.mean():+.4f} (C narrower on {np.mean(dlt < 0)*100:.0f}% of images), sign-flip p = {(np.sum(null >= abs(dlt.mean())) + 1)/10001:.4f}"
)

# ---------------- reliability 50-95%
say("\nRELIABILITY (refit on CALIB at each nominal level; TEST individual-rater coverage):")
LEVELS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
REL = {"E": [], "CFh": [], "T": []}
for lv in LEVELS:
    z = _N.ppf(0.5 + lv / 2)
    REL["E"].append(cov(score(ivs_of(TST, "E", fit_tau(CAL, "E", lv, z), z), TST)[0]))
    q = conformal_q(CAL, hpt, lv)
    REL["CFh"].append(cov(score([(hpt(d) - q, hpt(d) + q) for d in TST], TST)[0]))
    REL["T"].append(cov(score(ivs_of(TST, "E", coef_tau(CAL, lv, z, lambda d: d["e"] + np.diag(d["S"])), z), TST)[0]))
say("  nominal " + " ".join(f"{l*100:5.0f}" for l in LEVELS))
for k, v in REL.items():
    say(f"  {k:7s} " + " ".join(f"{x*100:5.1f}" for x in v))

# ---------------- 0.6 crossing agreement (E at 90%)
A = np.array([iv is None or (iv[0] < 0.6 < iv[1]) for iv in IV["E"]])
Bv = np.array([d["rx"].min() < 0.6 <= d["rx"].max() for d in TST])
po = np.mean(A == Bv)
pe = A.mean() * Bv.mean() + (1 - A.mean()) * (1 - Bv.mean())
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
say(
    f"\n0.6 CROSSING (TEST): interval crosses 0.6 on {A.sum()} images, raters split around 0.6 on {Bv.sum()}; both {np.sum(A & Bv)}, "
    f"interval only {np.sum(A & ~Bv)}, raters only {np.sum(~A & Bv)}, neither {np.sum(~A & ~Bv)}; agreement {po*100:.1f}%, Cohen kappa {kappa:.3f}"
)

# ---------------- chord vs extent on the rater contours
rx = np.concatenate([d["rx"] for d in CAL + TST])
rv = np.concatenate([d["rv"] for d in CAL + TST])
df = rv - rx
say(
    f"\nCHORD vs EXTENT vCDR on {len(rx)} rater contours (calib + test): Pearson r {np.corrcoef(rx, rv)[0, 1]:.4f}, "
    f"mean difference (chord - extent) {df.mean():+.4f}, mean absolute difference {np.abs(df).mean():.4f}, 95% limits of agreement [{df.mean() - 1.96*df.std():+.4f}, {df.mean() + 1.96*df.std():+.4f}]"
)


# ---------------- width diagnostic
def sd_v(d, Sig):
    g = np.concatenate([-d["a"] / d["b"] ** 2 * phi_v, phi_v / d["b"]])
    return float(np.sqrt(g @ Sig @ g))


sde = np.array([sd_v(d, np.diag(d["e"])) for d in CAL])
sds = np.array([sd_v(d, d["S"]) for d in CAL])
share = sds**2 / (sde**2 + sds**2)
say(
    f"\nWIDTH DIAGNOSTIC (CALIB, delta-method sd of the chord vCDR): model-only sd median {np.median(sde):.4f}, rater-dispersion sd median {np.median(sds):.4f}; "
    f"rater term share of the summed variance: median {np.median(share)*100:.1f}%. Fitted scalars: tau_B {fit['B']:.4g} (model only) vs tau_E {fit['E']:.4g}. "
    f"Spearman(model-only sd, observed rater spread) {__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(sde, [sp(d) for d in CAL])[0]:.3f}; "
    f"Spearman(summed sd, observed rater spread) {__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(np.sqrt(sde**2 + sds**2), [sp(d) for d in CAL])[0]:.3f}"
)

json.dump(
    {
        "fit": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fit.items()},
        "best_unet": BEST,
        "val_scores": vs,
        "test": {k: {kk: vv for kk, vv in v.items() if kk not in ("widths", "hn")} for k, v in RES.items()},
        "conditional": COND,
        "crossfit": {k: dict(cov=cov(v["hn"]), width=float(np.nanmean(v["w"]))) for k, v in cf.items()},
        "reliability": {"levels": LEVELS, **REL},
        "kappa_0.6": kappa,
        "spread_cuts": cuts.tolist(),
    },
    open(os.path.join(OUT, "calib_v4_results.json"), "w"),
    indent=1,
    default=float,
)
say("\nSaved", OUT)
