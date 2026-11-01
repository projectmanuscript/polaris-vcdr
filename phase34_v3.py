"""
Phase 3 + Phase 4 on the clean split (project_split_v2.json), using the new Phase 2 model (outputs/phase2_v3b/best_dice.pt).

Phase 3 (TRAIN images only): per-rater coefficients -> rater style (SVD, q=4) -> dispersion head (Gaussian NLL).
Phase 4 (CALIB images only): fit tau and lambda.  TEST images: coverage reported once, for configurations fixed in advance.

Configurations reported on test (decided before any test result is seen):
  A  tau fitted at coefficient level, lambda = 0             (temperature only, original order)
  B  tau fitted at vCDR level, lambda = 0                     (temperature scaling done at the clinical output: the strongest simple baseline)
  C  tau as in A, lambda* on [0, 3] (extrapolation allowed)   PRIMARY POLARIS configuration
  D  tau as in A, lambda* restricted to [0, 1]               (secondary)
Coverage "hit" (primary): Fieller 90% interval contains the vCDR of the rater-mean outline (original definition).
Secondary: fraction of individual rater vCDRs inside the interval.
Everything is written to outputs/phase34_v3/.
"""

import os, json, re, time
import numpy as np, torch, torch.nn as nn, cv2

exec(open('eval_testset_all_models.py').read().split('results = {}')[0])  # helpers + HarmonicRegressor + DEVICE + paths

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
OUT = os.path.join(BASE, "outputs", "phase34_v3")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(BASE, "outputs", "phase2_v3b", "best_dice.pt")
split = json.load(open(os.path.join(BASE, "project_split_v2.json")))
reg = json.load(open(os.path.join(BASE, "exclusion_registry.json")))
_un = reg.get("extraction_unstable", [])
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
unstable = {norm(e) for e in (_un.keys() if isinstance(_un, dict) else _un)}
Phi = build_harmonic_basis(12, N=360)
phi_v = Phi[90] + Phi[270]
Z = 1.6448536269514722
torch.manual_seed(0)
rng = np.random.default_rng(0)
LOG = open(os.path.join(OUT, "run_log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# ---------------------------------------------------------------- 0. self-checks of the maths
def transport_T(s0, S):  # s0: diagonal of Sigma0 (vector); returns the BW map T
    h = np.sqrt(s0)
    inner = (h[:, None] * S) * h[None, :]
    w, Q = np.linalg.eigh(inner)
    w = np.clip(w, 0, None)
    return (1 / h)[:, None] * ((Q * np.sqrt(w)) @ Q.T) * (1 / h)[None, :]


def sigma_lam(s0, T, lam):
    M = (1 - lam) * np.eye(len(s0)) + lam * T
    return (M * s0[None, :]) @ M.T


def dist(mu, Sl):
    mc, md = phi_v @ mu[25:], phi_v @ mu[:25]
    vc, vd = phi_v @ Sl[25:, 25:] @ phi_v, phi_v @ Sl[:25, :25] @ phi_v
    cdd = phi_v @ Sl[25:, :25] @ phi_v
    sc, sd = np.sqrt(vc), np.sqrt(vd)
    return mc, md, sc, sd, cdd / (sc * sd)


def fieller(mc, md, sc, sd, rho):
    A = md**2 - Z**2 * sd**2
    B = -2 * mc * md + 2 * Z**2 * rho * sc * sd
    C = mc**2 - Z**2 * sc**2
    if A <= 0:
        return None
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return ((-B - r) / (2 * A), (-B + r) / (2 * A))


s0t = rng.uniform(0.5, 2, 6)
Bm = rng.normal(size=(6, 6))
St = Bm @ Bm.T + np.diag(s0t)
Tt = transport_T(s0t, St)
ok = (
    np.allclose(sigma_lam(s0t, Tt, 0), np.diag(s0t))
    and np.allclose(sigma_lam(s0t, Tt, 1), St, atol=1e-8)
    and np.linalg.eigvalsh(sigma_lam(s0t, Tt, 2.0)).min() > -1e-9
)
say("Check 1 (transport: lambda=0 -> Sigma0, lambda=1 -> S, PSD at lambda=2):", "PASS" if ok else "FAIL")
assert ok


# ---------------------------------------------------------------- 1. per-rater coefficients
def valid_contour(crop, c=(256, 256)):
    if len(crop) < 10 or not check_star_shaped(crop, c):
        return False
    a = np.sort(np.arctan2(crop[:, 1] - c[1], crop[:, 0] - c[0]))
    if np.degrees(np.diff(np.concatenate([a, [a[0] + 2 * np.pi]])).max()) > 25:
        return False
    x, y = crop[:, 0], crop[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [crop.astype(np.int32).reshape(-1, 1, 2)], 1)
    return 0.85 <= area / max(m.sum(), 1) <= 1.15


n_unstable_hit = 0


def rater_coeffs(key):
    global n_unstable_hit
    ds, case = key.split("/", 1)
    rec = json.load(open(os.path.join(PROCESSED_DIR, ds, case + ".json")))
    tr = json.load(open(os.path.join(PREPROC_DIR, ds, case + "_transform.json")))
    rows, ids = [], []
    for rid, data in rec["raters"].items():
        dig = re.sub(r"\D", "", str(rid))
        if norm(case.replace("_", "/", 1) + "_rater" + dig) in unstable:
            n_unstable_hit += 1
            continue
        u = []
        for sk in ("disc_contour", "cup_contour"):
            cpts = data.get(sk)
            if cpts is None:
                break
            crop = map_to_crop_space(cpts, tr["x_min"], tr["y_min"], tr["scale"])
            if not valid_contour(crop):
                break
            u.append(Phi.T @ contour_to_radial(crop)[1])
        if len(u) == 2:
            rows.append(np.concatenate(u))
            ids.append(dig)
    return np.array(rows), ids


ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = HarmonicRegressor().to(DEVICE)
model.load_state_dict(ck["model_state"])
model.eval()
tm, ts = np.asarray(ck["target_mean"]), np.asarray(ck["target_std"])
say(f"Mean model: {CKPT} (epoch {ck.get('epoch')})")


def feat_mu(key):
    ds, case = key.split("/", 1)
    x = torch.from_numpy(np.load(os.path.join(PREPROC_DIR, ds, case + "_normalized.npy")).astype(np.float32)).permute(
        2, 0, 1
    )[None]
    x = torch.nn.functional.interpolate(x, size=256, mode="bilinear", align_corners=False).to(DEVICE)
    with torch.no_grad():
        f = model.features(x).flatten(1)
        mu = model.head(f)
    return f.cpu().numpy()[0], mu.cpu().numpy()[0] * ts + tm


D, t0 = {}, time.time()
for s in ["train", "val", "calib", "test"]:
    D[s], skipped = [], 0
    for k in split[s]:
        if not k.startswith("RIGA/"):
            continue
        U, ids = rater_coeffs(k)
        if len(U) < 2:
            skipped += 1
            continue
        f, mu = feat_mu(k)
        D[s].append(dict(key=k, U=U, ids=ids, ubar=U.mean(0), feat=f, mu=mu))
    say(
        f"{s:6s}: {len(D[s])} RIGA images with >= 2 valid raters (skipped {skipped}), mean raters/image {np.mean([len(d['ids']) for d in D[s]]):.2f}"
    )
say(
    f"Unstable rater/image pairs matched from the registry: {n_unstable_hit} (registry has {len(unstable)})  [{time.time()-t0:.0f}s]"
)
if n_unstable_hit == 0:
    say("WARNING: no registry entries matched; the key format differs; check the registry keys.")

# Check 2: rebuilt rater-mean equals the stored Phase 2 training target
dd, dc = [], []
for d in D["train"]:
    ds, case = d["key"].split("/", 1)
    tgt = np.load(os.path.join(BASE, "training_targets_v2", f"{ds}_{case}.npz"))["target"]
    dd.append(dice(coeffs_to_mask(d["ubar"][:25], Phi) > 0, coeffs_to_mask(tgt[:25], Phi) > 0))
    dc.append(dice(coeffs_to_mask(d["ubar"][25:], Phi) > 0, coeffs_to_mask(tgt[25:], Phi) > 0))
say(
    f"Check 2 (rebuilt rater mean vs stored Phase 2 target, train): disc Dice mean {np.mean(dd)*100:.2f} min {np.min(dd)*100:.2f} | "
    f"cup mean {np.mean(dc)*100:.2f} min {np.min(dc)*100:.2f}  (expect ~100; lower means the rater filters differ slightly)"
)

# ---------------------------------------------------------------- 2. Phase 3 on TRAIN
dev = np.concatenate([d["U"] - d["ubar"] for d in D["train"]])
C = np.corrcoef(dev, rowvar=False)
off = np.abs(C[~np.eye(50, dtype=bool)]).mean()
say(
    f"Phase 3: pooled deviations n={len(dev)}, mean |off-diagonal corr| (harmonic) = {off:.4f}  (earlier result: 0.0268)"
)
rater_ids = sorted({r for d in D["train"] for r in d["ids"]})
styles = np.array(
    [np.mean([d["U"][d["ids"].index(r)] - d["ubar"] for d in D["train"] if r in d["ids"]], 0) for r in rater_ids]
)
Uu, Ss, Vt = np.linalg.svd(styles - styles.mean(0), full_matrices=False)
q = 4
V = Vt[:q].T
E = Uu[:, :q] * Ss[:q]
style_cov = V @ np.cov(E, rowvar=False) @ V.T
say(
    f"Rater style: raters {rater_ids}, singular values {np.round(Ss, 2)}, top-{q} share {np.sum(Ss[:q]**2)/np.sum(Ss**2)*100:.1f}% (5 dof only with 6 raters)"
)
X, R = [], []
for d in D["train"]:
    for j, r in enumerate(d["ids"]):
        X.append(d["feat"])
        R.append(d["U"][j] - d["ubar"] - styles[rater_ids.index(r)])
X, R = np.array(X, np.float32), np.array(R, np.float32)
resid_std = R.std(0) + 1e-6


class DispersionHead(nn.Module):  # identical to the original Phase 3 head
    def __init__(self, feat_dim=512, out_dim=50):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


head = DispersionHead()
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(R / resid_std)), batch_size=32, shuffle=True
)
for ep in range(40):
    head.train()
    tot = 0.0
    for xb, rb in dl:
        opt.zero_grad()
        lv = head(xb)
        var = torch.exp(lv).clamp(min=1e-4)
        loss = (0.5 * (torch.log(var) + rb**2 / var)).mean()
        loss.backward()
        opt.step()
        tot += loss.item()
    if ep % 10 == 0 or ep == 39:
        say(f"  dispersion head epoch {ep+1}/40: NLL {tot/len(dl):.4f}")
head.eval()
for s in ["val", "calib", "test"]:
    for d in D[s]:
        with torch.no_grad():
            d["v0"] = np.exp(head(torch.tensor(d["feat"][None])).numpy()[0]) * resid_std**2
pred = [d["v0"].sum() for d in D["val"]]
obs = [
    np.mean(np.sum((d["U"] - d["ubar"] - np.array([styles[rater_ids.index(r)] for r in d["ids"]])) ** 2, 1))
    for d in D["val"]
]
rk = lambda a: np.argsort(np.argsort(a))
say(
    f"Per-image spread on VAL: Pearson {np.corrcoef(pred, obs)[0, 1]:.3f}, Spearman {np.corrcoef(rk(pred), rk(obs))[0, 1]:.3f}  (earlier: 0.18)"
)
torch.save(
    {
        "model_state": head.state_dict(),
        "resid_std": resid_std,
        "V": V,
        "rater_embeddings": E,
        "rater_ids": rater_ids,
        "styles": styles,
        "style_cov": style_cov,
        "mean_model": CKPT,
    },
    os.path.join(OUT, "phase3_dispersion_head_v3.pt"),
)


# ---------------------------------------------------------------- 3. Phase 4: fit on CALIB
def coef_cov(ds, tau):
    return float(np.mean([np.mean(np.abs(d["ubar"] - d["mu"]) <= Z * np.sqrt(tau * d["v0"])) for d in ds]))


def vcdr_eval(ds, tau, lams):
    res = {l: dict(hit=0, n=0, unb=0, w=[], rhit=0, rn=0) for l in lams}
    for d in ds:
        s0 = tau * d["v0"]
        T = transport_T(s0, np.diag(s0) + style_cov)
        tv = (phi_v @ d["ubar"][25:]) / (phi_v @ d["ubar"][:25])
        rv = (d["U"][:, 25:] @ phi_v) / (d["U"][:, :25] @ phi_v)
        for l in lams:
            iv = fieller(*dist(d["mu"], sigma_lam(s0, T, l)))
            r = res[l]
            r["n"] += 1
            r["rn"] += len(rv)
            if iv is None:
                r["unb"] += 1
                r["hit"] += 1
                r["rhit"] += len(rv)
                continue
            r["hit"] += int(iv[0] <= tv <= iv[1])
            r["w"].append(iv[1] - iv[0])
            r["rhit"] += int(np.sum((rv >= iv[0]) & (rv <= iv[1])))
    return {
        l: dict(
            cov=r["hit"] / r["n"],
            width=float(np.mean(r["w"])) if r["w"] else float("nan"),
            unbounded=r["unb"],
            n=r["n"],
            rater_cov=r["rhit"] / r["rn"],
        )
        for l, r in res.items()
    }


pick = lambda grid, covs: grid[int(np.argmin([abs(c - 0.90) + 1e-9 * i for i, c in enumerate(covs)]))]
taus = np.round(np.logspace(-2, 2, 81), 5)
tau_coef = float(pick(taus, [coef_cov(D["calib"], t) for t in taus]))
lams = [round(x, 2) for x in np.arange(0, 3.0001, 0.05)]
curve = vcdr_eval(D["calib"], tau_coef, lams)
lam_ext = float(pick(lams, [curve[l]["cov"] for l in lams]))
lam_01 = float(pick([l for l in lams if l <= 1], [curve[l]["cov"] for l in lams if l <= 1]))
tau_vcdr = float(pick(taus, [vcdr_eval(D["calib"], t, [0.0])[0.0]["cov"] for t in taus]))
say(
    f"\nCALIB fit (n={len(D['calib'])}): tau_coef={tau_coef:.4g} (coef coverage {coef_cov(D['calib'], tau_coef)*100:.1f}%), "
    f"lambda*[0,3]={lam_ext}, lambda*[0,1]={lam_01}, tau_vcdr={tau_vcdr:.4g}"
)
say("CALIB vCDR coverage vs lambda (tau_coef): " + "  ".join(f"{l}:{curve[l]['cov']*100:.1f}%" for l in lams[::4]))


# ---------------------------------------------------------------- 4. TEST: reported once
def wilson(k, n, zz=1.96):
    p = k / n
    c = (p + zz * zz / (2 * n)) / (1 + zz * zz / n)
    h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / (1 + zz * zz / n)
    return c - h, c + h


configs = {
    "A_temp_coef": (tau_coef, 0.0),
    "B_temp_vcdr": (tau_vcdr, 0.0),
    "C_polaris_lam_ext": (tau_coef, lam_ext),
    "D_polaris_lam_01": (tau_coef, lam_01),
}
say(f"\nTEST vCDR coverage, nominal 90% (n={len(D['test'])} RIGA test images):")
out = {
    "tau_coef": tau_coef,
    "tau_vcdr": tau_vcdr,
    "lambda_ext": lam_ext,
    "lambda_01": lam_01,
    "calib_curve": {str(l): curve[l] for l in lams},
    "test": {},
}
for name, (t, l) in configs.items():
    r = vcdr_eval(D["test"], t, [l])[l]
    k = round(r["cov"] * r["n"])
    lo, hi = wilson(k, r["n"])
    out["test"][name] = {**r, "tau": t, "lambda": l, "ci95": [lo, hi]}
    say(
        f"  {name:18s} tau={t:<8.4g} lambda={l:<5} coverage {r['cov']*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]  "
        f"mean width {r['width']:.4f}  per-rater coverage {r['rater_cov']*100:.1f}%  unbounded {r['unbounded']}"
    )
# Check 3: Fieller vs Monte Carlo on one test image, primary configuration
d = D["test"][0]
s0 = tau_coef * d["v0"]
Sl = sigma_lam(s0, transport_T(s0, np.diag(s0) + style_cov), lam_ext)
mc, md, sc, sd, rho = dist(d["mu"], Sl)
iv = fieller(mc, md, sc, sd, rho)
smp = rng.multivariate_normal([mc, md], [[sc * sc, rho * sc * sd], [rho * sc * sd, sd * sd]], 200000)
rt = smp[:, 0] / smp[:, 1]
say(
    f"Check 3 (Fieller vs Monte Carlo, 200k): Fieller ({iv[0]:.4f}, {iv[1]:.4f})  MC ({np.percentile(rt, 5):.4f}, {np.percentile(rt, 95):.4f})"
)
json.dump(out, open(os.path.join(OUT, "phase4_results_v3.json"), "w"), indent=1, default=float)
say("Saved", OUT)
