"""
Evaluates the direct-regression controls (train_direct.py) on the paper's test sets. Re-executes final_run.py in memory
(outputs/eval_direct/); POLARIS and its numbers are not refitted.
  1. RIGA test (94) and Chaksu test (336), same inputs and references as Table 1:
       radii ensemble : disc Dice, cup Dice, vCDR MAE, bias (masks drawn straight from the 360 predicted radii)
       vcdr ensemble  : vCDR MAE, bias (it predicts no outline, so no Dice)
     Paired against POLARIS (harmonic ensemble): mean difference, 95% CI (images resampled, 5000), sign-flip p (10000),
     Holm over the two controls per metric. Chaksu also per camera.
  2. Intervals: neither control gives an expert-level interval. For a like-for-like comparison, each gets a split-conformal
     band fitted on the same 117 RIGA calibration images (as CFh / CFu), scored on RIGA and Chaksu test; paired TK minus it.
Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 eval_direct.py 2>&1 | tee outputs/eval_direct_console.log
"""

import os

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
exec(fr.replace(a_, "'\"outputs\", \"eval_direct\")'"))
import torch.nn as nn

say("\n==================== DIRECT-REGRESSION CONTROLS ====================")
GRIDd = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def radii_mask(r, size=512, c=(256, 256)):
    pts = np.stack([c[0] + r * np.cos(GRIDd), c[1] + r * np.sin(GRIDd)], 1).astype(np.int32).reshape(-1, 1, 2)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


CTRL_M = {}
for v in ["radii", "vcdr"]:
    ms = []
    for k in range(1, 6):
        c = torch.load(
            os.path.join(BASE, "outputs", "phase9_direct", v, f"seed{k}_best.pt"),
            map_location=DEVICE,
            weights_only=False,
        )
        m = HarmonicRegressor()
        m.head = nn.Linear(m.head.in_features, c["out_dim"])
        m.load_state_dict(c["model_state"])
        m.to(DEVICE).eval()
        ms.append((m, np.asarray(c["norm_mean"]), np.asarray(c["norm_std"]), c["img"]))
        say(f"   {v} seed {k}: epoch {c['epoch']}, val metric {c['val_metric']:.4f}")
    CTRL_M[v] = ms


@torch.no_grad()
def ctrl_out(v, imgs):
    out = []
    for im in imgs:
        x = torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32)).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, size=CTRL_M[v][0][3], mode="bilinear", align_corners=False).to(DEVICE)
        out.append(np.mean([m(x).cpu().numpy()[0] * s_ + t_ for m, t_, s_, _ in CTRL_M[v]], 0))
    return out


def ci_mean(x, B=5000, seed=7):
    x = np.asarray(x)
    I = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.percentile(x[I].mean(1), [2.5, 97.5])


def signflip(x, B=10000, seed=2026):
    x = np.asarray(x)
    null = np.abs((np.random.default_rng(seed).choice([-1.0, 1.0], (B, len(x))) * x).mean(1))
    return (np.sum(null >= abs(x.mean())) + 1) / (B + 1)


# ---------- 1. segmentation / vCDR
for dname in ["RIGA", "Chaksu"]:
    P, items, imgs = PRED[dname]
    H = SEG[dname]["harmonic_ens"]
    Rr, Rv = ctrl_out("radii", imgs), ctrl_out("vcdr", imgs)
    rows = {"radii": [], "vcdr": []}
    for i, (_, case, gd, gc, grp) in enumerate(items):
        gv = vext(gd, gc)
        dm, cm = radii_mask(Rr[i][:360]), radii_mask(Rr[i][360:])
        pv = vext(dm, cm)
        rows["radii"].append(
            dict(
                group=grp,
                disc=float(dice(dm, gd)),
                cup=float(dice(cm, gc)),
                vcdr_ae=float(abs(pv - gv)),
                bias=float(pv - gv),
                out=float(np.logical_and(cm, ~dm).sum() > 0),
            )
        )
        pv2 = float(Rv[i][0])
        rows["vcdr"].append(dict(group=grp, vcdr_ae=abs(pv2 - gv), bias=pv2 - gv))
    groups = {"all": list(range(len(H)))}
    if dname == "Chaksu":
        groups.update({c: [i for i, r in enumerate(H) if r["group"] == c] for c in ["Bosch", "Forus", "Remidio"]})
    for g, idx in groups.items():
        f = lambda rr, k: float(np.mean([rr[i][k] for i in idx]))
        say(f"\n1. {dname} / {g} (n={len(idx)})")
        say(
            f"   POLARIS (harmonic)   disc {f(H, 'disc')*100:.2f}  cup {f(H, 'cup')*100:.2f}  vCDR MAE {f(H, 'vcdr_ae'):.4f}  bias {f(H, 'bias'):+.4f}"
        )
        say(
            f"   direct, 360 radii    disc {f(rows['radii'], 'disc')*100:.2f}  cup {f(rows['radii'], 'cup')*100:.2f}  vCDR MAE {f(rows['radii'], 'vcdr_ae'):.4f}  "
            f"bias {f(rows['radii'], 'bias'):+.4f}  cup outside disc {f(rows['radii'], 'out')*100:.1f}%"
        )
        say(f"   direct, vCDR value   vCDR MAE {f(rows['vcdr'], 'vcdr_ae'):.4f}  bias {f(rows['vcdr'], 'bias'):+.4f}")
        tests = []
        for v, mets in [("radii", ["disc", "cup", "vcdr_ae"]), ("vcdr", ["vcdr_ae"])]:
            for met in mets:
                dl = np.array([rows[v][i][met] - H[i][met] for i in idx])
                tests.append([v, met, dl.mean(), ci_mean(dl), signflip(dl)])
        for met in ["vcdr_ae"]:  # Holm over the two controls for vCDR MAE
            T_ = sorted([t for t in tests if t[1] == met], key=lambda t: t[4])
            run = 0.0
            for rk, t in enumerate(T_):
                run = max(run, min(1.0, (len(T_) - rk) * t[4]))
                t.append(run)
        for t in tests:
            sc = 1 if t[1] == "vcdr_ae" else 100
            dg = 4 if t[1] == "vcdr_ae" else 2
            say(
                f"   {t[0]:5s} minus POLARIS, {t[1]:7s}: {t[2]*sc:+.{dg}f} [{t[3][0]*sc:+.{dg}f}, {t[3][1]*sc:+.{dg}f}], p = {t[4]:.4f}"
                + (f" (Holm over 2 controls {t[5]:.4f})" if len(t) > 5 else "")
            )

# ---------- 2. conformal bands around the controls' vCDR
say("\n2. SPLIT-CONFORMAL BANDS AROUND THE CONTROLS (fitted on RIGA calibration, nominal 90%)")
CH_NPY = lambda d: os.path.join(CH_DIR, d["key"].split("/", 1)[1] + "_normalized.npy")
for ds_, getp in [(CAL + TST, lambda d: npy(d["key"])), (CT_, CH_NPY)]:
    ims = [np.load(getp(d)).astype(np.float32) for d in ds_]
    for d, r, v in zip(ds_, ctrl_out("radii", ims), ctrl_out("vcdr", ims)):
        d["dr"] = vext(radii_mask(r[:360]), radii_mask(r[360:]))
        d["dv"] = float(v[0])
for key, nm in [("dr", "CFdr: conformal, direct 360-radii vCDR"), ("dv", "CFdv: conformal, direct vCDR value")]:
    q = conformal_q(CAL, lambda d: d[key], LVL)
    say(f"   {nm}: half-width {q:.5f}")
    for dname, ds in [("RIGA", TST), ("Chaksu", CT_)]:
        iv = [(d[key] - q, d[key] + q) for d in ds]
        hn, w = score(iv, ds)
        ci_c, _ = boot(hn, w)
        hT = RES2[dname]["TK"]["hn"]
        I = np.random.default_rng(99).integers(0, len(ds), (5000, len(ds)))
        dc = hT[I, 0].sum(1) / hT[I, 1].sum(1) - hn[I, 0].sum(1) / hn[I, 1].sum(1)
        pv = float(min(1.0, 2 * min(np.mean(dc <= 0), np.mean(dc >= 0))))
        say(
            f"   {dname:6s}: coverage {cov(hn)*100:.1f}% [{ci_c[0]*100:.1f}, {ci_c[1]*100:.1f}], width {2*q:.4f}; "
            f"TK minus it {(cov(hT) - cov(hn))*100:+.1f} points [{np.percentile(dc, 2.5)*100:+.1f}, {np.percentile(dc, 97.5)*100:+.1f}], p = {pv:.4f}"
        )
say("\nEVAL DIRECT DONE")
