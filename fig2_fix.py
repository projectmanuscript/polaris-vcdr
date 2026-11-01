"""
Review checks and regeneration of Fig. 2. Re-executes final_run.py in memory (outputs/fig2_fix/); nothing refitted.
  1. 95% CI (images resampled, 5000) for the vCDR-MAE difference, pretrained ResNet-34 U-Net minus POLARIS, Chaksu and RIGA.
  2. Paired cluster bootstrap TK minus K (one-change vCDR-level tau row), Chaksu and RIGA.
  5. RIGA counts: exclusion registry contents, raw images on disk, processed records, images in the split.
  E. Fig. 2(e) source check: every expert's raw cup and disc mask for the high-spread Chaksu image.
  F. Fig. 2 regenerated: (a) legend "Conformal polar U-Net", 81.1% label inside the axes; (b-e) chosen only among images where
     every expert outline is valid (6 on RIGA, 5 on Chaksu), zoom 1.9x the disc diameter so the label clears the disc.
Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 fig2_fix.py 2>&1 | tee outputs/fig2_fix_console.log
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
exec(fr.replace(a_, "'\"outputs\", \"fig2_fix\")'"))
ev = open("eval_rn34_and_figs.py").read()
s0, s1 = "# ---------------- shared drawing helpers", "# ---------------- 4. Figure 2(b-e)"
assert s0 in ev and s1 in ev and ev.count("half = 0.8 * max(") == 1
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
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)
CM = 1 / 2.54
sp = lambda d: np.std(d["rx"], ddof=1)
exec(ev.split(s0)[1].split(s1)[0].replace("half = 0.8 * max(", "half = 0.95 * max("))  # xy, rgb_crop, annulus, panel
FD = os.path.join(OUT, "figures")
os.makedirs(FD, exist_ok=True)
say("\n==================== FIG 2 FIX AND CHECKS ====================")


def ci_mean(x, B=5000, seed=7):
    x = np.asarray(x)
    I = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.percentile(x[I].mean(1), [2.5, 97.5])


def signflip(x, B=10000, seed=2026):
    x = np.asarray(x)
    null = np.abs((np.random.default_rng(seed).choice([-1.0, 1.0], (B, len(x))) * x).mean(1))
    return (np.sum(null >= abs(x.mean())) + 1) / (B + 1)


# ---------- 1. ResNet-34 U-Net minus POLARIS, vCDR MAE
from rn34_unet import ResUNet34

RM = []
for k in range(1, 6):
    m_ = ResUNet34().to(DEVICE)
    m_.load_state_dict(
        torch.load(
            os.path.join(BASE, "outputs", "phase8_rn34", f"seed{k}_best.pt"), map_location=DEVICE, weights_only=True
        )
    )
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


say("\n1. vCDR MAE, pretrained ResNet-34 U-Net minus POLARIS (images resampled, 5000; sign-flip p, 10000):")
for dname in ["Chaksu", "RIGA"]:
    P, items, imgs = PRED[dname]
    R = rn_masks(imgs)
    ae_r = np.array([abs(vext(*R[i]) - vext(gd, gc)) for i, (_, _, gd, gc, _) in enumerate(items)])
    ae_h = np.array([r["vcdr_ae"] for r in SEG[dname]["harmonic_ens"]])
    dl = ae_r - ae_h
    c = ci_mean(dl)
    say(
        f"   {dname:6s} (n={len(dl)}): U-Net {ae_r.mean():.4f}, POLARIS {ae_h.mean():.4f}, difference {dl.mean():+.4f} [{c[0]:+.4f}, {c[1]:+.4f}], p = {signflip(dl):.4f}"
    )

# ---------- 2. TK minus K
say(
    "\n2. Coverage, TK (final) minus K (vCDR-level tau, constant rater covariance; the one-change row), paired cluster bootstrap:"
)
for dname, n in [("Chaksu", len(CT_)), ("RIGA", len(TST))]:
    r = paired(dname, "TK", "K", list(range(n)))
    say(
        f"   {dname:6s}: TK {RES2[dname]['TK']['cov']*100:.1f}% vs K {RES2[dname]['K']['cov']*100:.1f}%: {r['dcov']*100:+.1f} points [{r['dcov_ci'][0]*100:+.1f}, {r['dcov_ci'][1]*100:+.1f}], "
        f"p = {r['p_cov']:.4f}; width {r['dw']:+.4f} [{r['dw_ci'][0]:+.4f}, {r['dw_ci'][1]:+.4f}]"
    )

# ---------- 5. RIGA counts
say("\n5. RIGA COUNTS")
reg = json.load(open(os.path.join(BASE, "exclusion_registry.json")))
for k, v in reg.items():
    n = len(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
    say(
        f"   registry '{k}': {n} entries"
        + (f"  {list(v)[:10]}" if isinstance(v, (list, dict)) and len(v) <= 10 else "")
    )
raw = 0
for root, _, fs in os.walk(os.path.join(BASE, "RIGA")):
    raw += sum(1 for f in fs if re.search(r"prime\.(jpe?g|tiff?)$", f, re.I))
proc = len([f for f in os.listdir(os.path.join(PROCESSED_DIR, "RIGA")) if f.endswith(".json")])
spl = json.load(open(os.path.join(BASE, "project_split_v2.json")))
say(
    f"   raw RIGA 'prime' images on disk: {raw}; processed records: {proc}; RIGA images in the split: {sum(k.startswith('RIGA/') for v in spl.values() for k in v)}"
)
say(f"   per split: " + ", ".join(f"{s} {sum(k.startswith('RIGA/') for k in spl[s])}" for s in spl))

# ---------- E. Fig. 2(e) source masks
say("\nE. FIG 2(e) SOURCE MASK CHECK")
d_e = max([d for d in CT_ if len(d["rx"]) == 5], key=sp)
_, cam, kk = d_e["key"].split("/", 1)[1].split("_", 2)
ent = MAN["Test"][cam][kk]


def bm(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    m = m.max(axis=2) if m.ndim == 3 else m
    return m > (m.max() / 2 if m.max() else 0)


def vx(d, c):
    rd, rc = np.where(d.any(1))[0], np.where(c.any(1))[0]
    return (np.ptp(rc) + 1 if len(rc) else 0) / (np.ptp(rd) + 1)


masks = {e: (bm(r["disc"]), bm(r["cup"]), r["disc"], r["cup"]) for e, r in sorted(ent["raters"].items())}
say(f"   image {d_e['key']} (expert SD {sp(d_e):.3f}); raters in manifest: {list(masks)}")
for e, (dm, cm, pd, pc) in masks.items():
    others = [masks[o][0] for o in masks if o != e]
    say(
        f"   E{e}: cup vCDR {vx(dm, cm):.2f}, cup/disc area {cm.sum()/max(dm.sum(), 1):.2f}, Dice(own cup, own disc) {dice(cm, dm):.3f}, "
        f"Dice(own cup, other experts' discs) mean {np.mean([dice(cm, o) for o in others]):.3f}; cup file: {os.path.relpath(pc, BASE)}"
    )
fig, axs = plt.subplots(2, len(masks), figsize=(1.2 * len(masks), 2.4))
ys, xs = np.where(np.any([m[0] for m in masks.values()], axis=0))
y0, y1, x0, x1 = ys.min() - 40, ys.max() + 40, xs.min() - 40, xs.max() + 40
for j, (e, (dm, cm, _, _)) in enumerate(masks.items()):
    axs[0, j].imshow(dm[y0:y1, x0:x1], cmap="gray")
    axs[0, j].set_title(f"E{e} disc", fontsize=7)
    axs[1, j].imshow(cm[y0:y1, x0:x1], cmap="gray")
    axs[1, j].set_title(f"E{e} cup", fontsize=7)
for a in axs.ravel():
    a.set_axis_off()
fig.savefig(os.path.join(OUT, "fig2e_source_masks.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
say(f"   raw masks drawn side by side: {os.path.join(OUT, 'fig2e_source_masks.png')}")

# ---------- F. Fig. 2 regenerated
say("\nF. FIG 2 REGENERATED")
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
i9 = LEVELS.index(0.9)
a.axvline(90, color="0.85", lw=0.6, zorder=0)
a.plot([45, 101], [45, 101], color="0.55", lw=0.8, ls=(0, (3, 2)), zorder=1)
a.fill_between(L, lo_, hi_, color=BLUE, alpha=0.18, lw=0, zorder=2)
a.plot(
    L, CFc * 100, ls=(0, (4, 1.5)), marker="s", ms=2.6, color=ORANGE, lw=1.2, label="Conformal polar U-Net", zorder=3
)
a.plot(L, TKc * 100, ls="-", marker="o", ms=2.8, color=BLUE, lw=1.2, label="POLARIS", zorder=4)
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
    xytext=(2, -3),
    textcoords="offset points",
    ha="left",
    va="top",
    color=ORANGE,
)
a.set_xlim(47, 100.5)
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
a.legend(h_[::-1], l_[::-1], loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.0, labelspacing=0.2)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(FD, f"fig2a_reliability_chaksu.{e}"), dpi=600, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
say("   (a) saved")
for tag, (dn, ds, nfull) in zip(
    ["b", "c", "d", "e"], [("RIGA", TST, 6), ("RIGA", TST, 6), ("Chaksu", CT_, 5), ("Chaksu", CT_, 5)]
):
    pool = [d for d in ds if len(d["rx"]) == nfull]
    d = min(pool, key=sp) if tag in ("b", "d") else max(pool, key=sp)
    info = panel(d, os.path.join(FD, f"fig2{tag}_{dn}_{'low' if tag in ('b', 'd') else 'high'}_spread.png"))
    say(
        f"   ({tag}) {info['key']}: chosen among {len(pool)} images with all {nfull} expert outlines valid; expert SD {info['spread']:.3f}, "
        f"vCDR {info['vcdr']:.2f} [{info['interval'][0]:.2f}, {info['interval'][1]:.2f}], {info['inside']}/{info['n']} inside"
    )
say(f"\nSaved {FD}\nFIG2 FIX DONE")
