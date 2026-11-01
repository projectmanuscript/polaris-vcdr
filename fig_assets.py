"""
Exports every picture the method figure (Fig. 1) needs, for ONE test image, as separate files to drop into draw.io.
Nothing is trained or refitted beyond what final_run.py already does (same de-duplicated calibration, same TK fit).
Usage:  python3 fig_assets.py                      (picks the RIGA test image with median rater spread)
        CASE=RIGA/MagrabiaMale_image3 python3 fig_assets.py
Output: outputs/fig_assets/<case>/  PNG (300 dpi, transparent where it makes sense) + SVG for the plots, numbers.json
All pictures are black and white / grayscale.
"""

import os, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
fr = open("final_run.py").read()
a = "'\"outputs\", \"final_run\")'"
assert fr.count(a) == 1
fr = fr.replace(a, "'\"outputs\", \"fig_assets\")'")
cut = "# ---------------- Chaksu test set"
assert cut in fr
exec(fr.split(cut)[0])  # de-dup, refit, TK fit, get_ivs (about 5 min)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif", "svg.fonttype": "none"})
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)

sp = lambda d: np.std(d["rx"], ddof=1)
case = os.environ.get("CASE")
d = [x for x in TST if x["key"] == case][0] if case else sorted(TST, key=sp)[len(TST) // 2]
ds_, cs_ = d["key"].split("/", 1)
D_OUT = os.path.join(OUT, cs_)
os.makedirs(D_OUT, exist_ok=True)


def save(fig, name, svg=False):
    fig.savefig(os.path.join(D_OUT, name + ".png"), dpi=300, bbox_inches="tight", transparent=True)
    if svg:
        fig.savefig(os.path.join(D_OUT, name + ".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def xy(u25, f=1.0):
    r = (Phi @ u25) * f
    x, y = 256 + r * np.cos(GRID), 256 + r * np.sin(GRID)
    return np.append(x, x[0]), np.append(y, y[0])


def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


# 01 full photograph, and with the detected disc box and crop window
rec = json.load(open(os.path.join(PROCESSED_DIR, ds_, cs_ + ".json")))
tr = json.load(open(os.path.join(PREPROC_DIR, ds_, cs_ + "_transform.json")))
img = cv2.cvtColor(cv2.imread(resolve(rec["image_path"])), cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(D_OUT, "01_fundus_full.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
im2 = img.copy()
dia = tr["disc_diameter"]
cx, cy = tr["center_x"], tr["center_y"]
lw = max(2, img.shape[1] // 400)
cv2.rectangle(im2, (int(cx - dia / 2), int(cy - dia / 2)), (int(cx + dia / 2), int(cy + dia / 2)), (255, 255, 255), lw)
cv2.rectangle(
    im2,
    (int(cx - 1.3 * dia), int(cy - 1.3 * dia)),
    (int(cx + 1.3 * dia), int(cy + 1.3 * dia)),
    (255, 255, 255),
    lw,
    lineType=cv2.LINE_AA,
)
cv2.imwrite(os.path.join(D_OUT, "01b_fundus_box_and_crop_window.png"), cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))

# 02 crop as seen (RGB) and as the network sees it (CLAHE green, z-scored)
vj = os.path.join(PREPROC_DIR, ds_, cs_ + "_viewable.jpg")
crop = cv2.cvtColor(cv2.imread(vj), cv2.COLOR_BGR2RGB) if os.path.exists(vj) else None
if crop is not None:
    cv2.imwrite(os.path.join(D_OUT, "02_crop_rgb.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
g = np.load(npy(d["key"]))[:, :, 1]
lo_, hi_ = np.percentile(g, [1, 99])
G = np.clip((g - lo_) / (hi_ - lo_ + 1e-6), 0, 1)
cv2.imwrite(os.path.join(D_OUT, "02b_crop_network_input.png"), (G * 255).astype(np.uint8))


def crop_ax(fs=3.0):
    fig, a = plt.subplots(figsize=(fs, fs))
    a.imshow(G, cmap="gray")
    a.set_axis_off()
    R = float((Phi @ d["ubar"][:25]).max()) * 1.5
    a.set_xlim(256 - R, 256 + R)
    a.set_ylim(256 + R, 256 - R)
    return fig, a


# 03 rater outlines on the crop
fig, a = crop_ax()
for u in d["U"]:
    a.plot(*xy(u[:25]), color="w", lw=0.8)
    a.plot(*xy(u[25:]), color="k", lw=0.8)
save(fig, "03_rater_outlines")

# 04 radial function r(theta) of each rater and the rater mean (disc and cup)
fig, a = plt.subplots(figsize=(3.2, 2.0))
deg = np.degrees(GRID) % 360
o = np.argsort(deg)
for u in d["U"]:
    a.plot(deg[o], (Phi @ u[:25])[o], color="0.7", lw=0.6)
    a.plot(deg[o], (Phi @ u[25:])[o], color="0.7", lw=0.6, ls="--")
a.plot(deg[o], (Phi @ d["ubar"][:25])[o], "k-", lw=1.3, label="disc")
a.plot(deg[o], (Phi @ d["ubar"][25:])[o], "k--", lw=1.3, label="cup")
a.set_xlabel("angle θ (degrees)")
a.set_ylabel("radius r(θ) (px)")
a.set_xticks([0, 90, 180, 270, 360])
a.legend(frameon=False, fontsize=7)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
save(fig, "04_radial_function", svg=True)

# 05 harmonic basis functions (icon) m = 0..3
fig, axs = plt.subplots(4, 1, figsize=(1.4, 2.0), sharex=True)
t = np.linspace(0, 2 * np.pi, 400)
for m, a in enumerate(axs):
    a.plot(t, np.ones_like(t) if m == 0 else np.cos(m * t), "k-", lw=1.0)
    if m:
        a.plot(t, np.sin(m * t), color="0.55", lw=1.0)
    a.set_axis_off()
save(fig, "05_harmonic_basis_icon", svg=True)

# 06 coefficients: rater mean (target) and ensemble mean +/- sd across the 5 seeds
P = ens_pred(d["key"])
mu, sd = P.mean(0), P.std(0, ddof=1)
for nm, vals, err in [("06a_coefficients_rater_mean", d["ubar"], None), ("06b_coefficients_ensemble", mu, sd)]:
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.5), sharey=False)
    for a, sl, lab in [(axs[0], slice(1, 25), "disc"), (axs[1], slice(26, 50), "cup")]:
        k = np.arange(1, 25)
        a.bar(
            k,
            vals[sl],
            color="0.25",
            width=0.7,
            yerr=None if err is None else 1.96 * err[sl],
            error_kw=dict(lw=0.6, capsize=1),
        )
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"{lab} (24 of 25; m ≥ 1)", fontsize=7)
        a.set_xticks([1, 12, 24])
        a.tick_params(labelsize=6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    save(fig, nm, svg=True)

# 07 the 5 ensemble members' outlines on the crop
fig, a = crop_ax()
for p in P:
    a.plot(*xy(p[:25]), color="w", lw=0.7)
    a.plot(*xy(p[25:]), color="k", lw=0.7)
save(fig, "07_ensemble_member_outlines")

# 08 covariance pieces (50 x 50): ensemble variance, constant rater covariance, the final TK covariance
tau = fit["TK"]
Se = np.diag(d["e"])
Sk = tau * (Se + Sconst)
for nm, M_ in [("08a_cov_ensemble_variance", Se), ("08b_cov_constant_rater", Sconst), ("08c_cov_final_TK", Sk)]:
    fig, a = plt.subplots(figsize=(1.6, 1.6))
    v = np.abs(M_)
    a.imshow(v / v.max(), cmap="Greys", vmin=0, vmax=0.25)
    a.axhline(24.5, color="0.5", lw=0.4)
    a.axvline(24.5, color="0.5", lw=0.4)
    a.set_xticks([])
    a.set_yticks([])
    save(fig, nm)

# 09 output: predicted disc and cup + cup scaled to the interval ends (no text; add the numbers in draw.io)
iv = get_ivs("TK", [d])[0]
er = ext(d["mu"][25:]) / ext(d["mu"][:25])
pt = d["a"] / d["b"]
fig, a = crop_ax()
a.plot(*xy(d["mu"][:25]), color="w", lw=1.2)
a.plot(*xy(d["mu"][25:]), color="w", lw=1.4)
if iv is not None:
    for bnd in iv:
        a.plot(*xy(d["mu"][25:], max(bnd, 0) / er), color="w", lw=1.0, ls=(0, (3, 2)))
save(fig, "09_output_interval_on_crop")

# 10 vCDR number line: interval, point estimate, each rater's vCDR
fig, a = plt.subplots(figsize=(3.2, 0.8))
if iv is not None:
    a.plot([iv[0], iv[1]], [0, 0], "k-", lw=3, solid_capstyle="butt")
a.plot([pt], [0], "wo", mec="k", ms=6)
a.plot(d["rx"], np.full(len(d["rx"]), 0.18), "k|", ms=9)
a.set_xlim(0, 1)
a.set_ylim(-0.3, 0.35)
a.set_yticks([])
a.set_xlabel("vCDR")
for s in ("top", "right", "left"):
    a.spines[s].set_visible(False)
save(fig, "10_vcdr_interval_line", svg=True)

num = dict(
    case=d["key"],
    tau_TK=tau,
    vcdr_point_chord=float(pt),
    interval=None if iv is None else [float(iv[0]), float(iv[1])],
    rater_vcdr_extent=d["rx"].round(3).tolist(),
    n_raters=len(d["rx"]),
    rater_spread_sd=float(sp(d)),
    detector_box=[cx, cy, dia],
)
json.dump(num, open(os.path.join(D_OUT, "numbers.json"), "w"), indent=1)
say(f"\nFIGURE ASSETS for {d['key']} -> {D_OUT}\n" + json.dumps(num))
