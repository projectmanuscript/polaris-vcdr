"""
Follow-up to fig2_fix.py. Expert 5's cup mask for Chaksu Test_Remidio_img_3977 is a copy of the disc (Dice 0.998).
This script checks how often that happens and what it changes. Nothing is refitted; the final method is unchanged.
  A. RIGA: which of the 750 published images are missing from the 747 processed records.
  B. Every expert outline in RIGA calibration, RIGA test and Chaksu test: cup/disc area ratio from the expert's own harmonic
     outline. Rule fixed before looking: an outline is "cup = disc" if its cup area is >= 0.95 of its disc area.
     Prints the 15 largest ratios per set (so the gap between real large cups and copied discs is visible) and every flagged outline.
  C. Sensitivity: coverage of every method on Chaksu and RIGA test with the flagged expert outlines removed from scoring
     (same intervals), and the paired TK minus K / TK minus CFu bootstrap on the cleaned scoring.
  D. Fig. 2(e) re-chosen: highest expert spread among Chaksu images with all 5 outlines valid and none flagged.
Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 cupdisc_check.py 2>&1 | tee outputs/cupdisc_check_console.log
"""

import os, re, json

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)

# ---------- A. RIGA missing images (cheap, printed first)
print("\nA. RIGA: published images vs processed records", flush=True)
raw = {}
for root, _, fs in os.walk(os.path.join(BASE, "RIGA")):
    for f in fs:
        m = re.match(r"(image\d+)prime\.(jpe?g|tiff?)$", f, re.I)
        if m:
            raw.setdefault(os.path.relpath(root, os.path.join(BASE, "RIGA")), set()).add(m.group(1).lower())
for k in sorted(raw):
    print(f"   {k}: {len(raw[k])} 'prime' images", flush=True)
proc_dir = None
for cand in ["processed_data_v2", "processed_data"]:
    if os.path.isdir(os.path.join(BASE, cand, "RIGA")):
        proc_dir = os.path.join(BASE, cand, "RIGA")
        break
proc = sorted(f[:-5] for f in os.listdir(proc_dir) if f.endswith(".json"))
print(f"   processed records in {os.path.relpath(proc_dir, BASE)}: {len(proc)}", flush=True)
groups = {}
for p in proc:
    g, im = p.rsplit("_", 1)
    groups.setdefault(g, set()).add(im.lower())
for g in sorted(groups):
    print(f"   processed {g}: {len(groups[g])}", flush=True)
reg = json.load(open(os.path.join(BASE, "exclusion_registry.json")))
print(f"   registry 'misregistered_images': {reg.get('misregistered_images')}", flush=True)

# ---------- re-run the final pipeline in memory
fr = open("final_run.py").read()
a_ = "'\"outputs\", \"final_run\")'"
assert fr.count(a_) == 1
exec(fr.replace(a_, "'\"outputs\", \"cupdisc_check\")'"))
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
    }
)
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)
CM = 1 / 2.54
sp = lambda d: np.std(d["rx"], ddof=1)
exec(ev.split(s0)[1].split(s1)[0].replace("half = 0.8 * max(", "half = 0.95 * max("))  # xy, rgb_crop, annulus, panel
FD = os.path.join(OUT, "figures")
os.makedirs(FD, exist_ok=True)
say("\n==================== CUP = DISC CHECK ====================")


# ---------- B. flag outlines whose cup is the disc
def ratio(u):
    return float(np.mean((Phi @ u[25:]) ** 2) / np.mean((Phi @ u[:25]) ** 2))


FLAG = {}
for nm, ds in [("RIGA calib", CAL), ("RIGA test", TST), ("Chaksu test", CT_)]:
    rows = [(d["key"], j, ratio(u), float(d["rx"][j])) for d in ds for j, u in enumerate(d["U"])]
    fl = [r for r in rows if r[2] >= 0.95]
    FLAG[nm] = fl
    say(
        f"\nB. {nm}: {len(ds)} images, {len(rows)} expert outlines, flagged (cup area >= 0.95 x disc area): {len(fl)} outlines "
        f"in {len({r[0] for r in fl})} images"
    )
    say(
        "   15 largest cup/disc area ratios: "
        + ", ".join(
            f"{k.split('/')[-1]}#{j + 1} {r:.2f} (vCDR {x:.2f})"
            for k, j, r, x in sorted(rows, key=lambda r: -r[2])[:15]
        )
    )
    for k, j, r, x in fl:
        say(f"   flagged: {k} outline {j + 1}: area ratio {r:.3f}, vCDR {x:.2f}")


# ---------- C. sensitivity: drop flagged outlines from scoring only
def clean(ds, fl):
    bad = {}
    for k, j, _, _ in fl:
        bad.setdefault(k, set()).add(j)
    return [dict(d, rx=np.array([x for j, x in enumerate(d["rx"]) if j not in bad.get(d["key"], set())])) for d in ds]


def pboot(hA, hB, B=5000):
    I = np.random.default_rng(99).integers(0, len(hA), (B, len(hA)))
    dc = hA[I, 0].sum(1) / hA[I, 1].sum(1) - hB[I, 0].sum(1) / hB[I, 1].sum(1)
    return (
        cov(hA) - cov(hB),
        np.percentile(dc, [2.5, 97.5]),
        float(min(1.0, 2 * min(np.mean(dc <= 0), np.mean(dc >= 0)))),
    )


CLEAN = {}
for dname, ds, nm in [("RIGA", TST, "RIGA test"), ("Chaksu", CT_, "Chaksu test")]:
    dc_ = clean(ds, FLAG[nm])
    CLEAN[dname] = dc_
    H = {}
    say(
        f"\nC. {dname}: coverage at 90% with the {len(FLAG[nm])} flagged outlines removed from scoring "
        f"({sum(len(d['rx']) for d in ds)} -> {sum(len(d['rx']) for d in dc_)} expert outlines; intervals unchanged)"
    )
    for k in CONF + CTRL:
        H[k] = score(RES2[dname][k]["iv"], dc_)[0]
        say(f"   {k:12s} as reported {RES2[dname][k]['cov']*100:5.1f}%  ->  cleaned {cov(H[k])*100:5.1f}%")
    for a, b in [("TK", "K"), ("TK", "CFu"), ("TK", "E"), ("TK", "CFh")]:
        dlt, ci, p = pboot(H[a], H[b])
        say(f"   paired {a} - {b}: {dlt*100:+.1f} points [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}], p = {p:.4f}")
    say(
        f"   distance from 90 (cleaned): "
        + ", ".join(f"{k} {abs(cov(H[k])*100 - 90):.1f}" for k in sorted(CONF, key=lambda k: abs(cov(H[k]) - 0.9)))
    )

# ---------- D. Fig. 2(e) re-chosen without flagged images
say("\nD. FIG 2(e) RE-CHOSEN")
badk = {r[0] for r in FLAG["Chaksu test"]}
pool = [d for d in CT_ if len(d["rx"]) == 5 and d["key"] not in badk]
d = max(pool, key=sp)
info = panel(d, os.path.join(FD, "fig2e_Chaksu_high_spread.png"))
say(
    f"   (e) {info['key']}: chosen among {len(pool)} images with all 5 outlines valid and none flagged; expert SD {info['spread']:.3f}, "
    f"vCDR {info['vcdr']:.2f} [{info['interval'][0]:.2f}, {info['interval'][1]:.2f}], {info['inside']}/{info['n']} inside"
)
say(
    "   expert vCDRs: "
    + ", ".join(f"{x:.2f}" for x in d["rx"])
    + "; area ratios: "
    + ", ".join(f"{ratio(u):.2f}" for u in d["U"])
)
say(f"\nSaved {FD}\nCUPDISC CHECK DONE")
