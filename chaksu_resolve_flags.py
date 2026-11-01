"""
Resolve every Chaksu data flag in one pass and write the rater manifest the loader will use.
  A. Formats of the ORIGINAL drawings (2.0_Doctors_Annotations); sample pictures saved
  B. Drawing -> mask reconstruction, VALIDATED on Forus Experts 1-4 against their own 3.0 masks (gate: median Dice >= 0.97)
  C. Forus Expert 5 rebuilt from its drawings; compared with Expert 1 at drawing level -> keep (independent) or drop (copy)
  D. Remidio E4/E5 near-copy batch: are the drawings themselves near-identical? (20 images, method re-validated on Remidio)
  E. One duplicate rule for the whole dataset: if two experts agree at Dice >= 0.99 on BOTH disc and cup for an image,
     the higher-numbered one is dropped for that image
  F. Identify the two extra files in the fused folders (not used, recorded only)
Reads outputs/chaksu_probe/rater_check.json. Writes outputs/chaksu/ (manifest, rebuilt masks, samples, log).
"""

import os, glob, json, random, numpy as np, cv2

ROOT = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "raw/Chaksu/20123135")
BASE = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "outputs")
OUT = os.path.join(BASE, "chaksu")
os.makedirs(OUT, exist_ok=True)
RC = json.load(open(os.path.join(BASE, "chaksu_probe", "rater_check.json")))
GATE, DUP = 0.97, 0.99
stem = lambda p: os.path.splitext(os.path.basename(p))[0].lower()


def files(d):
    return (
        {
            stem(p): p
            for p in glob.glob(os.path.join(d, "*"))
            if os.path.isfile(p) and not os.path.basename(p).startswith(".")
        }
        if d
        else {}
    )


def subdir(parent, name):
    if not parent or not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower():
            return os.path.join(parent, d)


def sdir(split):
    return os.path.join(ROOT, split, split)


def masks3(split, cam, e, s):
    return files(subdir(os.path.join(sdir(split), "3.0_Doctors_Annotations_Binary_OD_OC", f"Expert {e}", cam), s))


def draws2(split, cam, e, s):
    return files(subdir(os.path.join(sdir(split), "2.0_Doctors_Annotations", f"Expert {e}", cam), s))


def images(split, cam):
    return files(os.path.join(sdir(split), "1.0_Original_Fundus_Images", cam))


def read(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        from PIL import Image

        m = np.array(Image.open(p))
    return m


def bin3(p):
    m = read(p)
    m = m.max(axis=2) if m.ndim == 3 else m
    return m > (m.max() / 2 if m.max() else 0)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def u8(g):
    return (
        g
        if g.dtype == np.uint8
        else cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )


def largest_fill(b):
    cs, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(b.shape, np.uint8)
    if cs:
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 1, thickness=-1)
    return out.astype(bool)


def m_thresh(d, img):
    g = u8(d if d.ndim == 2 else d[..., :3].max(2))
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:
        b = 255 - b
    return largest_fill(b > 0)


def m_alpha(d, img):
    if d.ndim != 3 or d.shape[2] != 4 or d[..., 3].min() == d[..., 3].max():
        return None
    _, b = cv2.threshold(u8(d[..., 3]), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return largest_fill(b > 0)


def m_diff(d, img):
    if d.ndim != 3 or img is None:
        return None
    d3 = u8(d[..., :3])
    im = img if img.shape[:2] == d3.shape[:2] else cv2.resize(img, (d3.shape[1], d3.shape[0]))
    b = (cv2.absdiff(d3, im).max(2) > 40).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return largest_fill(b)


METHODS = {"threshold": m_thresh, "alpha": m_alpha, "overlay_diff": m_diff}


def recon(path, img, shape, method):
    try:
        r = METHODS[method](read(path), img)
    except Exception:
        return None
    if r is None:
        return None
    return (
        r
        if r.shape == shape
        else cv2.resize(r.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    )


LOG = open(os.path.join(OUT, "resolve_flags.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# ---------------- A. formats
say("A. ORIGINAL DRAWING FORMATS (2.0)")
for split in ["Train", "Test"]:
    for cam in ["Forus", "Remidio", "Bosch"]:
        for e in [1, 5]:
            dd = draws2(split, cam, e, "Disc")
            im = images(split, cam)
            if not dd:
                say(f"  {split}/{cam} Expert {e}: no 2.0 drawings")
                continue
            k = sorted(set(dd) & set(im))[0] if set(dd) & set(im) else sorted(dd)[0]
            d = read(dd[k])
            say(
                f"  {split}/{cam} Expert {e} Disc: {len(dd)} files, {len(set(dd) & set(im))} match image names, ext {sorted({os.path.splitext(p)[1] for p in dd.values()})}, "
                f"e.g. {k}: shape {d.shape} {d.dtype}, {len(np.unique(d))} distinct values"
            )
            if split == "Train" and cam in ("Forus", "Remidio"):
                v = u8(d[..., :3] if d.ndim == 3 else d)
                cv2.imwrite(
                    os.path.join(OUT, f"sample_drawing_{cam}_E{e}_{k}.jpg"),
                    cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2)),
                )

# ---------------- B. validate reconstruction on Forus Experts 1-4
say("\nB. RECONSTRUCTION VALIDATED ON FORUS EXPERTS 1-4 (rebuilt from 2.0 drawing vs the dataset's own 3.0 mask)")
img_cache = {}


def get_img(split, cam, k):
    key = (split, cam, k)
    if key not in img_cache:
        p = images(split, cam).get(k)
        img_cache[key] = cv2.imread(p) if p else None
    return img_cache[key]


scores = {s: {m: [] for m in METHODS} for s in ["Disc", "Cup"]}
for split in ["Train", "Test"]:
    for e in [1, 2, 3, 4]:
        for s in ["Disc", "Cup"]:
            M3, D2 = masks3(split, "Forus", e, s), draws2(split, "Forus", e, s)
            for k in sorted(set(M3) & set(D2)):
                ref = bin3(M3[k])
                img = get_img(split, "Forus", k)
                for m in METHODS:
                    r = recon(D2[k], img, ref.shape, m)
                    if r is not None:
                        scores[s][m].append(dice(r, ref))
best = {}
for s in ["Disc", "Cup"]:
    for m, v in scores[s].items():
        if v:
            say(
                f"  {s:4s} {m:13s} n={len(v):4d}  median Dice {np.median(v)*100:.2f}  5th pct {np.percentile(v, 5)*100:.2f}  min {min(v)*100:.2f}"
            )
        else:
            say(f"  {s:4s} {m:13s} not applicable to this format")
    ok = {m: np.median(v) for m, v in scores[s].items() if v}
    best[s] = max(ok, key=ok.get) if ok else None
    say(
        f"  -> {s}: best method {best[s]} ({ok.get(best[s], 0)*100:.2f}); gate {GATE*100:.0f}%: {'PASS' if best[s] and ok[best[s]] >= GATE else 'FAIL'}"
    )
gate_ok = all(best[s] and np.median(scores[s][best[s]]) >= GATE for s in ["Disc", "Cup"])

# ---------------- C. Forus Expert 5
say("\nC. FORUS EXPERT 5 REBUILT FROM ITS OWN DRAWINGS")
forus_e5 = {}
if not gate_ok:
    say(
        "  Reconstruction failed the gate, so Expert 5 cannot be recovered reliably. Decision: Expert 5 DROPPED on all Forus images."
    )
    say("  (See sample_drawing_*.jpg in outputs/chaksu to inspect the drawing format.)")
else:
    for split in ["Train", "Test"]:
        rows = []
        for k in sorted(images(split, "Forus")):
            img = get_img(split, "Forus", k)
            rec = {}
            for s in ["Disc", "Cup"]:
                d5, d1 = draws2(split, "Forus", 5, s).get(k), draws2(split, "Forus", 1, s).get(k)
                ref1 = bin3(masks3(split, "Forus", 1, s)[k])
                r5 = recon(d5, img, ref1.shape, best[s]) if d5 else None
                r1 = recon(d1, img, ref1.shape, best[s]) if d1 else None
                rec[s] = r5
                rec[s + "_vs_E1"] = dice(r5, r1) if (r5 is not None and r1 is not None) else None
                rec[s + "_vs_others"] = (
                    [dice(r5, bin3(masks3(split, "Forus", e, s)[k])) for e in [2, 3, 4]] if r5 is not None else None
                )
            if rec["Disc"] is None or rec["Cup"] is None:
                forus_e5[(split, k)] = ("dropped", "no usable drawing")
                continue
            copy = rec["Disc_vs_E1"] >= DUP and rec["Cup_vs_E1"] >= DUP
            cup_in = np.logical_and(rec["Cup"], rec["Disc"]).sum() / max(rec["Cup"].sum(), 1)
            if copy:
                forus_e5[(split, k)] = ("dropped", "Expert 5 drawing duplicates Expert 1")
            elif cup_in < 0.98:
                forus_e5[(split, k)] = ("dropped", f"rebuilt cup only {cup_in*100:.0f}% inside rebuilt disc")
            else:
                for s in ["Disc", "Cup"]:
                    d = os.path.join(OUT, "rebuilt_E5_forus", split, s)
                    os.makedirs(d, exist_ok=True)
                    p = os.path.join(d, k + ".png")
                    cv2.imwrite(p, rec[s].astype(np.uint8) * 255)
                forus_e5[(split, k)] = (
                    "rebuilt",
                    {s: os.path.join(OUT, "rebuilt_E5_forus", split, s, k + ".png") for s in ["Disc", "Cup"]},
                )
            rows.append((rec["Disc_vs_E1"], rec["Cup_vs_E1"], np.mean(rec["Cup_vs_others"])))
        r = np.array(rows)
        n_keep = sum(1 for (sp, _), v in forus_e5.items() if sp == split and v[0] == "rebuilt")
        say(
            f"  {split}: {len(images(split, 'Forus'))} images. Rebuilt E5 vs rebuilt E1 (drawing level): disc median {np.median(r[:, 0])*100:.1f}, "
            f"cup median {np.median(r[:, 1])*100:.1f}.  Rebuilt E5 cup vs Experts 2-4: median {np.median(r[:, 2])*100:.1f} "
            f"(independent experts agree at about 78-86).  E5 kept on {n_keep}, dropped on {len(images(split, 'Forus')) - n_keep}"
        )

# ---------------- D. Remidio E4/E5 batch at drawing level
say("\nD. REMIDIO EXPERT 4 / EXPERT 5 NEAR-COPY BATCH, CHECKED ON THE ORIGINAL DRAWINGS")
pi = RC["Train/Remidio"]["per_image"]
batch = sorted(
    k
    for k, v in pi.items()
    if any(t in ("E4~E5", "E4=E5") for t in v["Disc"]) and any(t in ("E4~E5", "E4=E5") for t in v["Cup"])
)
say(f"  {len(batch)} Train/Remidio images where E4 and E5 agree at >= 99% on both disc and cup (from the full check)")
random.seed(0)
samp = random.sample(batch, min(20, len(batch)))
val, pair = [], []
for k in samp:
    img = get_img("Train", "Remidio", k)
    for s in ["Cup"]:
        refs = {e: bin3(masks3("Train", "Remidio", e, s)[k]) for e in [1, 4, 5]}
        r = {
            e: (
                recon(draws2("Train", "Remidio", e, s).get(k), img, refs[e].shape, best[s] or "threshold")
                if draws2("Train", "Remidio", e, s).get(k)
                else None
            )
            for e in [1, 4, 5]
        }
        if r[1] is not None:
            val.append(dice(r[1], refs[1]))
        if r[4] is not None and r[5] is not None:
            pair.append(dice(r[4], r[5]))
img_cache.clear()
if val:
    say(f"  method check on Remidio (Expert 1 rebuilt vs its 3.0 mask, cup): median {np.median(val)*100:.1f}")
if pair:
    say(f"  rebuilt E4 vs rebuilt E5 cup on the same images: median {np.median(pair)*100:.1f}, min {min(pair)*100:.1f}")
say(
    "  Reading: if the drawings also agree at ~99%, the near-copies are in the experts' own drawings (one edited the other's outline);"
    " either way rule E below counts them as one rater on those images."
)

# ---------------- E. duplicate rule + manifest
say(
    "\nE. DUPLICATE RULE (Dice >= 0.99 on BOTH disc and cup -> higher-numbered expert dropped for that image) AND MANIFEST"
)
manifest, stats = {}, {}
for split in ["Train", "Test"]:
    for cam in ["Bosch", "Forus", "Remidio"]:
        imgs = images(split, cam)
        pi = RC[f"{split}/{cam}"]["per_image"]
        M = {(e, s): masks3(split, cam, e, s) for e in range(1, 6) for s in ["Disc", "Cup"]}
        for k in sorted(imgs):
            ex, dropped = {}, {}
            for e in range(1, 6):
                if k in M[(e, "Disc")] and k in M[(e, "Cup")]:
                    ex[e] = {"source": "3.0", "disc": M[(e, "Disc")][k], "cup": M[(e, "Cup")][k]}
            if cam == "Forus" and 5 in ex:
                st, info = forus_e5.get((split, k), ("dropped", "reconstruction gate failed"))
                if st == "rebuilt":
                    ex[5] = {"source": "rebuilt_from_2.0", "disc": info["Disc"], "cup": info["Cup"]}
                else:
                    ex.pop(5)
                    dropped[5] = info
            pairs = lambda lst: {
                tuple(sorted(int(x) for x in t.replace("~", "=").replace("E", "").split("="))) for t in lst
            }
            both = pairs(pi[k]["Disc"]) & pairs(pi[k]["Cup"]) if k in pi else set()
            for a, b in sorted(both):
                if cam == "Forus" and 5 in (a, b):
                    continue  # E5 on Forus handled in C (3.0 copy replaced or dropped)
                if a in ex and b in ex:
                    ex.pop(b)
                    dropped[b] = f"duplicate of Expert {a} (Dice >= 0.99 on disc and cup)"
            if cam == "Forus" and 5 in ex:  # rebuilt E5 checked against the others too
                d5, c5 = bin3(ex[5]["disc"]), bin3(ex[5]["cup"])
                for a in [1, 2, 3, 4]:
                    if a in ex and dice(d5, bin3(ex[a]["disc"])) >= DUP and dice(c5, bin3(ex[a]["cup"])) >= DUP:
                        ex.pop(5)
                        dropped[5] = f"rebuilt Expert 5 duplicates Expert {a}"
                        break
            manifest.setdefault(split, {}).setdefault(cam, {})[k] = {
                "image": imgs[k],
                "raters": {str(e): v for e, v in ex.items()},
                "dropped": {str(e): r for e, r in dropped.items()},
            }
            stats.setdefault((split, cam), []).append(len(ex))
for (split, cam), v in stats.items():
    c = {n: v.count(n) for n in sorted(set(v))}
    say(f"  {split}/{cam}: {len(v)} images, raters per image {c}  (total rater/image pairs {sum(v)})")

# ---------------- F. extra files in fused folders
say("\nF. EXTRA FILES IN FUSED FOLDERS (not used by us; recorded)")
for split, rel, cam in [
    ("Train", "4.0_OD_CO_Fusion_Images/Expert 4/Remidio", "Remidio"),
    ("Test", "5.0_OD_OC_Mean_Median_Majority_STAPLE/Forus/Disc/Median", "Forus"),
]:
    f = files(os.path.join(sdir(split), rel))
    extra = sorted(set(f) - set(images(split, cam)))
    say(f"  {split}/{rel}: {len(f)} files, not matching any image: {[os.path.basename(f[x]) for x in extra]}")
say(
    "  Expert 3 / Bosch has no 2.0 drawings, but its 3.0 masks are independent of all other experts (full check: 0 duplicates). Kept."
)

json.dump(manifest, open(os.path.join(OUT, "rater_manifest.json"), "w"), indent=1)
say("\nSaved", os.path.join(OUT, "rater_manifest.json"), "and resolve_flags.log")
