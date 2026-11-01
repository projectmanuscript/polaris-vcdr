"""
Chaksu probe (read-only). Answers, before any loader is written:
  1. per split/camera/expert: how many images, how many disc/cup masks, do mask names match image names
  2. mask format: size vs image size, dtype, pixel values, is the cup inside the disc
  3. Expert 3 on Bosch (no raw drawings in 2.0): are those masks copies of another expert?
  4. where the disc sits horizontally (for left/right eye mirroring)
  5. one overlay image with all 5 experts' outlines, for a hand check
Writes only to outputs/chaksu_probe/.
"""

import os, glob, json, random, numpy as np, cv2

ROOT = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "raw/Chaksu/20123135")
OUT = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "outputs/chaksu_probe")
os.makedirs(OUT, exist_ok=True)
CAMS, EXPERTS = ["Bosch", "Forus", "Remidio"], [1, 2, 3, 4, 5]
stem = lambda p: os.path.splitext(os.path.basename(p))[0].lower()


def files(d):
    return sorted(
        p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(p) and not os.path.basename(p).startswith(".")
    )


def subdir(parent, name):
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower():
            return os.path.join(parent, d)
    return None


def read_mask(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return m


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * np.logical_and(a, b).sum() / s


random.seed(0)
summary = {}

for split in ["Train", "Test"]:
    base = os.path.join(ROOT, split, split)
    for cam in CAMS:
        imgs = {stem(p): p for p in files(os.path.join(base, "1.0_Original_Fundus_Images", cam))}
        exts = sorted({os.path.splitext(p)[1] for p in imgs.values()})
        print(f"\n===== {split} / {cam}: {len(imgs)} images {exts}  e.g. {list(imgs)[:3]}")
        masks = {}
        for e in EXPERTS:
            ed = os.path.join(base, "3.0_Doctors_Annotations_Binary_OD_OC", f"Expert {e}", cam)
            for s in ["Disc", "Cup"]:
                d = subdir(ed, s)
                fs = {stem(p): p for p in files(d)} if d else {}
                masks[(e, s)] = fs
                matched = len(set(fs) & set(imgs))
                extra = sorted(set(fs) - set(imgs))[:3]
                print(
                    f"  Expert {e} {s:4s}: {len(fs):4d} files, {matched:4d} match an image name"
                    + (f", unmatched e.g. {extra}" if extra else "")
                    + (f"  ext {sorted({os.path.splitext(p)[1] for p in fs.values()})}" if fs else "")
                )
        # mask format on 3 random images that have all experts' disc+cup
        full = [k for k in imgs if all(k in masks[(e, s)] for e in EXPERTS for s in ["Disc", "Cup"])]
        print(f"  images with all 5 experts' disc+cup masks: {len(full)}")
        xs, cupin, shapes = [], [], set()
        for k in random.sample(full, min(3, len(full))):
            im = cv2.imread(imgs[k])
            shapes.add(im.shape[:2])
            for e in EXPERTS:
                d, c = read_mask(masks[(e, "Disc")][k]), read_mask(masks[(e, "Cup")][k])
                if e == 1:
                    print(
                        f"  {k}: image {im.shape[:2]}  disc mask {d.shape} {d.dtype} values {np.unique(d)[:6]}  cup values {np.unique(c)[:6]}"
                    )
                db, cb = d > d.max() / 2, c > c.max() / 2
                if cb.sum():
                    cupin.append(np.logical_and(db, cb).sum() / cb.sum())
                if db.sum():
                    xs.append(np.argwhere(db)[:, 1].mean() / db.shape[1])
        print(
            f"  cup pixels inside disc: min {min(cupin)*100:.1f}%  mean {np.mean(cupin)*100:.1f}%"
            if cupin
            else "  no cup pixels found"
        )
        # disc horizontal position, Expert 1, up to 60 images
        pos = []
        for k in random.sample(full, min(60, len(full))):
            d = read_mask(masks[(1, "Disc")][k])
            db = d > d.max() / 2 if d.max() else None
            if db is not None and db.sum():
                pos.append(np.argwhere(db)[:, 1].mean() / db.shape[1])
        pos = np.array(pos)
        print(
            f"  disc centre x / width over {len(pos)} images: left half {np.mean(pos < 0.5)*100:.0f}%, right half {np.mean(pos >= 0.5)*100:.0f}%  "
            f"(quartiles {np.percentile(pos, 25):.2f} {np.percentile(pos, 50):.2f} {np.percentile(pos, 75):.2f})"
        )
        # pairwise expert agreement on 20 images (Expert 3 copy check, all cameras for comparison)
        sample = random.sample(full, min(20, len(full)))
        M = np.zeros((5, 5))
        I = np.zeros((5, 5))
        for k in sample:
            ms = [read_mask(masks[(e, "Cup")][k]) for e in EXPERTS]
            ms = [m > m.max() / 2 if m.max() else m > 0 for m in ms]
            for i in range(5):
                for j in range(5):
                    dd = dice(ms[i], ms[j])
                    M[i, j] += dd / len(sample)
                    I[i, j] += dd == 1.0
        print("  cup Dice between experts (rows/cols = Expert 1..5), and count of IDENTICAL masks out of", len(sample))
        for i in range(5):
            print(
                "   E%d " % (i + 1)
                + " ".join(f"{M[i, j]*100:5.1f}" for j in range(5))
                + "   identical: "
                + " ".join(f"{int(I[i, j]):2d}" for j in range(5))
            )
        summary[f"{split}/{cam}"] = dict(
            n_images=len(imgs), n_full=len(full), disc_x_right_frac=float(np.mean(pos >= 0.5)) if len(pos) else None
        )
        # one overlay for the hand check (first Train/Remidio image only)
        if split == "Train" and cam == "Remidio" and full:
            k = sorted(full)[0]
            im = cv2.imread(imgs[k])
            ov = im.copy()
            for e, col in zip(EXPERTS, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]):
                for s in ["Disc", "Cup"]:
                    m = read_mask(masks[(e, s)][k])
                    b = (m > m.max() / 2).astype(np.uint8) if m.max() else (m > 0).astype(np.uint8)
                    if b.shape != im.shape[:2]:
                        b = cv2.resize(b, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cs, -1, col, 3)
            d = read_mask(masks[(1, "Disc")][k])
            yx = np.argwhere(d > d.max() / 2)
            cy, cx = yx.mean(0).astype(int)
            if d.shape != im.shape[:2]:
                cy, cx = int(cy * im.shape[0] / d.shape[0]), int(cx * im.shape[1] / d.shape[1])
            r = int(max(np.ptp(yx[:, 0]), np.ptp(yx[:, 1])) * im.shape[0] / d.shape[0]) + 80
            crop = ov[max(cy - r, 0) : cy + r, max(cx - r, 0) : cx + r]
            cv2.imwrite(
                os.path.join(OUT, f"overlay_{k}_full.jpg"), cv2.resize(ov, (ov.shape[1] // 3, ov.shape[0] // 3))
            )
            cv2.imwrite(os.path.join(OUT, f"overlay_{k}_crop.png"), crop)
            print(f"  overlay saved for {k}: colours E1 red, E2 green, E3 blue, E4 yellow, E5 magenta")
json.dump(summary, open(os.path.join(OUT, "probe_summary.json"), "w"), indent=1)
print("\nSaved to", OUT)
