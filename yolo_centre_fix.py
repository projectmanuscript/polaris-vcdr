"""
Test-time centre fix.
Finding: preprocessed_data/ (every crop used so far) was made with YOLO 'disc_detector_test-2', trained on a random 85% of ALL
RIGA+DRIONS images (initial.ipynb cell 38), i.e. on most of today's test images. 'disc_detector_leakfree' (Secondary.ipynb
cell 25) was trained on the old project train set (= today's train + val + calib) but crops were never regenerated with it.
This script:
  1. trains YOLO on project_split_v2 TRAIN only (same settings as before: yolov8n, 15 epochs, imgsz 384, batch 8, patience 10)
  2. reports centre error vs ground truth (average of the raters' disc boxes = how the YOLO labels were made) on val/calib/test
     for: old leaky detector (from the saved transforms), existing leak-free detector, new detector. In original and crop pixels.
  3. writes preprocessed_data_v3/: val/calib/test re-cropped with the NEW detector (same crop_and_normalize as the notebook);
     train entries are symlinks to the old crops (models are NOT retrained). Old preprocessed_data/ is untouched.
Outputs: outputs/centre_v3/ (centre_errors.json, log), outputs/yolo_runs/disc_detector_v2split/, preprocessed_data_v3/
"""

import os, json, random, shutil, numpy as np, cv2
from PIL import Image
from ultralytics import YOLO

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
OUT = os.path.join(BASE, "outputs", "centre_v3")
os.makedirs(OUT, exist_ok=True)
OLD_PRE, NEW_PRE = os.path.join(BASE, "preprocessed_data"), os.path.join(BASE, "preprocessed_data_v3")
PROC = (
    os.path.join(BASE, "processed_data_v2")
    if os.path.isdir(os.path.join(BASE, "processed_data_v2"))
    else os.path.join(BASE, "processed_data")
)
LOG = open(os.path.join(OUT, "centre_fix.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


split = json.load(open(os.path.join(BASE, "project_split_v2.json")))
say(f"Annotations read from {PROC}; split sizes: " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))

# ---------------- 1. YOLO on v2 train only
RUN = os.path.join(BASE, "outputs", "yolo_runs", "disc_detector_v2split")
BEST = os.path.join(RUN, "weights", "best.pt")
if not os.path.exists(BEST):
    Y = os.path.join(BASE, "yolo_data_v3")
    shutil.rmtree(Y, ignore_errors=True)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(Y, d))
    keys = list(split["train"])
    random.Random(2026).shuffle(keys)
    cut = int(0.85 * len(keys))
    miss = 0
    for part, ks in (("train", keys[:cut]), ("val", keys[cut:])):
        for k in ks:
            c = k.split("/", 1)[1]
            si, sl = f"yolo_data/all_images/{c}.jpg", f"yolo_data/all_labels/{c}.txt"
            if os.path.exists(si) and os.path.exists(sl):
                shutil.copy(si, f"{Y}/images/{part}/{c}.jpg")
                shutil.copy(sl, f"{Y}/labels/{part}/{c}.txt")
            else:
                miss += 1
    say(
        f"YOLO set from v2 TRAIN only: {cut} train / {len(keys)-cut} internal val (early stopping), {miss} missing label files"
    )
    open(f"{Y}/dataset.yaml", "w").write(f"path: {Y}\ntrain: images/train\nval: images/val\nnames:\n  0: optic_disc\n")
    YOLO("yolov8n.pt").train(
        data=f"{Y}/dataset.yaml",
        epochs=15,
        imgsz=384,
        batch=8,
        patience=10,
        seed=2026,
        project=os.path.join(BASE, "outputs", "yolo_runs"),
        name="disc_detector_v2split",
        exist_ok=True,
        verbose=False,
    )
say("New detector:", BEST)
det = {"new_v2split": YOLO(BEST), "leakfree_v1": YOLO("runs/detect/yolo_runs/disc_detector_leakfree/weights/best.pt")}


# ---------------- helpers (crop_and_normalize copied from initial.ipynb cell 56)
def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def crop_and_normalize(img, center_x, center_y, disc_diameter, target_size=512):
    roi_side = disc_diameter * 2.6
    half = roi_side / 2
    x_min, y_min, x_max, y_max = int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)
    h, w = img.shape[:2]
    pl, pt, pr, pb = max(0, -x_min), max(0, -y_min), max(0, x_max - w), max(0, y_max - h)
    if any([pl, pt, pr, pb]):
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        x_min += pl
        x_max += pl
        y_min += pt
        y_max += pt
    crop = cv2.resize(img[y_min:y_max, x_min:x_max], (target_size, target_size))
    enh = crop.copy()
    enh[:, :, 1] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop[:, :, 1])
    f = enh.astype(np.float32)
    return (f - f.mean(axis=(0, 1), keepdims=True)) / (f.std(axis=(0, 1), keepdims=True) + 1e-6), crop


def resolve(p):
    q = p.replace("\\", "/")
    for c in (q, os.path.join(BASE, q)):
        if os.path.exists(c):
            return c


def gt_centre(rec):
    b = [
        [np.min(p[:, 0]), np.min(p[:, 1]), np.max(p[:, 0]), np.max(p[:, 1])]
        for p in (np.asarray(d["disc_contour"], float) for d in rec["raters"].values() if d.get("disc_contour"))
    ]
    b = np.mean(b, 0)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1])


def detect(model, path):
    r = model(path, conf=0.01, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    x, y, w, h = r.boxes[0].xywh[0].tolist()
    return x, y, max(w, h), float(r.boxes[0].conf[0])


# ---------------- 2 + 3. centre errors and re-crop
rows, fails = [], []
for part in ["train", "val", "calib", "test"]:
    for k in split[part]:
        ds, case = k.split("/", 1)
        od, nd = os.path.join(OLD_PRE, ds), os.path.join(NEW_PRE, ds)
        os.makedirs(nd, exist_ok=True)
        if part == "train":  # unchanged training crops: link, do not copy
            for suf in ["_normalized.npy", "_transform.json", "_viewable.jpg"]:
                s, t = os.path.join(od, case + suf), os.path.join(nd, case + suf)
                if os.path.exists(s) and not os.path.lexists(t):
                    os.symlink(s, t)
            continue
        try:
            rec = json.load(open(os.path.join(PROC, ds, case + ".json")))
            ip = resolve(rec["image_path"])
            if ip is None:
                fails.append((k, "image not found: " + rec["image_path"]))
                continue
            gx, gy, gd = gt_centre(rec)
            row = {"key": k, "part": part, "gt": [gx, gy], "gt_diam": gd}
            tr_old = json.load(open(os.path.join(od, case + "_transform.json")))
            row["old_leaky"] = {
                "px": float(np.hypot(tr_old["center_x"] - gx, tr_old["center_y"] - gy)),
                "scale": tr_old["scale"],
            }
            for name, m in det.items():
                d = detect(m, ip)
                if d is None:
                    row[name] = None
                    continue
                x, y, dia, conf = d
                sc = 512 / (dia * 2.6)
                row[name] = {"px": float(np.hypot(x - gx, y - gy)), "scale": sc, "conf": conf, "diam_ratio": dia / gd}
                if name == "new_v2split":
                    norm, view = crop_and_normalize(load_rgb(ip), x, y, dia)
                    np.save(os.path.join(nd, case + "_normalized.npy"), norm)
                    Image.fromarray(view).save(os.path.join(nd, case + "_viewable.jpg"), quality=90)
                    json.dump(
                        {
                            "x_min": x - dia * 1.3,
                            "y_min": y - dia * 1.3,
                            "scale": sc,
                            "center_x": x,
                            "center_y": y,
                            "disc_diameter": dia,
                            "detector": "disc_detector_v2split",
                            "conf": conf,
                        },
                        open(os.path.join(nd, case + "_transform.json"), "w"),
                    )
            rows.append(row)
        except Exception as e:
            fails.append((k, str(e)))
say(
    f"\nRe-cropped {sum(1 for r in rows if r.get('new_v2split'))} val/calib/test images; failures {len(fails)}"
    + (f", e.g. {fails[:3]}" if fails else "")
)
say("No-detection with the new detector: " + str([r["key"] for r in rows if r.get("new_v2split") is None]))
say(
    "Low confidence (< 0.25) with the new detector: "
    + str(
        [
            (r["key"], round(r["new_v2split"]["conf"], 2))
            for r in rows
            if r.get("new_v2split") and r["new_v2split"]["conf"] < 0.25
        ]
    )
)
say(
    "\nCENTRE ERROR vs ground truth (average rater disc box). Crop px = error after the 512 crop (the scale the model sees;"
    " the earlier shift test: harmonic disc Dice 94.06 -> 85.09 at 20 crop px)."
)
for part in ["val", "calib", "test"]:
    for name in ["old_leaky", "leakfree_v1", "new_v2split"]:
        v = [r[name] for r in rows if r["part"] == part and r.get(name)]
        if not v:
            continue
        cp = np.array([x["px"] * x["scale"] for x in v])
        op = np.array([x["px"] for x in v])
        say(
            f"  {part:5s} {name:12s} n={len(v):3d}  crop px: median {np.median(cp):5.1f}  90th {np.percentile(cp, 90):5.1f}  max {cp.max():6.1f}  "
            f"| >10 px: {np.mean(cp > 10)*100:4.1f}%  >20 px: {np.mean(cp > 20)*100:4.1f}%  | original px median {np.median(op):.1f}"
        )
json.dump({"rows": rows, "fails": fails}, open(os.path.join(OUT, "centre_errors.json"), "w"), indent=1, default=float)
say("\nSaved", OUT, "and", NEW_PRE)
