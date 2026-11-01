"""
Full duplicate-rater check on Chaksu (read-only). Every image, every expert pair, disc and cup separately.
  identical      = binary masks equal pixel for pixel
  near-identical = Dice >= 0.99 (catches re-saved copies)
Also, for Forus: per-image listing of which experts match, and a byte-level comparison of the ORIGINAL drawings
(2.0_Doctors_Annotations) for Expert 1 vs Expert 5, to see where the copy happened.
Output: outputs/chaksu_probe/rater_check.json (+ printed summary)
"""

import os, glob, json, hashlib, itertools, numpy as np, cv2

ROOT = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "raw/Chaksu/20123135")
OUT = os.path.join(os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "outputs/chaksu_probe")
os.makedirs(OUT, exist_ok=True)
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
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() == name.lower():
            return os.path.join(parent, d)


def binmask(p):
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m.max(axis=2)
    return m > (m.max() / 2 if m.max() else 0)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


PAIRS = list(itertools.combinations(range(1, 6), 2))
report = {}
for split in ["Train", "Test"]:
    for cam in ["Bosch", "Forus", "Remidio"]:
        base = os.path.join(ROOT, split, split)
        imgs = files(os.path.join(base, "1.0_Original_Fundus_Images", cam))
        M = {
            (e, s): files(subdir(os.path.join(base, "3.0_Doctors_Annotations_Binary_OD_OC", f"Expert {e}", cam), s))
            for e in range(1, 6)
            for s in ["Disc", "Cup"]
        }
        res = {s: {f"E{a}-E{b}": {"identical": 0, "near": 0, "n": 0} for a, b in PAIRS} for s in ["Disc", "Cup"]}
        per_image = {}
        for k in sorted(imgs):
            per_image[k] = {}
            for s in ["Disc", "Cup"]:
                ms = {e: binmask(M[(e, s)][k]) for e in range(1, 6) if k in M[(e, s)]}
                same = []
                for a, b in PAIRS:
                    if a not in ms or b not in ms:
                        continue
                    r = res[s][f"E{a}-E{b}"]
                    r["n"] += 1
                    tot = ms[a].sum() + ms[b].sum()
                    d = 1.0 if tot == 0 else 2 * np.logical_and(ms[a], ms[b]).sum() / tot
                    if np.array_equal(ms[a], ms[b]):
                        r["identical"] += 1
                        same.append(f"E{a}=E{b}")
                    elif d >= 0.99:
                        r["near"] += 1
                        same.append(f"E{a}~E{b}")
                per_image[k][s] = same
        print(
            f"\n===== {split} / {cam}: {len(imgs)} images  (pairs with any identical or near-identical masks; all others are 0)"
        )
        clean = True
        for s in ["Disc", "Cup"]:
            for pr, r in res[s].items():
                if r["identical"] or r["near"]:
                    clean = False
                    print(f"  {s:4s} {pr}: identical {r['identical']}/{r['n']}, near-identical {r['near']}/{r['n']}")
        if clean:
            print("  no duplicated experts at all")
        if cam == "Forus":
            pat = {}
            for k, v in per_image.items():
                pat.setdefault((tuple(v["Disc"]), tuple(v["Cup"])), []).append(k)
            print("  per-image patterns (disc matches | cup matches): number of images, examples")
            for (dd, cc), ks in sorted(pat.items(), key=lambda x: -len(x[1])):
                print(f"    disc {list(dd) or 'none'} | cup {list(cc) or 'none'}: {len(ks)} images, e.g. {ks[:5]}")
            raw = {}
            for s in ["Disc", "Cup"]:
                r1 = files(subdir(os.path.join(base, "2.0_Doctors_Annotations", "Expert 1", cam), s))
                r5 = files(subdir(os.path.join(base, "2.0_Doctors_Annotations", "Expert 5", cam), s))
                common = sorted(set(r1) & set(r5))
                same = [k for k in common if md5(r1[k]) == md5(r5[k])]
                raw[s] = {"n": len(common), "byte_identical": len(same)}
                print(
                    f"  ORIGINAL drawings (2.0) {s}: Expert 1 vs Expert 5 byte-identical on {len(same)}/{len(common)} images"
                )
            report[f"{split}/{cam}/raw_2.0_E1_vs_E5"] = raw
        report[f"{split}/{cam}"] = {"pairs": res, "per_image": per_image}
json.dump(report, open(os.path.join(OUT, "rater_check.json"), "w"), indent=1)
print("\nSaved", os.path.join(OUT, "rater_check.json"))
