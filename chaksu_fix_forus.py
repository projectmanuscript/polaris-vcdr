"""
Forus E1/E5: decide, image by image, WHOSE drawing the dataset's 3.0 "Expert 1" mask (= its "Expert 5" mask) comes from.
Hypothesis from resolve_flags: the 3.0 mask is Expert 5's, and Expert 1's own mask is the one missing.
For each Forus image:
  a = Dice(rebuilt E1 drawing, 3.0 E1 mask), b = Dice(rebuilt E5 drawing, 3.0 E1 mask), disc and cup
  b >= 0.97 on both and a < 0.97 -> 3.0 mask belongs to E5; E1 uses its rebuilt drawing
  a >= 0.97 on both and b < 0.97 -> 3.0 mask belongs to E1; E5 uses its rebuilt drawing
  otherwise                       -> unresolved: keep the 3.0 mask as ONE rater, drop the other
Rebuilt masks must have the cup >= 98% inside the disc; E1 and E5 must not duplicate each other (Dice >= 0.99 on both).
Also re-checks the rebuild method PER EXPERT (E1..E4 vs their 3.0 masks) so the 77% cup 5th percentile is explained.
Updates outputs/chaksu/rater_manifest.json (backup kept as rater_manifest_before_forus_fix.json).
"""

import os, json, shutil, numpy as np, cv2

exec(open("chaksu_resolve_flags.py").read().split("LOG = open")[0])  # helpers only (paths, readers, rebuild methods)
LOG = open(os.path.join(OUT, "fix_forus.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


MF = os.path.join(OUT, "rater_manifest.json")
man = json.load(open(MF))
if not os.path.exists(os.path.join(OUT, "rater_manifest_before_forus_fix.json")):
    shutil.copy(MF, os.path.join(OUT, "rater_manifest_before_forus_fix.json"))
METHOD = "overlay_diff"


def cup_inside(d, c):
    return np.logical_and(c, d).sum() / max(c.sum(), 1)


per_exp = {e: {"Disc": [], "Cup": []} for e in [1, 2, 3, 4, 5]}
counts = {}
for split in ["Train", "Test"]:
    imgs = images(split, "Forus")
    for k in sorted(imgs):
        img = cv2.imread(imgs[k])
        m3 = {(e, s): bin3(masks3(split, "Forus", e, s)[k]) for e in [1, 2, 3, 4] for s in ["Disc", "Cup"]}
        rb = {}
        for e in [1, 2, 3, 4, 5]:
            for s in ["Disc", "Cup"]:
                p = draws2(split, "Forus", e, s).get(k)
                rb[(e, s)] = recon(p, img, m3[(1, s)].shape, METHOD) if p else None
                if rb[(e, s)] is not None:
                    per_exp[e][s].append(dice(rb[(e, s)], m3[(min(e, 1) if e == 5 else e, s)]))
        a = [dice(rb[(1, s)], m3[(1, s)]) if rb[(1, s)] is not None else 0 for s in ["Disc", "Cup"]]
        b = [dice(rb[(5, s)], m3[(1, s)]) if rb[(5, s)] is not None else 0 for s in ["Disc", "Cup"]]
        A, B = min(a) >= GATE, min(b) >= GATE
        if B and not A:
            owner, other = 5, 1
        elif A and not B:
            owner, other = 1, 5
        else:
            owner, other = None, None
        entry = man[split]["Forus"][k]
        R = entry["raters"]
        D = entry["dropped"]
        for e in ("1", "5"):
            R.pop(e, None)
            D.pop(e, None)
        three = {
            "source": "3.0 (labelled Expert 1 and Expert 5 by the dataset)",
            "disc": masks3(split, "Forus", 1, "Disc")[k],
            "cup": masks3(split, "Forus", 1, "Cup")[k],
        }
        if owner is None:
            R["1"] = three
            D["5"] = (
                f"unresolved: rebuilt E1 vs 3.0 {np.round(a, 3).tolist()}, rebuilt E5 vs 3.0 {np.round(b, 3).tolist()}"
            )
            res = "unresolved"
        else:
            R[str(owner)] = three
            d, c = rb[(other, "Disc")], rb[(other, "Cup")]
            if d is None or c is None:
                D[str(other)] = "no usable drawing"
                res = f"3.0 is E{owner}; E{other} no drawing"
            elif cup_inside(d, c) < 0.98:
                D[str(other)] = f"rebuilt cup only {cup_inside(d, c)*100:.0f}% inside rebuilt disc"
                res = f"3.0 is E{owner}; E{other} rebuilt failed nesting"
            elif dice(d, m3[(1, 'Disc')]) >= DUP and dice(c, m3[(1, 'Cup')]) >= DUP:
                D[str(other)] = f"duplicate of Expert {owner}"
                res = "duplicate"
            else:
                dd = os.path.join(OUT, f"rebuilt_E{other}_forus", split)
                os.makedirs(os.path.join(dd, "Disc"), exist_ok=True)
                os.makedirs(os.path.join(dd, "Cup"), exist_ok=True)
                cv2.imwrite(os.path.join(dd, "Disc", k + ".png"), d.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(dd, "Cup", k + ".png"), c.astype(np.uint8) * 255)
                R[str(other)] = {
                    "source": "rebuilt_from_2.0",
                    "disc": os.path.join(dd, "Disc", k + ".png"),
                    "cup": os.path.join(dd, "Cup", k + ".png"),
                }
                res = f"3.0 is E{owner}; E{other} rebuilt"
        entry["raters"] = dict(sorted(R.items()))
        counts.setdefault(split, {}).setdefault(res, 0)
        counts[split][res] += 1
say(
    "Rebuild method checked per expert (rebuilt drawing vs the dataset's 3.0 mask with the SAME label; for E5 the 3.0 'Expert 1' mask):"
)
for e in [1, 2, 3, 4, 5]:
    say(
        f"  Expert {e}: disc median {np.median(per_exp[e]['Disc'])*100:.1f} (5th pct {np.percentile(per_exp[e]['Disc'], 5)*100:.1f}), "
        f"cup median {np.median(per_exp[e]['Cup'])*100:.1f} (5th pct {np.percentile(per_exp[e]['Cup'], 5)*100:.1f})"
    )
say("\nWho owns the dataset's Forus 'Expert 1' mask, per image:")
for split, c in counts.items():
    say(f"  {split}: {c}")
for split in ["Train", "Test"]:
    v = [len(x["raters"]) for x in man[split]["Forus"].values()]
    say(
        f"  {split}/Forus raters per image after fix: { {n: v.count(n) for n in sorted(set(v))} }  (total pairs {sum(v)})"
    )
json.dump(man, open(MF, "w"), indent=1)
say("\nUpdated", MF)
