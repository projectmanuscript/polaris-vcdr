"""
Fairness compute: the polar U-Net behind CFu, retrained on the SAME data and targets as the harmonic model.
Changed relative to train_baselines_v2.py (MODEL=polar), and nothing else:
  - training images: project_split_v2 TRAIN, every source (RIGA + DRIONS), i.e. the images the harmonic model was trained on
  - targets: masks rasterised from training_targets_v2 (the rater-mean M=12 outlines the harmonic model regresses),
    drawn on the -pi..pi grid that the outlines were sampled on
  - images without a cup outline (DRIONS): the cup channel's loss is masked out; the disc channel is trained as usual
  - validation targets: the same rasterised outlines; epoch selected on validation by mean(disc Dice, cup Dice), the rule
    used for every earlier model (disc Dice over all validation images, cup Dice over those with a cup outline)
Unchanged: PlainUNet on the polar warp of the 512-px crop, BCE + soft Dice, AdamW lr 1e-4, batch 4, 30 epochs, bf16 autocast,
no augmentation, crops read from preprocessed_data/ (the crops every model was trained and validated on).
Usage:   SANITY_CHECK_ONLY=1 python3 train_fair_polar.py          (about 1 min: counts, target overlays, one training step)
         SEEDS=1,2,3,4,5 python3 train_fair_polar.py              (3 seeds is enough for the evaluation; 5 if time allows)
Outputs: outputs/phase8_fair/polar/  (seed{k}_best.pt, seed{k}_log.json, sanity/)
"""

import os, json, time, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
exec(
    open("eval_testset_all_models.py").read().split("results = {}")[0]
)  # PlainUNet, to_polar, to_cartesian, dice, DEVICE, coeffs_to_mask
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
SANITY = os.environ.get("SANITY_CHECK_ONLY") == "1"
EPOCHS = 30
PREPROC = os.path.join(BASE, "preprocessed_data")
TARGETS = os.path.join(BASE, "training_targets_v2")
OUT = os.path.join(BASE, "outputs", "phase8_fair", "polar")
os.makedirs(os.path.join(OUT, "sanity"), exist_ok=True)
LOG = open(os.path.join(OUT, "train_log.txt"), "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


say(f"\n==== train_fair_polar  seeds={SEEDS}  device={DEVICE}  out={OUT}")

PHI = build_harmonic_basis(12, N=360)
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)  # outlines were sampled on this grid (contour_to_radial)


def raster(u):
    r = PHI @ u
    pts = np.stack([256 + r * np.cos(GRID), 256 + r * np.sin(GRID)], 1).astype(np.int32).reshape(-1, 1, 2)
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


split = json.load(open(os.path.join(BASE, "project_split_v2.json")))


def build(name):
    items, miss = [], []
    for k in split[name]:
        ds, case = k.split("/", 1)
        npy = os.path.join(PREPROC, ds, f"{case}_normalized.npy")
        npz = os.path.join(TARGETS, f"{ds}_{case}.npz")
        if not (os.path.exists(npy) and os.path.exists(npz)):
            miss.append(k)
            continue
        z = np.load(npz, allow_pickle=True)
        u = z["target"].astype(np.float64)
        hc = bool(z["has_cup"])
        disc = raster(u[:25])
        cup = raster(u[25:]) if hc else np.zeros_like(disc)
        items.append(dict(key=k, npy=npy, disc=disc, cup=cup, hc=hc, u=u))
    return items, miss


t0 = time.time()
train, m1 = build("train")
val, m2 = build("val")
src = lambda its: {
    s: sum(it["key"].startswith(s + "/") for it in its) for s in sorted({it["key"].split("/")[0] for it in its})
}
say(
    f"TRAIN {len(train)} images {src(train)}, with cup outline {sum(it['hc'] for it in train)}, without {sum(not it['hc'] for it in train)}; missing files {len(m1)} {m1[:3]}"
)
say(
    f"VAL   {len(val)} images {src(val)}, with cup outline {sum(it['hc'] for it in val)}; missing files {len(m2)}  ({time.time() - t0:.0f}s)"
)
for it in train:  # polar targets computed once
    it["pt"] = np.stack(
        [(to_polar(np.ascontiguousarray(t.astype(np.float32))) > 0.5) for t in (it["disc"], it["cup"])]
    ).astype(np.float32)


class DS(Dataset):
    def __init__(s, items):
        s.items = items

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        it = s.items[i]
        img = np.load(it["npy"]).astype(np.float32)
        img = np.stack([to_polar(np.ascontiguousarray(img[:, :, c])) for c in range(3)], -1)
        return (
            torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
            torch.from_numpy(it["pt"]),
            torch.tensor(float(it["hc"])),
            i,
        )


class VDS(Dataset):
    def __init__(s, items):
        s.items = items

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        img = np.load(s.items[i]["npy"]).astype(np.float32)
        img = np.stack([to_polar(np.ascontiguousarray(img[:, :, c])) for c in range(3)], -1)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), i


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + eps) / (union + eps)


def loss_fn(p, y, hc):
    # per image and channel: BCE (pixel mean) + soft Dice. With every cup present this equals the baseline loss
    # (BCE mean over both channels + Dice mean over both channels); images without a cup outline drop the cup term.
    l = nn.functional.binary_cross_entropy_with_logits(p.float(), y, reduction="none").mean(dim=(2, 3)) + dice_loss(
        p.float(), y
    )
    return 0.5 * (l[:, 0].mean() + (l[:, 1] * hc).sum() / hc.sum().clamp(min=1))


@torch.no_grad()
def val_dice(model, loader):
    model.eval()
    dd, dc = [], []
    for x, idx in loader:
        P = torch.sigmoid(model(x.to(DEVICE)).float()).cpu().numpy()
        for j, i in enumerate(idx.numpy()):
            it = val[i]
            m = [to_cartesian(np.ascontiguousarray(P[j, c])) > 0.5 for c in range(2)]
            dd.append(dice(m[0], it["disc"] > 0))
            if it["hc"]:
                dc.append(dice(m[1], it["cup"] > 0))
    return float(np.mean(dd)), float(np.mean(dc))


if SANITY:
    chk = [it for it in train if it["key"].startswith("RIGA/")][:3] + [it for it in train if not it["hc"]][:1]
    for it in chk:
        im = np.load(it["npy"])[:, :, 1]
        v = ((im - im.min()) / (np.ptp(im) + 1e-6) * 255).astype(np.uint8)
        v = np.stack([v] * 3, -1)
        for m, col in ((it["disc"], (0, 255, 0)), (it["cup"], (0, 0, 255))):
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cs, -1, col, 2)
        cv2.imwrite(os.path.join(OUT, "sanity", it["key"].replace("/", "_") + ".png"), v)
        rt = to_cartesian(np.ascontiguousarray(it["pt"][0])) > 0.5
        agree = dice(it["disc"] > 0, coeffs_to_mask(it["u"][:25], PHI) > 0)
        say(
            f"  {it['key']}: polar round trip disc Dice {dice(rt, it['disc'] > 0):.4f} (expect > 0.99); "
            f"raster vs project coeffs_to_mask disc Dice {agree:.4f} (expect > 0.99: same drawing grid as the evaluation)"
        )
    model = PlainUNet().to(DEVICE)
    x, y, hc, _ = next(iter(DataLoader(DS(chk), batch_size=4)))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p = model(x.to(DEVICE))
    l = loss_fn(p, y.to(DEVICE), hc.to(DEVICE))
    l.backward()
    say(
        f"  forward/backward OK: {tuple(x.shape)} -> {tuple(p.shape)}, loss {l.item():.3f}, has_cup in batch {hc.tolist()}"
    )
    say(
        f"  overlays in {os.path.join(OUT, 'sanity')}: green disc and red cup must sit on the disc and cup; the DRIONS image shows no cup."
    )
    raise SystemExit(0)

vl = DataLoader(VDS(val), batch_size=4, shuffle=False, num_workers=4)
for seed in SEEDS:
    if os.path.exists(os.path.join(OUT, f"seed{seed}_log.json")):
        say(f"seed {seed} already finished, skipped")
        continue
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(DS(train), batch_size=4, shuffle=True, num_workers=4, generator=g)
    model = PlainUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best, log = -1.0, []
    say(f"\n=== fair polar seed {seed} ===")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, hc, _ in tl:
            x, y, hc = x.to(DEVICE), y.to(DEVICE), hc.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p = model(x)
            loss = loss_fn(p, y, hc)
            loss.backward()
            opt.step()
            tot += loss.item()
        vd, vc = val_dice(model, vl)
        comb = (vd + vc) / 2
        tag = ""
        if comb > best:
            best = comb
            torch.save(model.state_dict(), os.path.join(OUT, f"seed{seed}_best.pt"))
            tag = "  <- best"
        log.append({"epoch": ep, "train_loss": tot / len(tl), "val_disc": vd, "val_cup": vc})
        say(
            f"fair polar seed {seed} epoch {ep:2d}: loss {tot/len(tl):.4f}  val disc {vd*100:.2f}  cup {vc*100:.2f}  ({time.time() - t0:.0f}s){tag}"
        )
    json.dump(log, open(os.path.join(OUT, f"seed{seed}_log.json"), "w"), indent=1)
    say(
        f"fair polar seed {seed} done, best val combined Dice {best*100:.2f} at epoch {1 + int(np.argmax([(r['val_disc'] + r['val_cup'])/2 for r in log]))}"
    )
say("All requested seeds done. Evaluate with: python3 extras_run.py (it picks up >= 3 seeds automatically).")
