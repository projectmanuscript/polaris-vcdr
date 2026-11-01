"""
Pretrained, augmented U-Net baseline: U-Net with an ImageNet ResNet-34 encoder (rn34_unet.py), 5 seeds.
Same data, targets and augmentation as POLARIS:
  - training images: the 537 POLARIS training images (RIGA + DRIONS-DB), crops from preprocessed_data/
  - targets: masks rasterised from training_targets_v2 (the rater-mean outlines POLARIS regresses); cup loss masked
    where no cup outline exists (same rule as the matched polar U-Net)
  - augmentation (POLARIS's): rotation +-15 deg and scale 0.95-1.05 about the crop centre (masks warped with the same
    matrix), per-channel intensity x U(0.9, 1.1)
Same as the other U-Nets: BCE + soft Dice, AdamW lr 1e-4, batch 4, 30 epochs, bf16 autocast, epoch selected on validation by
mean(disc Dice, cup Dice).
Usage:  SANITY_CHECK_ONLY=1 python3 train_rn34_unet.py      then      SEEDS=1,2,3,4,5 python3 train_rn34_unet.py
Outputs: outputs/phase8_rn34/  (seed{k}_best.pt, seed{k}_log.json, train_log.txt, sanity/)
"""

import os, json, time, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
exec(open("eval_testset_all_models.py").read().split("results = {}")[0])  # dice, DEVICE, build_harmonic_basis
from rn34_unet import ResUNet34

W_IMAGENET = next(
    (
        p
        for p in [
            os.path.join(BASE, "resnet34-b627a593.pth"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"),
        ]
        if os.path.exists(p)
    ),
    None,
)
assert W_IMAGENET, "copy resnet34-b627a593.pth into the data root"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
SANITY = os.environ.get("SANITY_CHECK_ONLY") == "1"
EPOCHS = 30
PREPROC = os.path.join(BASE, "preprocessed_data")
TARGETS = os.path.join(BASE, "training_targets_v2")
OUT = os.path.join(BASE, "outputs", "phase8_rn34")
os.makedirs(os.path.join(OUT, "sanity"), exist_ok=True)
LOG = open(os.path.join(OUT, "train_log.txt"), "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


say(f"\n==== train_rn34_unet  seeds={SEEDS}  device={DEVICE}  imagenet={W_IMAGENET}")

PHI = build_harmonic_basis(12, N=360)
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)


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
        items.append(
            dict(
                key=k, npy=npy, disc=raster(u[:25]), cup=raster(u[25:]) if hc else np.zeros((512, 512), np.uint8), hc=hc
            )
        )
    return items, miss


train, m1 = build("train")
val, m2 = build("val")
say(
    f"TRAIN {len(train)} images, with cup outline {sum(it['hc'] for it in train)}, missing {len(m1)}; VAL {len(val)}, missing {len(m2)}"
)


def augment(img, d, c):  # identical parameters to POLARIS (train_phase2_v3.py)
    ang = random.uniform(-15.0, 15.0)
    sc = random.uniform(0.95, 1.05)
    A = cv2.getRotationMatrix2D((256, 256), ang, sc)
    img = cv2.warpAffine(img, A, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    d = cv2.warpAffine(d, A, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0)
    c = cv2.warpAffine(c, A, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0)
    jit = np.array([random.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    return img * jit[None, None, :], d, c


class DS(Dataset):
    def __init__(s, items, aug):
        s.items, s.aug = items, aug

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        it = s.items[i]
        img = np.load(it["npy"]).astype(np.float32)
        d, c = it["disc"], it["cup"]
        if s.aug:
            img, d, c = augment(img, d, c)
        return (
            torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
            torch.from_numpy(np.stack([d, c]).astype(np.float32)),
            torch.tensor(float(it["hc"])),
            i,
        )


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    return 1 - (2 * (p * t).sum(dim=(2, 3)) + eps) / (p.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) + eps)


def loss_fn(p, y, hc):
    l = nn.functional.binary_cross_entropy_with_logits(p.float(), y, reduction="none").mean(dim=(2, 3)) + dice_loss(
        p.float(), y
    )
    return 0.5 * (l[:, 0].mean() + (l[:, 1] * hc).sum() / hc.sum().clamp(min=1))


@torch.no_grad()
def val_dice(model, loader):
    model.eval()
    dd, dc = [], []
    for x, _, _, idx in loader:
        P = torch.sigmoid(model(x.to(DEVICE)).float()).cpu().numpy()
        for j, i in enumerate(idx.numpy()):
            it = val[i]
            dd.append(dice(P[j, 0] > 0.5, it["disc"] > 0))
            if it["hc"]:
                dc.append(dice(P[j, 1] > 0.5, it["cup"] > 0))
    return float(np.mean(dd)), float(np.mean(dc))


if SANITY:
    for it in train[:2]:
        img = np.load(it["npy"]).astype(np.float32)
        random.seed(0)
        ia, da, ca = augment(img, it["disc"], it["cup"])
        for tag, (im, d, c) in [("orig", (img, it["disc"], it["cup"])), ("aug", (ia, da, ca))]:
            g = im[:, :, 1]
            v = ((g - g.min()) / (np.ptp(g) + 1e-6) * 255).astype(np.uint8)
            v = np.stack([v] * 3, -1)
            for m, col in ((d, (0, 158, 115)), (c, (178, 114, 0))):
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cs, -1, col, 2)
            cv2.imwrite(os.path.join(OUT, "sanity", it["key"].replace("/", "_") + f"_{tag}.png"), v)
    model = ResUNet34(imagenet_weights=W_IMAGENET).to(DEVICE)
    x, y, hc, _ = next(iter(DataLoader(DS(train, True), batch_size=4, shuffle=True)))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p = model(x.to(DEVICE))
    l = loss_fn(p, y.to(DEVICE), hc.to(DEVICE))
    l.backward()
    say(
        f"  forward/backward OK: {tuple(x.shape)} -> {tuple(p.shape)}, loss {l.item():.3f}, params {sum(q.numel() for q in model.parameters())/1e6:.1f}M"
    )
    say(
        f"  overlays in {os.path.join(OUT, 'sanity')}: outlines must sit on the disc and cup in both the original and the augmented image"
    )
    raise SystemExit(0)

vl = DataLoader(DS(val, False), batch_size=4, shuffle=False, num_workers=4)
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
    tl = DataLoader(
        DS(train, True),
        batch_size=4,
        shuffle=True,
        num_workers=4,
        generator=g,
        worker_init_fn=lambda w: (random.seed(seed * 100 + w), np.random.seed(seed * 100 + w)),
    )
    model = ResUNet34(imagenet_weights=W_IMAGENET).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best, log = -1.0, []
    say(f"\n=== rn34 U-Net seed {seed} ===")
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
            f"rn34 seed {seed} epoch {ep:2d}: loss {tot/len(tl):.4f}  val disc {vd*100:.2f}  cup {vc*100:.2f}  ({time.time() - t0:.0f}s){tag}"
        )
    json.dump(log, open(os.path.join(OUT, f"seed{seed}_log.json"), "w"), indent=1)
    say(
        f"rn34 seed {seed} done, best val combined Dice {best*100:.2f} at epoch {1 + int(np.argmax([(r['val_disc'] + r['val_cup'])/2 for r in log]))}"
    )
say("All requested seeds done. Evaluate with: python3 eval_rn34_and_figs.py")
