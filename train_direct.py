"""
Direct-regression controls for POLARIS ("is it the harmonics, or just regression?").
Setup fixed before any run:
  - Everything except the output is POLARIS's own training script: this file executes train_phase2_v3.py (the server copy)
    up to its training loop, so the data split, crops, target outlines (rater-mean, training_targets_v2), input size,
    augmentation (rotation +-15 deg, scale 0.95-1.05, intensity x0.9-1.1, with the verified rotation sign), ImageNet
    ResNet-34 backbone, AdamW, learning rate, batch, epochs, cosine schedule and patience are taken from it unchanged.
  - Only the head and the loss change:
      VARIANT=radii : 720 outputs = 360 disc radii + 360 cup radii of the same target outline (no harmonic basis at the
                      output). Loss: MSE on per-angle z-scored radii; cup term masked where no cup outline exists.
      VARIANT=vcdr  : 1 output = the vertical-extent vCDR of the same target outline (computed after augmentation, since
                      rotation changes it). Loss: MSE on the z-scored value; images with no cup outline skipped.
  - Epoch selection on validation: radii by mean(disc Dice, cup Dice) (as POLARIS's best_dice); vcdr by vCDR MAE.
    Early stop after `patience` epochs without improvement of that metric.
  - 5 seeds (1-5), ensembled by averaging outputs, exactly as POLARIS's 5-seed ensemble.
  - Evaluation (eval_direct.py) uses the same test images, references and statistics as the paper. Nothing is tuned on test.
Usage:  SANITY_DIRECT=1 VARIANT=radii python3 train_direct.py        (checks, no training)
        VARIANT=radii SEEDS=1,2,3,4,5 python3 train_direct.py ;  VARIANT=vcdr SEEDS=1,2,3,4,5 python3 train_direct.py
Outputs: outputs/phase9_direct/<variant>/seed{k}_best.pt, seed{k}_log.json, train_log.txt
"""

import os, json, time, random

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
os.chdir(BASE)
VARIANT = os.environ.get("VARIANT", "")
assert VARIANT in ("radii", "vcdr"), "set VARIANT=radii or VARIANT=vcdr"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
SANITY = os.environ.get("SANITY_DIRECT") == "1"
os.environ["SANITY_CHECK_ONLY"] = "0"
os.environ["OUT_NAME"] = "phase9_direct_shared"
os.environ.setdefault("SEED", str(SEEDS[0]))

src = open("train_phase2_v3.py").read()
MARK = "# ---------- training ----------"
assert src.count(MARK) == 1, "training marker not found in train_phase2_v3.py"
exec(src.split(MARK)[0])  # data, targets, augment (sign-checked), backbone, CFG
for nm in [
    "sets",
    "augment",
    "PHI",
    "GRID",
    "CFG",
    "to_mask",
    "dice",
    "load_pretrained",
    "HarmonicRegressor",
    "DEVICE",
]:
    assert nm in globals(), f"{nm} missing from train_phase2_v3.py"
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

OUT = os.path.join(BASE, "outputs", "phase9_direct", VARIANT)
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "train_log.txt"), "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


CFGs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}
say(
    f"\n==== train_direct VARIANT={VARIANT} seeds={SEEDS} device={DEVICE}\n   POLARIS settings reused: {json.dumps(CFGs)}"
)

SING = np.sin(GRID)


def ext_r(r):
    y = r * SING
    return y.max() - y.min()


def vcdr_of(u):
    return ext_r(PHI @ u[25:]) / ext_r(PHI @ u[:25])


def radii_mask(r, size=512, c=(256, 256)):
    pts = np.stack([c[0] + r * np.cos(GRID), c[1] + r * np.sin(GRID)], 1).astype(np.int32).reshape(-1, 1, 2)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


TR = sets["train"]
hcm = np.array([it[3] for it in TR])
if VARIANT == "radii":
    R = np.stack([np.concatenate([PHI @ it[2][:25], PHI @ it[2][25:]]) for it in TR])
    nm_, ns_ = R.mean(0), R.std(0)
    nm_[360:], ns_[360:] = R[hcm, 360:].mean(0), R[hcm, 360:].std(0)
    OUT_DIM = 720
else:
    V = np.array([vcdr_of(it[2]) for it in TR if it[3]])
    nm_, ns_ = np.array([V.mean()]), np.array([V.std()])
    OUT_DIM = 1
say(f"   train {len(TR)} images ({hcm.sum()} with cup outline), val {len(sets['val'])}; output dim {OUT_DIM}")


def target(u, hc):
    if VARIANT == "radii":
        y = (np.concatenate([PHI @ u[:25], PHI @ u[25:]]) - nm_) / ns_
        if not hc:
            y[360:] = 0
    else:
        y = (np.array([vcdr_of(u)]) - nm_) / ns_ if hc else np.zeros(1)
    return y


class DD(Dataset):
    def __init__(s, items, aug):
        s.items, s.aug = items, aug

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        k, npy, u, hc = s.items[i]
        img = np.load(npy).astype(np.float32)
        if s.aug:
            img, u = augment(img, u, hc)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        x = torch.nn.functional.interpolate(x[None], size=CFG["img"], mode="bilinear", align_corners=False)[0]
        return x, torch.tensor(target(u, hc), dtype=torch.float32), torch.tensor(hc), i


def build_model(pretrained=True):
    m = HarmonicRegressor()
    if pretrained:
        load_pretrained(m)
    m.head = nn.Linear(m.head.in_features, OUT_DIM)
    return m.to(DEVICE)


def loss_fn(p, y, hc):
    d = (p - y) ** 2
    c = hc.float()
    if VARIANT == "vcdr":
        return (d[:, 0] * c).sum() / c.sum().clamp(min=1)
    return d[:, :360].mean() + (d[:, 360:].mean(1) * c).sum() / c.sum().clamp(min=1)


@torch.no_grad()
def val_metric(model, loader, items):
    model.eval()
    dd, dc, ae = [], [], []
    for x, _, _, idx in loader:
        P = model(x.to(DEVICE)).cpu().numpy() * ns_ + nm_
        for j, i in enumerate(idx.numpy()):
            k, npy, u, hc = items[i]
            if VARIANT == "radii":
                dd.append(dice(radii_mask(P[j, :360]), to_mask(u[:25])))
                if hc:
                    dc.append(dice(radii_mask(P[j, 360:]), to_mask(u[25:])))
            elif hc:
                ae.append(abs(P[j, 0] - vcdr_of(u)))
    if VARIANT == "radii":
        return (np.mean(dd) + np.mean(dc)) / 2, f"val disc {np.mean(dd)*100:.2f}  cup {np.mean(dc)*100:.2f}", True
    return float(np.mean(ae)), f"val vCDR MAE {np.mean(ae):.4f}", False


if SANITY:
    k, npy, u, hc = next(it for it in TR if it[3])
    img = np.load(npy).astype(np.float32)
    r = np.concatenate([PHI @ u[:25], PHI @ u[25:]])
    say(
        f"   radii target reproduces the outline: Dice disc {dice(radii_mask(r[:360]), to_mask(u[:25])):.4f}, cup {dice(radii_mask(r[360:]), to_mask(u[25:])):.4f} (expect ~1)"
    )
    random.seed(0)
    ia, ua = augment(img, u, hc)
    say(f"   vCDR target before / after one augmentation: {vcdr_of(u):.4f} / {vcdr_of(ua):.4f}")
    m = build_model()
    x, y, h, _ = next(iter(DataLoader(DD(TR, True), batch_size=4, shuffle=True)))
    l = loss_fn(m(x.to(DEVICE)), y.to(DEVICE), h.to(DEVICE))
    l.backward()
    say(f"   forward/backward OK: {tuple(x.shape)} -> {OUT_DIM} outputs, loss {l.item():.3f}")
    raise SystemExit(0)

vl = DataLoader(DD(sets["val"], False), batch_size=16, shuffle=False, num_workers=4)
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
        DD(TR, True),
        batch_size=CFG["batch"],
        shuffle=True,
        num_workers=4,
        generator=g,
        worker_init_fn=lambda w: (random.seed(seed * 100 + w), np.random.seed(seed * 100 + w)),
    )
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    best, bad, log = None, 0, []
    say(f"\n=== {VARIANT} seed {seed} ===")
    for ep in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, y, h, _ in tl:
            opt.zero_grad()
            l = loss_fn(model(x.to(DEVICE)), y.to(DEVICE), h.to(DEVICE))
            l.backward()
            opt.step()
            tot += l.item()
            nb += 1
        sched.step()
        met, txt, higher = val_metric(model, vl, sets["val"])
        tag = ""
        if best is None or (met > best if higher else met < best):
            best, bad, tag = met, 0, "  <- best"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": VARIANT,
                    "out_dim": OUT_DIM,
                    "norm_mean": nm_,
                    "norm_std": ns_,
                    "img": CFG["img"],
                    "epoch": ep,
                    "val_metric": met,
                },
                os.path.join(OUT, f"seed{seed}_best.pt"),
            )
        else:
            bad += 1
        log.append({"epoch": ep, "train_loss": tot / nb, "val_metric": float(met)})
        say(f"{VARIANT} seed {seed} epoch {ep:3d}: loss {tot/nb:.4f}  {txt}  ({time.time() - t0:.0f}s){tag}")
        if bad >= CFG["patience"]:
            say(f"   early stop: no improvement for {CFG['patience']} epochs")
            break
    json.dump(log, open(os.path.join(OUT, f"seed{seed}_log.json"), "w"), indent=1)
    say(f"{VARIANT} seed {seed} done, best val {'Dice' if VARIANT == 'radii' else 'vCDR MAE'} {best:.4f}")
say(f"All requested seeds done for {VARIANT}. Evaluate with: python3 eval_direct.py")
