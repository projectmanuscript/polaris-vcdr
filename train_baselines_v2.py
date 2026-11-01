"""
Retrain the four Phase 8 baselines on the clean split (project_split_v2.json), like-for-like with the harmonic ensemble:
  - RIGA training images of the new split (as before, baselines use RIGA only), 5 seeds per baseline
  - epoch chosen on the 64-image validation set by mean disc+cup Dice (same rule as before and as the harmonic model)
  - the 5 seeds are averaged (probabilities) at test time, scored on the same 111 test images and fused masks as every earlier result
  - architectures, loss, optimiser, batch size and 30 epochs are unchanged from the original scripts (model classes are taken
    from eval_testset_all_models.py, which loads the original checkpoints, so attribute names are guaranteed identical)
  - training masks use the same rater filter as the harmonic targets (star shape, angular gap, self-intersection, unstable list)
Usage:  MODEL=plain|attention|deepres|polar  [SEEDS=1,2,3,4,5]  [SANITY_CHECK_ONLY=1]  [EVAL_ONLY=1]  python3 train_baselines_v2.py
Outputs: outputs/phase8_v2/<MODEL>/
"""

import os, json, re, time, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

exec(
    open('eval_testset_all_models.py').read().split('results = {}')[0]
)  # model classes, fixed polar warps, test set, helpers

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
MODEL = os.environ.get("MODEL", "plain")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
SANITY = os.environ.get("SANITY_CHECK_ONLY") == "1"
EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"
EPOCHS = 30
CLASSES = {"plain": PlainUNet, "attention": AttentionUNet, "deepres": DeepResUNet, "polar": PlainUNet}
assert MODEL in CLASSES, f"MODEL must be one of {list(CLASSES)}"
IS_POLAR = MODEL == "polar"
OUT = os.path.join(BASE, "outputs", "phase8_v2", MODEL)
os.makedirs(OUT, exist_ok=True)
print(f"MODEL={MODEL} seeds={SEEDS} device={DEVICE} out={OUT}", flush=True)

# ---------- training / validation masks with the same rater filter as the harmonic targets ----------
split = json.load(open(os.path.join(BASE, "project_split_v2.json")))
reg = json.load(open(os.path.join(BASE, "exclusion_registry.json")))
_un = reg.get("extraction_unstable", [])
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
unstable = {norm(e) for e in (_un.keys() if isinstance(_un, dict) else _un)}


def valid_contour(crop, c=(256, 256)):
    if len(crop) < 10 or not check_star_shaped(crop, c):
        return False
    a = np.sort(np.arctan2(crop[:, 1] - c[1], crop[:, 0] - c[0]))
    if np.degrees(np.diff(np.concatenate([a, [a[0] + 2 * np.pi]])).max()) > 25:
        return False
    x, y = crop[:, 0], crop[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [crop.astype(np.int32).reshape(-1, 1, 2)], 1)
    return 0.85 <= area / max(m.sum(), 1) <= 1.15


def fused_masks(key):
    ds, case = key.split("/", 1)
    rec = json.load(open(os.path.join(PROCESSED_DIR, ds, case + ".json")))
    tr = json.load(open(os.path.join(PREPROC_DIR, ds, case + "_transform.json")))
    rd, rc = [], []
    for rid, data in rec["raters"].items():
        if norm(case.replace("_", "/", 1) + "_rater" + re.sub(r"\D", "", str(rid))) in unstable:
            continue
        for sk, acc in (("disc_contour", rd), ("cup_contour", rc)):
            cpts = data.get(sk)
            if cpts is None:
                continue
            crop = map_to_crop_space(cpts, tr["x_min"], tr["y_min"], tr["scale"])
            if valid_contour(crop):
                acc.append(contour_to_radial(crop)[1])
    disc = radial_to_mask(np.mean(rd, 0)) > 0 if rd else None
    cup = radial_to_mask(np.mean(rc, 0)) > 0 if rc else None
    return disc, cup


def build(name):
    out = []
    for k in split[name]:
        if not k.startswith("RIGA/"):
            continue
        d, c = fused_masks(k)
        if d is None:
            continue
        ds, case = k.split("/", 1)
        out.append(
            dict(
                key=k,
                npy=os.path.join(PREPROC_DIR, ds, case + "_normalized.npy"),
                disc=d.astype(np.uint8),
                cup=(c if c is not None else np.zeros_like(d)).astype(np.uint8),
            )
        )
    return out


class DS(Dataset):
    def __init__(s, items):
        s.items = items

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        it = s.items[i]
        img = np.load(it["npy"]).astype(np.float32)
        tgt = np.stack([it["disc"], it["cup"]]).astype(np.float32)
        if IS_POLAR:
            img = np.stack([to_polar(np.ascontiguousarray(img[:, :, c])) for c in range(3)], -1)
            tgt = np.stack([(to_polar(np.ascontiguousarray(t)) > 0.5).astype(np.float32) for t in tgt])
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), torch.from_numpy(tgt), i


def predict_probs(models, x):
    with torch.no_grad():
        return (sum(torch.sigmoid(m(x).float()) for m in models) / len(models)).cpu().numpy()


def to_cart_mask(p2):
    if IS_POLAR:
        return np.stack([to_cartesian(np.ascontiguousarray(p2[c])) > 0.5 for c in range(2)])
    return p2 > 0.5


def val_dice(model, loader, items):
    model.eval()
    dd, dc = [], []
    for x, _, idx in loader:
        P = predict_probs([model], x.to(DEVICE))
        for j, i in enumerate(idx.numpy()):
            m = to_cart_mask(P[j])
            it = items[i]
            dd.append(dice(m[0], it["disc"] > 0))
            dc.append(dice(m[1], it["cup"] > 0))
    return float(np.mean(dd)), float(np.mean(dc))


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + eps) / (union + eps)


bce = nn.BCEWithLogitsLoss()


def evaluate_test():
    paths = [os.path.join(OUT, f"seed{s}_best.pt") for s in (1, 2, 3, 4, 5)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("Test evaluation skipped; missing:", missing)
        return
    models = []
    for p in paths:
        m = CLASSES[MODEL]().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)
    res = []
    for s in test_samples:
        img = np.load(s["npy"]).astype(np.float32)
        if IS_POLAR:
            img = np.stack([to_polar(np.ascontiguousarray(img[:, :, c])) for c in range(3)], -1)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].to(DEVICE)
        m = to_cart_mask(predict_probs(models, x)[0])
        td = s["disc_mask"] > 0
        tc = s["cup_mask"] > 0 if s["cup_mask"] is not None else np.zeros_like(td)
        res.append({"case": s["case"], "disc": float(dice(m[0], td)), "cup": float(dice(m[1], tc))})
    json.dump(res, open(os.path.join(OUT, "test_results.json"), "w"), indent=1)
    print(
        f"TEST ({len(res)} images, 5-seed ensemble): disc {np.mean([r['disc'] for r in res])*100:.2f}  cup {np.mean([r['cup'] for r in res])*100:.2f}",
        flush=True,
    )


if EVAL_ONLY:
    evaluate_test()
    raise SystemExit(0)

t0 = time.time()
train, val = build("train"), build("val")
print(f"Train {len(train)} RIGA images, val {len(val)}  ({time.time()-t0:.0f}s to build masks)", flush=True)
vl = DataLoader(DS(val), batch_size=4, shuffle=False, num_workers=4)

if SANITY:
    if IS_POLAR:
        m0 = val[0]["disc"].astype(np.float32)
        rt = to_cartesian(to_polar(np.ascontiguousarray(m0))) > 0.5
        print(f"Polar round trip on a real disc mask: Dice {dice(rt, m0 > 0):.4f} (expect > 0.99)")
    model = CLASSES[MODEL]().to(DEVICE)
    x, y, _ = next(iter(DataLoader(DS(train), batch_size=2, shuffle=True)))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p = model(x.to(DEVICE))
        loss = bce(p, y.to(DEVICE)) + dice_loss(p, y.to(DEVICE)).mean()
    loss.backward()
    print(
        f"Forward/backward OK: {tuple(x.shape)} -> {tuple(p.shape)}, loss {loss.item():.3f}, params {sum(q.numel() for q in model.parameters())/1e6:.1f}M"
    )
    raise SystemExit(0)

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(DS(train), batch_size=4, shuffle=True, num_workers=4, generator=g)
    model = CLASSES[MODEL]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best, log = -1.0, []
    print(f"\n=== {MODEL} seed {seed} ===", flush=True)
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, _ in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p = model(x)
                loss = bce(p, y) + dice_loss(p, y).mean()
            loss.backward()
            opt.step()
            tot += loss.item()
        vd, vc = val_dice(model, vl, val)
        comb = (vd + vc) / 2
        tag = ""
        if comb > best:
            best = comb
            torch.save(model.state_dict(), os.path.join(OUT, f"seed{seed}_best.pt"))
            tag = "  <- best"
        log.append({"epoch": ep, "train_loss": tot / len(tl), "val_disc": vd, "val_cup": vc})
        print(
            f"{MODEL} seed {seed} epoch {ep:2d}: loss {tot/len(tl):.4f}  val disc {vd*100:.2f}  cup {vc*100:.2f}  ({time.time()-t0:.0f}s){tag}",
            flush=True,
        )
    json.dump(log, open(os.path.join(OUT, f"seed{seed}_log.json"), "w"), indent=1)
    print(f"{MODEL} seed {seed} done, best val combined Dice {best*100:.2f}", flush=True)
evaluate_test()
