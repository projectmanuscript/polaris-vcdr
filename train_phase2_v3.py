import os, json, time, random
import numpy as np, cv2, torch, torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
SPLIT = os.path.join(BASE, "project_split_v2.json")
TARGETS = os.path.join(BASE, "training_targets_v2")
PREPROC = os.path.join(BASE, "preprocessed_data")
OUT = os.path.join(BASE, "outputs", os.environ.get("OUT_NAME", "phase2_v3b"))
os.makedirs(os.path.join(OUT, "sanity"), exist_ok=True)
CFG = dict(
    M=12,
    kappa=1.0,
    lr=1e-4,
    batch=8,
    img=256,
    epochs=80,
    patience=15,
    rot_deg=15.0,
    scale=(0.95, 1.05),
    jitter=(0.9, 1.1),
    seed=int(os.environ.get("SEED", 2026)),
    split="project_split_v2.json",
)
SANITY = os.environ.get("SANITY_CHECK_ONLY") == "1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
print("Device:", DEVICE, flush=True)

# ---------- harmonic helpers (same convention as the existing checkpoint) ----------
N = 360
GRID = np.linspace(-np.pi, np.pi, N, endpoint=False)  # index n <-> drawing angle


def basis(M):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cols = [np.ones(N) / np.sqrt(N)]
    for m in range(1, M + 1):
        cols += [np.sqrt(2 / N) * np.cos(m * t), np.sqrt(2 / N) * np.sin(m * t)]
    return np.stack(cols, 1)


PHI = basis(CFG["M"])


def to_mask(u, size=512, c=(256, 256)):
    r = PHI @ u
    pts = np.stack([c[0] + r * np.cos(GRID), c[1] + r * np.sin(GRID)], 1).astype(np.int32).reshape(-1, 1, 2)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def dice(a, b):
    return float(2 * np.sum(a & b) / (a.sum() + b.sum() + 1e-6))


def rotate_u(u, phi):  # r'(t) = r(t - phi)
    out = u.copy()
    for m in range(1, CFG["M"] + 1):
        a, b = u[2 * m - 1], u[2 * m]
        out[2 * m - 1] = a * np.cos(m * phi) - b * np.sin(m * phi)
        out[2 * m] = a * np.sin(m * phi) + b * np.cos(m * phi)
    return out


# ---------- data ----------
split = json.load(open(SPLIT))


def load_set(name):
    items, missing = [], []
    for k in split[name]:
        ds, case = k.split("/", 1)
        npy = os.path.join(PREPROC, ds, f"{case}_normalized.npy")
        npz = os.path.join(TARGETS, f"{ds}_{case}.npz")
        if os.path.exists(npy) and os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            items.append((k, npy, d["target"].astype(np.float64), bool(d["has_cup"])))
        else:
            missing.append(k)
    return items, missing


sets = {}
for s in ["train", "val", "calib", "test"]:
    sets[s], miss = load_set(s)
    print(
        f"{s:6s}: {len(sets[s])} usable, {sum(it[3] for it in sets[s])} with cup, missing files: {len(miss)} {miss[:3]}",
        flush=True,
    )
bad = [k for k in sum(split.values(), []) if k.split("/")[-1] in ("MagrabiaMale_image8", "MagrabiaMale_image11")]
print("Misregistered images present in any split:", bad)
T = np.stack([it[2] for it in sets["train"]])
HC = np.array([it[3] for it in sets["train"]])
tmean, tstd = T.mean(0), T.std(0)
tmean[25:], tstd[25:] = T[HC, 25:].mean(0), T[HC, 25:].std(0)  # cup statistics from cup-bearing images only


def augment(img, u, hc, ang=None, sc=None):
    ang = random.uniform(-CFG["rot_deg"], CFG["rot_deg"]) if ang is None else ang
    sc = random.uniform(*CFG["scale"]) if sc is None else sc
    A = cv2.getRotationMatrix2D((256, 256), ang, sc)
    img = cv2.warpAffine(img, A, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    u = u.copy()
    phi = ROT_SIGN * np.radians(ang)
    u[:25] = rotate_u(u[:25], phi) * sc
    if hc:
        u[25:] = rotate_u(u[25:], phi) * sc
    jit = np.array([random.uniform(*CFG["jitter"]) for _ in range(3)], np.float32)
    return img * jit[None, None, :], u


class DS(Dataset):
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
        y = (u - tmean) / tstd
        if not hc:
            y[25:] = 0
        return x, torch.tensor(y, dtype=torch.float32), torch.tensor(hc)


# ---------- verify augmentation direction against a real warped mask (stops if wrong) ----------
def find_rot_sign():
    k, npy, u, hc = next(it for it in sets["train"] if it[3])
    res = {}
    for sign in (+1, -1):
        ds = []
        for ang, sc in [(12, 1.04), (-9, 0.96)]:
            A = cv2.getRotationMatrix2D((256, 256), ang, sc)
            for sl in (slice(0, 25), slice(25, 50)):
                mw = cv2.warpAffine(to_mask(u[sl]).astype(np.uint8), A, (512, 512), flags=cv2.INTER_NEAREST) > 0
                ds.append(dice(to_mask(rotate_u(u[sl], sign * np.radians(ang)) * sc), mw))
        res[sign] = min(ds)
    best = max(res, key=res.get)
    print(f"Augmentation check (min Dice over 4 tests) per sign: {res} -> using sign {best}", flush=True)
    assert res[best] > 0.97, "Coefficient transform does not match the image warp. Do not train."
    return best


ROT_SIGN = find_rot_sign()


# ---------- model ----------
class HarmonicRegressor(nn.Module):  # same attribute names as the original checkpoint
    def __init__(self, out_dim=50):
        super().__init__()
        self.features = nn.Sequential(*list(tvm.resnet34(weights=None).children())[:-1])
        self.head = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def load_pretrained(model):
    cands = [
        os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"),
        os.path.join(BASE, "resnet34-b627a593.pth"),
    ]
    p = next((c for c in cands if os.path.exists(c)), None)
    if p is None:
        raise SystemExit(
            "ImageNet ResNet-34 weights not found: copy resnet34-b627a593.pth into ~/POLARIS/BTP-Datasets/ and rerun."
        )
    r = tvm.resnet34(weights=None)
    r.load_state_dict(torch.load(p, map_location="cpu"))
    model.features.load_state_dict(nn.Sequential(*list(r.children())[:-1]).state_dict())
    print("Loaded ImageNet weights:", p)


w = [1.0] + sum([[1 / (1 + m) ** CFG["kappa"]] * 2 for m in range(1, CFG["M"] + 1)], [])
BW = torch.tensor(np.concatenate([w, w]), dtype=torch.float32, device=DEVICE)


def wloss(p, t, hc):
    d = (p - t) ** 2 * BW
    cm = hc.float().unsqueeze(1)
    return d[:, :25].mean() + ((d[:, 25:] * cm).sum(1) / 25).sum() / hc.float().sum().clamp(min=1)


model = HarmonicRegressor().to(DEVICE)
load_pretrained(model)

if SANITY:

    def overlay(img, u, hc, path):
        g = img[:, :, 1]
        g = ((g - g.min()) / (np.ptp(g) + 1e-6) * 255).astype(np.uint8)
        vis = np.stack([g] * 3, -1)
        for sl, col in [(slice(0, 25), (0, 255, 0)), (slice(25, 50), (0, 0, 255))]:
            if sl.start == 25 and not hc:
                continue
            cs, _ = cv2.findContours(to_mask(u[sl]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, col, 2)
        cv2.imwrite(path, vis)

    for i, (k, npy, u, hc) in enumerate([it for it in sets["train"] if it[3]][:3]):
        img = np.load(npy).astype(np.float32)
        overlay(img, u, hc, os.path.join(OUT, "sanity", f"{i}_original.png"))
        ia, ua = augment(img, u, hc, ang=15.0, sc=1.05)
        overlay(ia, ua, hc, os.path.join(OUT, "sanity", f"{i}_augmented_rot15_scale105.png"))
    x, y, hc = next(iter(DataLoader(DS(sets["train"], True), batch_size=4, shuffle=True)))
    p = model(x.to(DEVICE))
    l = wloss(p, y.to(DEVICE), hc.to(DEVICE))
    l.backward()
    print(f"Forward/backward OK: input {tuple(x.shape)} -> output {tuple(p.shape)}, loss {l.item():.3f}")
    print(
        "Overlays saved in",
        os.path.join(OUT, "sanity"),
        "- contours must sit on the disc and cup in BOTH original and augmented images.",
    )
    raise SystemExit(0)

# ---------- training ----------
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
tl = DataLoader(DS(sets["train"], True), batch_size=CFG["batch"], shuffle=True, num_workers=4)
vl = DataLoader(DS(sets["val"], False), batch_size=16, shuffle=False, num_workers=4)


def evaluate():
    model.eval()
    tot, n, dd, dc = 0.0, 0, [], []
    with torch.no_grad():
        for x, y, hc in vl:
            p = model(x.to(DEVICE))
            tot += wloss(p, y.to(DEVICE), hc.to(DEVICE)).item() * len(x)
            n += len(x)
            pr = p.cpu().numpy() * tstd + tmean
            tr = y.numpy() * tstd + tmean
            for i in range(len(x)):
                dd.append(dice(to_mask(pr[i, :25]), to_mask(tr[i, :25])))
                if hc[i]:
                    dc.append(dice(to_mask(pr[i, 25:]), to_mask(tr[i, 25:])))
    return tot / n, float(np.mean(dd)), float(np.mean(dc))


best, best_dice, bad_epochs, hist = float("inf"), -1.0, 0, []
json.dump(
    {**CFG, "scale": list(CFG["scale"]), "jitter": list(CFG["jitter"]), "rot_sign": int(ROT_SIGN)},
    open(os.path.join(OUT, "config.json"), "w"),
    indent=1,
)
for ep in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    model.train()
    tr_loss, nb = 0.0, 0
    for x, y, hc in tl:
        opt.zero_grad()
        l = wloss(model(x.to(DEVICE)), y.to(DEVICE), hc.to(DEVICE))
        l.backward()
        opt.step()
        tr_loss += l.item()
        nb += 1
    sched.step()
    vloss, vdd, vdc = evaluate()
    hist.append({"epoch": ep, "train_loss": tr_loss / nb, "val_loss": vloss, "val_disc_dice": vdd, "val_cup_dice": vdc})
    state = {
        "model_state": model.state_dict(),
        "target_mean": tmean,
        "target_std": tstd,
        "epoch": ep,
        "val_loss": vloss,
        "config": CFG,
    }
    torch.save(state, os.path.join(OUT, "last.pt"))
    tag = ""
    vcomb = (vdd + vdc) / 2
    if vloss < best:
        best = vloss
        tag += "  <- best loss"
        torch.save(state, os.path.join(OUT, "best_loss.pt"))
    if vcomb > best_dice:
        best_dice, bad_epochs = vcomb, 0
        tag += "  <- best Dice (primary)"
        torch.save(state, os.path.join(OUT, "best_dice.pt"))
    else:
        bad_epochs += 1
    print(
        f"Epoch {ep:3d}: train {tr_loss/nb:.4f}  val {vloss:.4f}  val disc {vdd*100:.2f}  val cup {vdc*100:.2f}  ({time.time()-t0:.0f}s){tag}",
        flush=True,
    )
    json.dump(hist, open(os.path.join(OUT, "history.json"), "w"), indent=1)
    if bad_epochs >= CFG["patience"]:
        print(f"Early stop: no val Dice improvement for {CFG['patience']} epochs.")
        break
print(f"Done. Best val loss {best:.4f}, best val Dice {best_dice*100:.2f}. Checkpoints in {OUT}")
