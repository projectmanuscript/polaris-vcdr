import os, json, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as tvmodels

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

BASE = os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets"))
PROCESSED_DIR = os.path.join(BASE, "processed_data_v2")
PREPROC_DIR = os.path.join(BASE, "preprocessed_data")
SPLIT_PATH = os.path.join(BASE, "project_split.json")
OUT_PATH = os.path.join(
    os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")),
    "outputs/phase8/phase8_testset_eval_v4.json",
)


# ---------- shared helpers (same as every training script) ----------
def contour_to_radial(contour_pts, center=(256, 256), n_angles=360):
    pts = np.array(contour_pts, dtype=float)
    cx, cy = center
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx**2 + dy**2)
    order = np.argsort(angles)
    query = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    r = np.interp(query, angles[order], radii[order], period=2 * np.pi)
    return query, r


def check_star_shaped(contour_pts, center=(256, 256), tolerance=0.15):
    pts = np.array(contour_pts, dtype=float)
    cx, cy = center
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    unwrapped = np.unwrap(angles)
    diffs = np.diff(unwrapped)
    net_dir = np.sign(unwrapped[-1] - unwrapped[0]) or 1
    return np.sum(diffs * net_dir < -tolerance) == 0


def map_to_crop_space(contour_pts, x_min, y_min, scale):
    return (np.array(contour_pts) - [x_min, y_min]) * scale


def radial_to_mask(radii, center=(256, 256), img_size=512, n_angles=360):
    theta = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    x = center[0] + radii * np.cos(theta)
    y = center[1] + radii * np.sin(theta)
    pts = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def fuse_raters_to_mask(record, transform, structure_key, center=(256, 256)):
    x_min, y_min, scale = transform["x_min"], transform["y_min"], transform["scale"]
    all_radii = []
    for rid, data in record["raters"].items():
        contour = data.get(structure_key)
        if contour is None:
            continue
        crop = map_to_crop_space(contour, x_min, y_min, scale)
        if not check_star_shaped(crop, center):
            continue
        _, r = contour_to_radial(crop, center)
        all_radii.append(r)
    if not all_radii:
        return None
    return radial_to_mask(np.mean(all_radii, axis=0), center)


def dice(pred_bin, true_bin):
    inter = np.sum(pred_bin & true_bin)
    return 2 * inter / (np.sum(pred_bin) + np.sum(true_bin) + 1e-6)


# ---------- build the held-out test set (129 images), same filter every script has used ----------
with open(SPLIT_PATH) as f:
    split = json.load(f)
test_keys = set(split["test"])

test_samples = []
for jf in glob.glob(os.path.join(PROCESSED_DIR, "RIGA", "*.json")):
    case_name = os.path.splitext(os.path.basename(jf))[0]
    key = f"RIGA/{case_name}"
    if key not in test_keys:
        continue
    transform_path = os.path.join(PREPROC_DIR, "RIGA", f"{case_name}_transform.json")
    npy_path = os.path.join(PREPROC_DIR, "RIGA", f"{case_name}_normalized.npy")
    if not (os.path.exists(transform_path) and os.path.exists(npy_path)):
        continue
    with open(jf) as f:
        record = json.load(f)
    with open(transform_path) as f:
        transform = json.load(f)
    disc_mask = fuse_raters_to_mask(record, transform, "disc_contour")
    cup_mask = fuse_raters_to_mask(record, transform, "cup_contour")
    if disc_mask is None:
        continue
    test_samples.append({"case": case_name, "npy": npy_path, "disc_mask": disc_mask, "cup_mask": cup_mask})

print(f"Held-out test set: {len(test_samples)} images", flush=True)
assert len(test_samples) > 50, "Too few test images found -- check paths before continuing."


# ---------- model definitions (identical to the training scripts, needed to load the weights) ----------
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PlainUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


class AttentionGate(nn.Module):
    def __init__(self, fg, fx, fi):
        super().__init__()
        self.W_g = nn.Conv2d(fg, fi, 1)
        self.W_x = nn.Conv2d(fx, fi, 1)
        self.psi = nn.Conv2d(fi, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, g, x):
        return x * self.sig(self.psi(self.relu(self.W_g(g) + self.W_x(x))))


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, e4)], 1))
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], 1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], 1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], 1))
        return self.final(d1)


class ResBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(o)
        self.conv2 = nn.Conv2d(o, o, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x):
        idn = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idn)


class DeepResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        ch = [64, 128, 256, 512, 768, 1024]
        self.enc1 = ResBlock(in_ch, ch[0])
        self.enc2 = ResBlock(ch[0], ch[1])
        self.enc3 = ResBlock(ch[1], ch[2])
        self.enc4 = ResBlock(ch[2], ch[3])
        self.enc5 = ResBlock(ch[3], ch[4])
        self.bottleneck = ResBlock(ch[4], ch[5])
        self.pool = nn.MaxPool2d(2)
        self.up5 = nn.ConvTranspose2d(ch[5], ch[4], 2, 2)
        self.dec5 = ResBlock(ch[4] * 2, ch[4])
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], 2, 2)
        self.dec4 = ResBlock(ch[3] * 2, ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.dec3 = ResBlock(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.dec2 = ResBlock(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.dec1 = ResBlock(ch[0] * 2, ch[0])
        self.final = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        d5 = self.dec5(torch.cat([self.up5(b), e5], 1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


POLAR_CENTER = (256, 256)
POLAR_MAX_RADIUS = 256
POLAR_SIZE = (512, 512)


def to_polar(img2d):
    return cv2.warpPolar(img2d, POLAR_SIZE, POLAR_CENTER, POLAR_MAX_RADIUS, cv2.INTER_LINEAR)


_CORNER = np.hypot(*np.meshgrid(np.arange(512) - 256, np.arange(512) - 256)) > 256


def to_cartesian(img2d):
    src = np.ascontiguousarray(img2d, dtype=np.float32)
    out = cv2.warpPolar(
        src,
        POLAR_SIZE,
        POLAR_CENTER,
        POLAR_MAX_RADIUS,
        cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        np.zeros(src.shape, np.float32),
    )
    out[_CORNER] = 0
    return out


class HarmonicRegressor(nn.Module):
    def __init__(self, out_dim=50):
        super().__init__()
        self.features = nn.Sequential(*list(tvmodels.resnet34(weights=None).children())[:-1])
        self.head = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def build_harmonic_basis(M, N=360):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cols = [np.ones(N) / np.sqrt(N)]
    for m in range(1, M + 1):
        cols.append(np.sqrt(2 / N) * np.cos(m * theta))
        cols.append(np.sqrt(2 / N) * np.sin(m * theta))
    return np.stack(cols, axis=1)


def coeffs_to_mask(u, Phi, center=(256, 256), img_size=512):
    radii = Phi @ u
    theta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    x = center[0] + radii * np.cos(theta)
    y = center[1] + radii * np.sin(theta)
    pts = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ---------- run every model on every test image, ensembling the 5 folds for each baseline ----------
def eval_ensemble(model_class, out_dir, is_polar=False, **kwargs):
    models = []
    for k in range(1, 6):
        ckpt_path = os.path.join(os.path.expanduser(out_dir), f"fold{k}_best.pt")
        m = model_class(**kwargs).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)

    per_image = []
    with torch.no_grad():
        for s in test_samples:
            img = np.load(s["npy"]).astype(np.float32)
            if is_polar:
                img_in = np.stack([to_polar(img[:, :, c]) for c in range(3)], axis=-1)
            else:
                img_in = img
            img_t = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            probs_sum = None
            for m in models:
                p = torch.sigmoid(m(img_t)).cpu().numpy()[0]
                probs_sum = p if probs_sum is None else probs_sum + p
            probs_avg = probs_sum / len(models)

            if is_polar:
                disc_pred = to_cartesian(probs_avg[0]) > 0.5
                cup_pred = to_cartesian(probs_avg[1]) > 0.5
            else:
                disc_pred = probs_avg[0] > 0.5
                cup_pred = probs_avg[1] > 0.5

            true_disc = s["disc_mask"] > 0
            true_cup = s["cup_mask"] > 0 if s["cup_mask"] is not None else np.zeros_like(true_disc)
            per_image.append({"case": s["case"], "disc": dice(disc_pred, true_disc), "cup": dice(cup_pred, true_cup)})
    return per_image


results = {}
print("Evaluating baseline 1 (plain U-Net)...", flush=True)
results["baseline1_unet"] = eval_ensemble(PlainUNet, "~/POLARIS/BTP-Datasets/outputs/phase8/phase8_baseline_unet")
print("Evaluating baseline 2 (attention U-Net)...", flush=True)
results["baseline2_attention"] = eval_ensemble(
    AttentionUNet, "~/POLARIS/BTP-Datasets/outputs/phase8/phase8_attention_unet"
)
print("Evaluating baseline 3 (deep ResUNet)...", flush=True)
results["baseline3_deepres"] = eval_ensemble(DeepResUNet, "~/POLARIS/BTP-Datasets/outputs/phase8/phase8_deep_resunet")
print("Evaluating baseline 4 (polar U-Net)...", flush=True)
results["baseline4_polar"] = eval_ensemble(
    PlainUNet, "~/POLARIS/BTP-Datasets/outputs/phase8/phase8_polar_unet_v2", is_polar=True
)

# ---------- our own Phase 2 harmonic model ----------
print("Evaluating our own Phase 2 harmonic model...", flush=True)
ckpt2 = torch.load(
    os.path.join(
        os.path.expanduser(os.environ.get("POLARIS_ROOT", "~/POLARIS/BTP-Datasets")), "phase2_checkpoints_v2/best.pt"
    ),
    map_location=DEVICE,
    weights_only=False,
)
mean_model = HarmonicRegressor().to(DEVICE)
mean_model.load_state_dict(ckpt2["model_state"])
mean_model.eval()
target_mean, target_std = ckpt2["target_mean"], ckpt2["target_std"]
Phi = build_harmonic_basis(12, N=360)

harmonic_results = []
with torch.no_grad():
    for s in test_samples:
        img = np.load(s["npy"]).astype(np.float32)
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_t = torch.nn.functional.interpolate(img_t, size=256, mode="bilinear", align_corners=False).to(DEVICE)
        pred_norm = mean_model(img_t).cpu().numpy()[0]
        pred_raw = pred_norm * target_std + target_mean
        disc_pred = coeffs_to_mask(pred_raw[:25], Phi) > 0
        cup_pred = coeffs_to_mask(pred_raw[25:], Phi) > 0
        true_disc = s["disc_mask"] > 0
        true_cup = s["cup_mask"] > 0 if s["cup_mask"] is not None else np.zeros_like(true_disc)
        harmonic_results.append(
            {"case": s["case"], "disc": dice(disc_pred, true_disc), "cup": dice(cup_pred, true_cup)}
        )
results["ours_harmonic"] = harmonic_results

# ---------- summary ----------
print("\n=== Test-set results (129 held-out images, fair comparison) ===", flush=True)
for name, res in results.items():
    disc_scores = [r["disc"] for r in res]
    cup_scores = [r["cup"] for r in res]
    print(
        f"{name}: disc={np.mean(disc_scores)*100:.2f}% (+/-{np.std(disc_scores)*100:.2f}), "
        f"cup={np.mean(cup_scores)*100:.2f}% (+/-{np.std(cup_scores)*100:.2f})",
        flush=True,
    )

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved per-image results to {OUT_PATH}", flush=True)
