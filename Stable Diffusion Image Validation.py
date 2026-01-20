import argparse, os
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
from torchvision import models, transforms

from tqdm import tqdm
from scipy import linalg
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import imagehash

# ----------------------------
# Config: your paths (Windows)
# ----------------------------
DEFAULT_REAL = r"C:\Users\user\Mitigating Imbalance with SMOTE and Stable Diffusion in Solar Panel Dataset using CNN Models\data\dirty"
DEFAULT_SYN  = r"C:\Users\user\Mitigating Imbalance with SMOTE and Stable Diffusion in Solar Panel Dataset using CNN Models\generated_cropped"

# ----------------------------
# Helpers
# ----------------------------
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def list_images(root):
    files = [p for p in glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(IMG_EXTS)]
    if not files:
        raise ValueError(f"No images found under: {root}")
    return files

def pil_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------
# InceptionV3 pool3 features (2048-D)
# ----------------------------
class InceptionPool3(nn.Module):
    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3, nn.MaxPool2d(3, 2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

def get_feats_mu_sigma(paths, model, device, batch=32, resize=299):
    tfm = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    feats = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="Featurizing", leave=False):
            ims = [tfm(pil_rgb(p)) for p in paths[i:i+batch]]
            x = torch.stack(ims).to(device, non_blocking=True)
            f = model(x).cpu().numpy().astype(np.float64)
            feats.append(f)
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return feats, mu, sigma

# ----------------------------
# FID & KID
# ----------------------------
def fid(mu1, sig1, mu2, sig2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + off).dot(sig2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2*np.trace(covmean))

def polynomial_mmd_ub(x, y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    k_xx = ((x @ x.T) * gamma + coef0) ** degree
    k_yy = ((y @ y.T) * gamma + coef0) ** degree
    k_xy = ((x @ y.T) * gamma + coef0) ** degree
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    n, m = x.shape[0], y.shape[0]
    mmd = k_xx.sum()/(n*(n-1)) + k_yy.sum()/(m*(m-1)) - 2.0*k_xy.mean()
    return float(mmd)

def compute_kid(x, y, subsets=20, subset_size=100, seed=123):
    rng = np.random.default_rng(seed)
    n, m = x.shape[0], y.shape[0]
    ss = int(min(subset_size, n, m))
    vals = []
    for _ in range(subsets):
        ix = rng.choice(n, ss, replace=False)
        iy = rng.choice(m, ss, replace=False)
        vals.append(polynomial_mmd_ub(x[ix], y[iy], degree=3, coef0=1.0))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())

# ----------------------------
# Image stats + pHash
# ----------------------------
def image_stats(paths):
    means, stds, hsv_h, hsv_s, hsv_v = [], [], [], [], []
    for p in paths:
        im = pil_rgb(p)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        means.append(arr.mean()); stds.append(arr.std())
        hsv = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
        hsv = np.asarray(hsv, dtype=np.float32)/255.0
        hsv_h.append(hsv[...,0].mean()); hsv_s.append(hsv[...,1].mean()); hsv_v.append(hsv[...,2].mean())
    return (np.array(means), np.array(stds), np.array(hsv_h), np.array(hsv_s), np.array(hsv_v))

def phash_signatures(paths):
    sigs = []
    for p in paths:
        im = pil_rgb(p)
        sigs.append(imagehash.phash(im))
    return sigs

def phash_overlap_rate(real_sigs, syn_sigs, threshold=5):
    total, hits = 0, 0
    for s in syn_sigs:
        dists = [s - r for r in real_sigs]  # Hamming distance
        if dists:
            total += 1
            if min(dists) <= threshold:
                hits += 1
    return hits/total if total else 0.0

# ----------------------------
# Plots
# ----------------------------
def save_hist(a, b, title, xlabel, out):
    plt.figure()
    plt.hist(a, bins=30, alpha=0.6, label="Real", density=True)
    plt.hist(b, bins=30, alpha=0.6, label="Synthetic", density=True)
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

def save_tsne(real_feats, syn_feats, out_png):
    # small subsample for readability
    max_points = 3000
    r_ix = np.random.choice(real_feats.shape[0], min(max_points//2, real_feats.shape[0]), replace=False)
    s_ix = np.random.choice(syn_feats.shape[0], min(max_points//2, syn_feats.shape[0]), replace=False)
    X = np.vstack([real_feats[r_ix], syn_feats[s_ix]])
    y = np.array([0]*len(r_ix) + [1]*len(s_ix))

    tsne = TSNE(n_components=2, init="random", learning_rate="auto",
                perplexity=min(30, max(5, len(X)//50)), random_state=42)
    Z = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(Z[y==0,0], Z[y==0,1], s=10, label="Real", alpha=0.7)
    plt.scatter(Z[y==1,0], Z[y==1,1], s=10, label="Synthetic", alpha=0.7)
    plt.title("t-SNE of Inception Features (Real vs Synthetic)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Validate SD images: FID, KID, pHash, t-SNE, histograms")
    ap.add_argument("--real", default=DEFAULT_REAL, help="Folder with REAL images")
    ap.add_argument("--synthetic", default=DEFAULT_SYN, help="Folder with SYNTHETIC images")
    ap.add_argument("--outdir", default="sd_validation_outputs", help="Where to save plots & report")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resize", type=int, default=299)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kid_subsets", type=int, default=20)
    ap.add_argument("--kid_subset_size", type=int, default=100)
    ap.add_argument("--phash_threshold", type=int, default=5)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    real_paths = list_images(args.real)
    syn_paths  = list_images(args.synthetic)
    print(f"Found {len(real_paths)} real, {len(syn_paths)} synthetic images.")

    device = torch.device(args.device)
    model = InceptionPool3().to(device)

    # Features
    real_feats, mu_r, sig_r = get_feats_mu_sigma(real_paths, model, device, batch=args.batch, resize=args.resize)
    syn_feats,  mu_s, sig_s = get_feats_mu_sigma(syn_paths,  model, device, batch=args.batch, resize=args.resize)

    # FID & KID
    fid_val = fid(mu_r, sig_r, mu_s, sig_s)
    kid_mean, kid_std = compute_kid(real_feats, syn_feats, subsets=args.kid_subsets, subset_size=args.kid_subset_size)

    # pHash near-duplicate rate
    real_hashes = phash_signatures(real_paths)
    syn_hashes  = phash_signatures(syn_paths)
    overlap = phash_overlap_rate(real_hashes, syn_hashes, threshold=args.phash_threshold)

    # t-SNE
    save_tsne(real_feats, syn_feats, os.path.join(outdir, "tsne_real_vs_synthetic.png"))

    # Histograms
    def image_stats(paths):
        means, stds, hsv_h, hsv_s, hsv_v = [], [], [], [], []
        for p in paths:
            im = pil_rgb(p)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            means.append(arr.mean()); stds.append(arr.std())
            hsv = Image.fromarray((arr*255).astype(np.uint8)).convert("HSV")
            hsv = np.asarray(hsv, dtype=np.float32)/255.0
            hsv_h.append(hsv[...,0].mean()); hsv_s.append(hsv[...,1].mean()); hsv_v.append(hsv[...,2].mean())
        return (np.array(means), np.array(stds), np.array(hsv_h), np.array(hsv_s), np.array(hsv_v))

    r_mean, r_std, r_h, r_s, r_v = image_stats(real_paths)
    s_mean, s_std, s_h, s_s, s_v = image_stats(syn_paths)

    save_hist(r_mean, s_mean, "Mean Intensity Distribution", "Mean Intensity (0–1)",
              os.path.join(outdir, "hist_mean_intensity.png"))
    save_hist(r_std, s_std, "Intensity Std. Dev. Distribution", "Intensity Std (0–1)",
              os.path.join(outdir, "hist_intensity_std.png"))
    save_hist(r_h, s_h, "Hue Distribution", "Hue (0–1)",
              os.path.join(outdir, "hist_hue.png"))
    save_hist(r_s, s_s, "Saturation Distribution", "Saturation (0–1)",
              os.path.join(outdir, "hist_saturation.png"))
    save_hist(r_v, s_v, "Value (Brightness) Distribution", "Value (0–1)",
              os.path.join(outdir, "hist_value.png"))

    # Report
    report = os.path.join(outdir, "report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("Stable Diffusion Validation Report\n")
        f.write("---------------------------------\n")
        f.write(f"Real images: {len(real_paths)}\n")
        f.write(f"Synthetic images: {len(syn_paths)}\n\n")
        f.write(f"FID: {fid_val:.3f} (lower is better)\n")
        f.write(f"KID: {kid_mean*1000:.3f} x1e-3  ± {kid_std*1000:.3f}\n")
        f.write(f"pHash near-duplicate rate (<= {args.phash_threshold} distance): {overlap*100:.2f}% of synthetic\n")
        f.write("\nSee saved figures in this folder for t-SNE and histogram comparisons.\n")

    print("\n=== Results ===")
    print(f"FID: {fid_val:.3f}  (lower is better)")
    print(f"KID: {kid_mean*1000:.3f} x1e-3  ± {kid_std*1000:.3f}  (lower is better)")
    print(f"pHash near-duplicate rate: {overlap*100:.2f}% of synthetic")
    print(f"Saved plots & report to: {outdir}")

if __name__ == "__main__":
    main()
