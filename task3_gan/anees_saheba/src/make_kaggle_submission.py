"""
Build the Kaggle submission for Task 3.

The competition does not take images. It takes a one row submission.csv with
columns ID, FID and MiFID, and the leaderboard score is the negative mean of
those two numbers, so a score of -54.699 means FID and MiFID averaged 54.699.
Lower FID is better, so a less negative score is better.

The scoring logic below is the instructor's, unchanged. FID is the Frechet
distance between Inception v3 activations. MiFID here is the mean cosine
distance between paired activations, which is not the usual definition but is
what the scorer uses, so it is what gets submitted.

What this script does:
  1. loads the saved generators
  2. translates all 300 Monet paintings to photos      -> pred_A2B
  3. translates held out photographs to Monet style    -> pred_B2A
  4. scores both directions against the real images
  5. writes submission.csv

The training run only translated 30 Monet paintings, which is too few for a
stable FID, so step 2 regenerates all 300.

Usage from the repository root:

    python task3_gan/anees_saheba/src/make_kaggle_submission.py

Environment overrides:
    DATA266_MONET_DIR, DATA266_PHOTO_DIR, DATA266_CHECKPOINT, DATA266_OUT
    DATA266_REUSE_B2A   folder of already generated photo to Monet images
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

IMAGE_SIZE = 256
N_EVAL = 300           # the scorer evaluates on 300 images per direction
BATCH_SIZE = 32


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Must match the training definition exactly or the weights will not load."""

    def __init__(self, input_channels=3, output_channels=3, residual_blocks=6,
                 base_filters=80):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_channels, base_filters, 7),
                  nn.InstanceNorm2d(base_filters), nn.ReLU(inplace=True)]
        channels = base_filters
        for _ in range(2):
            nxt = channels * 2
            layers += [nn.Conv2d(channels, nxt, 3, 2, 1), nn.InstanceNorm2d(nxt),
                       nn.ReLU(inplace=True)]
            channels = nxt
        for _ in range(residual_blocks):
            layers.append(ResidualBlock(channels))
        for _ in range(2):
            nxt = channels // 2
            layers += [nn.ConvTranspose2d(channels, nxt, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(nxt), nn.ReLU(inplace=True)]
            channels = nxt
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, output_channels, 7),
                   nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_dir(env_var, name):
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    for root in [Path.cwd(), Path.cwd().parent, Path("/app"), Path("/app/dataset")]:
        for candidate in root.rglob(name):
            if candidate.is_dir():
                return candidate.resolve()
    raise SystemExit(f"Could not find {name}. Set {env_var}.")


def find_checkpoint():
    override = os.environ.get("DATA266_CHECKPOINT")
    if override:
        return Path(override).expanduser().resolve()
    # My own folder first. A loose search finds my teammate's checkpoint too,
    # and hers has a different generator shape, so it loads with a size
    # mismatch rather than failing in an obvious way.
    preferred = Path.cwd() / "task3_gan" / "anees_saheba" / "checkpoints" / "generators_epoch_050.pt"
    if preferred.is_file():
        return preferred.resolve()
    for root in [Path.cwd(), Path("/app")]:
        for candidate in sorted(root.rglob("generators_epoch_050.pt")):
            if "yashsharee" in str(candidate) or "yashashree" in str(candidate).lower():
                continue
            return candidate.resolve()
    raise SystemExit("Could not find generators_epoch_050.pt. Set DATA266_CHECKPOINT.")


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


TO_TENSOR = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@torch.no_grad()
def translate(generator, paths, out_dir, device, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for i in range(0, len(paths), 8):
        chunk = paths[i:i + 8]
        batch = torch.stack([TO_TENSOR(Image.open(p).convert("RGB")) for p in chunk])
        fake = generator(batch.to(device))
        fake = (fake.clamp(-1, 1) + 1) / 2
        for path, tensor in zip(chunk, fake.cpu()):
            arr = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            Image.fromarray(arr).save(out_dir / f"{path.stem}.png")
        if i % 80 == 0:
            print(f"  {label}: {min(i + 8, len(paths))}/{len(paths)}", flush=True)
    print(f"  {label}: done in {time.time() - start:.0f} s")
    return list_images(out_dir)


# The scoring below is the instructor's logic, unchanged.

INCEPTION_TF = T.Compose([
    T.Resize(299), T.CenterCrop(299), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get_inception(device):
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                              transform_input=False)
    net.fc = nn.Identity()
    return net.to(device).eval()


@torch.no_grad()
def activations(model, paths, device, batch_size=BATCH_SIZE):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch_size]])
        feats.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_and_mifid(model, real_paths, gen_paths, device):
    real_paths, gen_paths = sorted(real_paths), sorted(gen_paths)
    n = min(len(real_paths), len(gen_paths))
    real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    real = activations(model, real_paths, device)
    gen = activations(model, gen_paths, device)
    fid = frechet_distance(real.mean(axis=0), np.cov(real, rowvar=False),
                           gen.mean(axis=0), np.cov(gen, rowvar=False))
    m = min(len(real), len(gen))
    mifid = float(np.mean([cosine(real[i], gen[i]) for i in range(m)]))
    return fid, mifid, n


def main():
    device = pick_device()
    monet_dir = find_dir("DATA266_MONET_DIR", "monet_jpg")
    photo_dir = find_dir("DATA266_PHOTO_DIR", "photo_jpg")
    checkpoint_path = find_checkpoint()
    out_root = Path(os.environ.get("DATA266_OUT", "kaggle_submission")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"device     : {device}")
    print(f"monet      : {monet_dir}")
    print(f"photo      : {photo_dir}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"output     : {out_root}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"checkpoint epoch {checkpoint['epoch']}, seed {checkpoint['seed']}")

    g_a2b = Generator().to(device); g_a2b.load_state_dict(checkpoint["G_A2B"]); g_a2b.eval()
    g_b2a = Generator().to(device); g_b2a.load_state_dict(checkpoint["G_B2A"]); g_b2a.eval()

    monet_images = list_images(monet_dir)
    photo_images = list_images(photo_dir)
    print(f"real monet {len(monet_images)}, real photo {len(photo_images)}")

    # Monet to photo, all 300 paintings.
    pred_a2b = translate(g_a2b, monet_images, out_root / "pred_A2B", device, "A2B")

    # Photo to Monet. Reuse an existing folder if one was given, since the run
    # already produced these and regenerating changes nothing.
    reuse = os.environ.get("DATA266_REUSE_B2A")
    if reuse and Path(reuse).is_dir():
        pred_b2a = list_images(reuse)
        print(f"  B2A: reusing {len(pred_b2a)} images from {reuse}")
    else:
        pred_b2a = translate(g_b2a, photo_images[:N_EVAL * 3],
                             out_root / "pred_B2A", device, "B2A")

    print("\nscoring, this loads Inception v3 and runs it over four sets")
    inception = get_inception(device)

    fid_b2a, mifid_b2a, n_b2a = fid_and_mifid(
        inception, monet_images[:N_EVAL], pred_b2a[:N_EVAL], device)
    print(f"  photo to Monet : FID {fid_b2a:.4f}  MiFID {mifid_b2a:.4f}  on {n_b2a} pairs")

    fid_a2b, mifid_a2b, n_a2b = fid_and_mifid(
        inception, photo_images[:N_EVAL], pred_a2b[:N_EVAL], device)
    print(f"  Monet to photo : FID {fid_a2b:.4f}  MiFID {mifid_a2b:.4f}  on {n_a2b} pairs")

    fid = (fid_a2b + fid_b2a) / 2
    mifid = (mifid_a2b + mifid_b2a) / 2
    submission = out_root / "submission.csv"
    submission.write_text(f"ID,FID,MiFID\n1,{fid},{mifid}\n")

    print(f"\nFID   {fid:.4f}")
    print(f"MiFID {mifid:.4f}")
    print(f"expected leaderboard score {-(fid + mifid) / 2:.4f}")
    print(f"\nwrote {submission}. Upload that file.")


if __name__ == "__main__":
    main()
