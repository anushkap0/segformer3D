"""Which source-file order matches our channels (t1,t1ce,t2,flair)? Run the trained model on a few cases
under all 24 permutations; the correct order gives clearly the highest Dice."""
import argparse, glob, itertools, os, sys, yaml
import nibabel as nib
import numpy as np
import torch
sys.path.insert(0, "/workspace/segformer3D")
from monai.inferers import sliding_window_inference
from src.train import build_model

LABEL_MAPS = {"brats": {1: 1, 2: 2, 4: 3}, "msd": {1: 2, 2: 1, 3: 3}, "same": {1: 1, 2: 2, 3: 3}}
ap = argparse.ArgumentParser()
ap.add_argument("--raw_dir", required=True); ap.add_argument("--label_scheme", required=True, choices=list(LABEL_MAPS))
ap.add_argument("--config", default="configs/v2_phase1_baseline.yaml")
ap.add_argument("--checkpoint", default="checkpoints/v2_phase1/best.pt")
ap.add_argument("--n_cases", type=int, default=8)
a = ap.parse_args()
dev = "cuda"
cfg = yaml.safe_load(open(a.config)); cfg["model"].pop("init_from_phase1_checkpoint", None)
model = build_model(cfg).to(dev).eval()
model.load_state_dict(torch.load(a.checkpoint, map_location=dev)["model_state_dict"])

def zscore(v):
    m = v > 0
    if m.sum() == 0:
        return v.astype(np.float32)
    mean, std = v[m].mean(), v[m].std()
    out = (v - mean) / (std if std > 1e-6 else 1.0)
    out[~m] = 0.0
    return out.astype(np.float32)

def regions(lab_raw):
    lab = np.zeros(lab_raw.shape, np.uint8)
    for k, v in LABEL_MAPS[a.label_scheme].items():
        lab[lab_raw == k] = v
    return np.stack([np.isin(lab, [1, 2, 3]), np.isin(lab, [1, 3]), lab == 3]).astype(bool)

def dice(p, t):
    return (2 * (p & t).sum() + 1e-5) / (p.sum() + t.sum() + 1e-5)

cases = sorted(os.path.basename(f)[:-len("_0000.nii.gz")] for f in glob.glob(f"{a.raw_dir}/imageTr/*_0000.nii.gz"))[: a.n_cases]
perms = list(itertools.permutations(range(4)))
scores = {p: [] for p in perms}
for case in cases:
    vols = np.stack([zscore(nib.load(f"{a.raw_dir}/imageTr/{case}_{i:04d}.nii.gz").get_fdata().astype(np.float32)) for i in range(4)])
    gt = regions(nib.load(f"{a.raw_dir}/labelsTr/{case}.nii.gz").get_fdata().astype(np.int64))
    c = np.argwhere((vols != 0).any(0)); lo, hi = c.min(0), c.max(0) + 1
    vols = vols[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; gt = gt[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    xt = torch.from_numpy(vols).float().to(dev)
    for p in perms:
        x = xt[list(p)].unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda"):
            out = sliding_window_inference(x, (128, 128, 128), 1, model, overlap=0.5, mode="gaussian")
        pred = (torch.sigmoid(out[0].float()) > 0.5).cpu().numpy()
        scores[p].append([dice(pred[r], gt[r]) for r in range(3)])
    print("done", case, flush=True)
ranked = sorted(perms, key=lambda p: -np.mean(scores[p]))
print("\nsource-file index used for our (t1, t1ce, t2, flair)  ->  mean Dice [WT TC ET], overall")
for p in ranked[:5]:
    print(p, np.round(np.mean(scores[p], 0), 3), round(float(np.mean(scores[p])), 3))
print("README-assumed order (1,2,3,0):", np.round(np.mean(scores[(1, 2, 3, 0)], 0), 3))
