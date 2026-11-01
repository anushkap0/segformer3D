"""Convert an imageTr/labelsTr dataset to this project's .npy format (our channel order t1,t1ce,t2,flair).
--order: source file index for each of our channels. --label_scheme: how raw labels map to our key."""
import argparse, glob, json, multiprocessing as mp, os
import nibabel as nib
import numpy as np

LABEL_MAPS = {"brats": {1: 1, 2: 2, 4: 3}, "msd": {1: 2, 2: 1, 3: 3}, "same": {1: 1, 2: 2, 3: 3}}

def zscore(v):
    m = v > 0
    if m.sum() == 0:
        return v.astype(np.float32)
    mean, std = v[m].mean(), v[m].std()
    out = (v - mean) / (std if std > 1e-6 else 1.0)
    out[~m] = 0.0
    return out.astype(np.float32)

def work(t):
    case, raw, out, order, lmap = t
    try:
        vols, zooms = [], None
        for idx in order:
            nii = nib.load(f"{raw}/imageTr/{case}_{idx:04d}.nii.gz")
            if zooms is None:
                zooms = [float(z) for z in nii.header.get_zooms()[:3]]
            vols.append(zscore(nii.get_fdata().astype(np.float32)))
        lab_raw = nib.load(f"{raw}/labelsTr/{case}.nii.gz").get_fdata().astype(np.int64)
        unknown = set(np.unique(lab_raw).tolist()) - set(lmap) - {0}
        if unknown:
            return case, f"unexpected labels {unknown}", zooms
        lab = np.zeros(lab_raw.shape, np.uint8)
        for k, v in lmap.items():
            lab[lab_raw == k] = v
        img = np.stack(vols, 0)
        c = np.argwhere((img != 0).any(0)); lo, hi = c.min(0), c.max(0) + 1
        img = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        lab = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = os.path.join(out, case)
        os.makedirs(d, exist_ok=True)
        np.save(d + "/image.npy", img.astype(np.float32)); np.save(d + "/label.npy", lab)
        json.dump({"spacing": zooms, "orig_spacing": zooms}, open(d + "/spacing.json", "w"))
        return case, None, zooms
    except Exception as e:
        return case, f"{type(e).__name__}: {e}", None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--order", default="1,2,3,0", help="source index for t1,t1ce,t2,flair (README order assumed FLAIR,T1,T1gd,T2)")
    ap.add_argument("--label_scheme", required=True, choices=list(LABEL_MAPS))
    a = ap.parse_args()
    order = [int(x) for x in a.order.split(",")]
    cases = sorted(os.path.basename(f)[:-len("_0000.nii.gz")] for f in glob.glob(f"{a.raw_dir}/imageTr/*_0000.nii.gz"))
    print("cases found:", len(cases), "| order:", order, "| scheme:", a.label_scheme)
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(c, a.raw_dir, a.out_dir, order, LABEL_MAPS[a.label_scheme]) for c in cases]
    bad, odd = [], 0
    with mp.Pool(16) as pool:
        for i, (c, err, z) in enumerate(pool.imap_unordered(work, jobs)):
            if err: bad.append((c, err))
            elif z and any(abs(x - 1.0) > 0.05 for x in z): odd += 1
            if i % 50 == 0: print(i, c, flush=True)
    done = sorted(d for d in os.listdir(a.out_dir) if os.path.exists(f"{a.out_dir}/{d}/image.npy"))
    json.dump({"train": [], "val": [], "test": done}, open(a.out_dir + "/split_manifest.json", "w"), indent=2)
    print(f"done: {len(done)} cases | failed: {len(bad)} {bad[:3]} | non-1mm spacing: {odd}")
