import glob, json, os
import numpy as np
from multiprocessing import Pool

def fp(d):
    try:
        im = np.load(d + "/image.npy", mmap_mode="r")
        lab = np.load(d + "/label.npy")
        c0 = int((np.asarray(im[0]) != 0).sum())
        c3 = int((np.asarray(im[3]) != 0).sum())
        shape = tuple(sorted(int(x) for x in im.shape[1:]))
        labc = (int((lab > 0).sum()), int(np.isin(lab, [1, 3]).sum()), int((lab == 3).sum()))
        return os.path.basename(d), (shape, c0, c3), labc
    except Exception:
        return os.path.basename(d), None, None

if __name__ == "__main__":
    ours = sorted(glob.glob("data/processed/BraTS2021_FULL/BraTS2021_*"))
    ext = sorted(glob.glob("data/processed/BraTS2017ext/BRATS_*"))
    with Pool(16) as p:
        A = p.map(fp, ours, chunksize=8)
        B = p.map(fp, ext, chunksize=8)
    manifest = json.load(open("data/processed/BraTS2021_FULL/split_manifest.json"))
    split_of = {c: s for s, cs in manifest.items() for c in cs}
    index = {}
    for c, f, l in A:
        if f:
            index.setdefault(f, []).append((c, l))
    out = {}
    for c, f, l in B:
        m = index.get(f, []) if f else []
        out[c] = {"matches": [x[0] for x in m],
                  "splits": sorted({split_of[x[0]] for x in m}),
                  "label_equal": [x[1] == l for x in m]}
    os.makedirs("logs/ext", exist_ok=True)
    json.dump(out, open("logs/ext/overlap.json", "w"), indent=1)
    n = len(out)
    anym = sum(1 for v in out.values() if v["matches"])
    tr = sum(1 for v in out.values() if "train" in v["splits"])
    oth = sum(1 for v in out.values() if v["matches"] and "train" not in v["splits"])
    print(f"ext cases: {n} | image match in BraTS2021: {anym} | matched to a TRAIN case: {tr} | matched only to val/test: {oth} | no match: {n - anym}")
    print("fingerprints read ok: ours", sum(1 for _, f, _ in A if f), "of", len(A), "| ext", sum(1 for _, f, _ in B if f), "of", len(B))
