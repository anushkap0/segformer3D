import os, json, shutil, numpy as np, torch
from multiprocessing import Pool
S = "/workspace/sf3d_ref/data/brats2017_seg/BraTS2017_Training_Data"
D = "data/processed/BraTS2017_repro"
def load(p):
    x = torch.load(p, map_location="cpu", weights_only=False)
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
def conv(c):
    o = f"{D}/{c}"
    if os.path.exists(f"{o}/label.npy"): return c, "skip"
    m = load(f"{S}/{c}/{c}_modalities.pt"); l = load(f"{S}/{c}/{c}_label.pt")
    assert m.shape == (4, 128, 128, 128) and l.shape == (3, 128, 128, 128), (c, m.shape, l.shape)
    tc, wt, et = l[0] > 0, l[1] > 0, l[2] > 0                     # authors' channel order: TC, WT, ET
    y = np.zeros(l.shape[1:], np.uint8); y[wt] = 2; y[tc] = 1; y[et] = 3
    os.makedirs(o, exist_ok=True)
    np.save(f"{o}/image.npy", np.ascontiguousarray(m[[1, 2, 3, 0]], dtype=np.float32))   # [FLAIR,T1,T1ce,T2] -> [T1,T1ce,T2,FLAIR]
    json.dump({"spacing": [1.0, 1.0, 1.0]}, open(f"{o}/spacing.json", "w"))
    np.save(f"{o}/label.npy", y)                                   # saved last = completion marker
    shutil.rmtree(f"{S}/{c}")
    return c, "ok"
if __name__ == "__main__":
    ids = sorted(os.listdir(S)); os.makedirs(D, exist_ok=True)
    with Pool(8) as p: res = p.map(conv, ids, chunksize=2)
    shutil.copy("data/processed/BraTS2017ext/split_manifest_authors.json", f"{D}/split_manifest.json")
    print(len(res), "cases;", sum(r[1] == "ok" for r in res), "converted")
