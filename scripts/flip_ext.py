import os, json, shutil, numpy as np
from multiprocessing import Pool
S, D = "data/processed/BraTS2017ext", "data/processed/BraTS2017ext_flip"
def f(c):
    o = f"{D}/{c}"
    if os.path.exists(o + "/label.npy"): return c
    os.makedirs(o, exist_ok=True)
    x = np.load(f"{S}/{c}/image.npy"); y = np.load(f"{S}/{c}/label.npy")
    np.save(o + "/image.npy", np.ascontiguousarray(x[:, ::-1, ::-1, :]))
    for n in os.listdir(f"{S}/{c}"):
        if n.endswith(".json"): shutil.copy(f"{S}/{c}/{n}", o)
    np.save(o + "/label.npy", np.ascontiguousarray(y[::-1, ::-1, :]))   # saved last = completion marker
    return c
if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    shutil.copy(f"{S}/split_manifest.json", D)
    ids = json.load(open(f"{S}/split_manifest.json"))["test"]
    with Pool(8) as p: done = p.map(f, ids, chunksize=4)
    print("flipped", len(done))
