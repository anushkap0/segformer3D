import argparse, json, os, random
from glob import glob
import nibabel as nib, numpy as np
from scipy.ndimage import zoom

ORDER = ["t1", "t1ce", "t2", "flair"]            # this project's channel order

def mod_name(s):
    s = s.lower().replace("-", "").replace("_", "")
    if "flair" in s: return "flair"
    if "gd" in s or "t1ce" in s or "t1c" in s or "contrast" in s: return "t1ce"
    if "t2" in s: return "t2"
    if "t1" in s: return "t1"
    raise SystemExit(f"cannot interpret modality name {s!r} in dataset.json")

def zscore(v):
    m = v > 0
    if m.sum() == 0: return v.astype(np.float32)
    sd = v[m].std(); out = (v - v[m].mean()) / (sd if sd > 1e-6 else 1.0); out[~m] = 0
    return out.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.8); ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(a.raw_dir, "dataset.json"))) if os.path.exists(os.path.join(a.raw_dir, "dataset.json")) else {}
    mods = meta.get("modality") or meta.get("channel_names") or {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}
    chan = {mod_name(v): int(k) for k, v in mods.items()}
    assert sorted(chan) == sorted(ORDER), f"unexpected modalities {mods}"
    lmap = {}
    for k, name in (meta.get("labels") or {"1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}).items():
        n = name.lower()
        if int(k) == 0 or "backgr" in n: continue
        lmap[int(k)] = 2 if "edema" in n or "oedema" in n else 1 if ("non" in n or "necro" in n) else 3 if "enhanc" in n else None
    assert sorted(lmap.values()) == [1, 2, 3], f"could not map labels {meta.get('labels')} -> {lmap}"
    print("modality channels:", chan, "| raw->internal labels:", lmap)

    labs = sorted(p for p in glob(os.path.join(a.raw_dir, "labelsTr", "*.nii*")) if not os.path.basename(p).startswith("."))
    if not labs: raise SystemExit(f"no files in {a.raw_dir}/labelsTr")
    ids = []
    for i, lp in enumerate(labs, 1):
        cid = os.path.basename(lp).split(".nii")[0]
        out = os.path.join(a.out_dir, cid)
        if os.path.exists(os.path.join(out, "spacing.json")): ids.append(cid); continue      # resumable
        four_d = glob(os.path.join(a.raw_dir, "imagesTr", cid + ".nii*"))
        if four_d:
            nii = nib.load(four_d[0]); d = nii.get_fdata(dtype=np.float32)
            assert d.ndim == 4 and d.shape[-1] == 4, f"{cid}: {d.shape}"
            vols = {m: d[..., c] for m, c in chan.items()}
        else:
            vols = {}
            for m, c in chan.items():
                f = glob(os.path.join(a.raw_dir, "imagesTr", f"{cid}_{c:04d}.nii*"))
                assert f, f"{cid}: missing channel {c} ({m})"
                nii = nib.load(f[0]); vols[m] = nii.get_fdata(dtype=np.float32)
        sp = [float(s) for s in nii.header.get_zooms()[:3]]
        lab = np.rint(nib.load(lp).get_fdata()).astype(np.uint8)
        assert set(np.unique(lab).tolist()) <= {0, *lmap}, f"{cid}: unexpected label values {np.unique(lab)}"
        assert all(v.shape == lab.shape for v in vols.values()), f"{cid}: shape mismatch"
        vl = [zscore(vols[m]) for m in ORDER]
        if any(abs(s - 1.0) > 1e-3 for s in sp):
            vl = [zoom(v, sp, order=1) for v in vl]; lab = zoom(lab, sp, order=0)
        img = np.stack(vl); y = np.zeros_like(lab)
        for raw, internal in lmap.items(): y[lab == raw] = internal
        nz = np.argwhere((img != 0).any(0)); lo, hi = nz.min(0), nz.max(0) + 1
        img = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; y = y[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "image.npy"), img.astype(np.float32)); np.save(os.path.join(out, "label.npy"), y)
        json.dump({"spacing": [1.0, 1.0, 1.0], "orig_spacing": sp}, open(os.path.join(out, "spacing.json"), "w"))
        ids.append(cid); print(f"[{i}/{len(labs)}] {cid}", flush=True)
    random.Random(a.seed).shuffle(ids); n = len(ids); ntr, nva = int(n * a.train_frac), int(n * a.val_frac)
    m = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
    json.dump(m, open(os.path.join(a.out_dir, "split_manifest.json"), "w"), indent=2)
    print("DONE", {k: len(v) for k, v in m.items()})
main()
