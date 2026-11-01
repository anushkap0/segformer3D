import json, sys, numpy as np
def load(arg):
    runs=[]
    for p in arg.split(","):
        d=json.load(open(p))["per_case"]
        runs.append({c: np.mean([v[k]["dice"] for k in ("WT","TC","ET")]) for c,v in d.items()})
    ids=set.intersection(*[set(r) for r in runs])
    return {c: np.mean([r[c] for r in runs]) for c in ids}
a,b=load(sys.argv[1]),load(sys.argv[2])
ids=sorted(set(a)&set(b)); diff=np.array([a[i]-b[i] for i in ids])
rng=np.random.default_rng(0)
boot=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(10000)])
print(f"n={len(ids)}  A-B = {diff.mean():+.4f}  95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]")
