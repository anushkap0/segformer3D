import json, random
base = "data/processed/BraTS2021_FULL/"
m = json.load(open(base + "split_manifest.json"))
pool = sorted(m["train"] + m["val"])
random.Random(42).shuffle(pool)
k = 5
folds = [pool[i::k] for i in range(k)]
for i in range(k):
    val = sorted(folds[i])
    train = sorted(c for j in range(k) if j != i for c in folds[j])
    json.dump({"train": train, "val": val, "test": m["test"]}, open(base + f"fold_{i}_manifest.json", "w"), indent=2)
    print(i, len(train), len(val), len(m["test"]))
