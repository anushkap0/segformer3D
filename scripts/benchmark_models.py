import json, os, sys, time, yaml, torch
sys.path.insert(0, "/workspace/segformer3D")
from torch.utils.flop_counter import FlopCounterMode
from src.train import build_model

names = [a for a in sys.argv[1:] if not a.startswith("--")]
do_time = "--time" in sys.argv
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__)

def plain_attention():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

path = "logs/efficiency.json"
res = json.load(open(path)) if os.path.exists(path) else {}
for name in names:
    cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
    cfg["model"].pop("init_from_phase1_checkpoint", None)
    m = build_model(cfg).to(dev).eval()
    x = torch.randn(1, 4, 128, 128, 128, device=dev)
    rec = res.get(name, {})
    rec["params_M"] = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad(), plain_attention(), FlopCounterMode(display=False) as fc:
        m(x)
    rec["gmacs"] = fc.get_total_flops() / 2 / 1e9
    if do_time and dev == "cuda":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(5):
                m(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(30):
                t0 = time.perf_counter(); m(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ts.sort()
        rec["ms_median"] = ts[len(ts) // 2] * 1000
        rec["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    res[name] = rec
    print(name, {k: round(v, 2) for k, v in rec.items()})
os.makedirs("logs", exist_ok=True)
json.dump(res, open(path, "w"), indent=2)
