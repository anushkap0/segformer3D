import sys, yaml, torch
sys.path.insert(0, "/workspace/segformer3D")
from torch.utils.flop_counter import FlopCounterMode
from src.train import build_model
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__)
for name in sys.argv[1:]:
    cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
    cfg["model"].pop("init_from_phase1_checkpoint", None)
    m = build_model(cfg).to(dev).eval()
    x = torch.randn(1, 4, 128, 128, 128, device=dev)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    g = fc.get_total_flops() / 2 / 1e9
    p = sum(q.numel() for q in m.parameters())
    print(f"{name:24s} params {p/1e6:6.2f}M   GMACs {g:8.1f}")
