import sys, torch, yaml
sys.path.insert(0, "/workspace/segformer3D")
from src.train import build_model
def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {}
for name in ["v2_phase1_baseline", "v2_phase2_boundary"]:
    cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
    cfg["model"].pop("init_from_phase1_checkpoint", None)
    m = build_model(cfg).to(dev)
    res[name] = count(m)
    print(name, "params (M):", round(res[name] / 1e6, 3))
    out = m(torch.randn(1, 4, 128, 128, 128, device=dev))
    if isinstance(out, dict):
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        print(shapes)
        assert shapes["region_logits"] == (1, 3, 128, 128, 128) and shapes["refined_logits"] == (1, 3, 128, 128, 128)
        assert shapes["boundary_logits"] == (1, 1, 128, 128, 128)
        out["refined_logits"].mean().backward()
        assert all(p.grad is not None for p in m.proj.parameters()), "hooks se gradient nahi aa raha"
        print("hook gradients OK")
    else:
        assert tuple(out.shape) == (1, 3, 128, 128, 128)
        print("logits", tuple(out.shape))
assert 4.45e6 < res["v2_phase1_baseline"] < 4.60e6, "baseline param count expected range se bahar"
assert 4.9e6 < res["v2_phase2_boundary"] < 5.7e6, "boundary model param count expected range se bahar"
print("add-on parameters (M):", round((res["v2_phase2_boundary"] - res["v2_phase1_baseline"]) / 1e6, 3))
repo = yaml.safe_load(open("configs/v2_phase1_baseline.yaml"))["model"]["official_repo_dir"]
sys.path.insert(0, repo)
from architectures.segformer3d import SegFormer3D as Off
o = Off(in_channels=3, sr_ratios=[4,2,1,1], embed_dims=[32,64,160,256], patch_kernel_size=[7,3,3,3], patch_stride=[4,2,2,2], patch_padding=[3,1,1,1], mlp_ratios=[4,4,4,4], num_heads=[1,2,5,8], depths=[2,2,2,2], decoder_head_embedding_dim=256, num_classes=3, decoder_dropout=0.0)
print("official class, 3 channels (M):", round(count(o) / 1e6, 3))
print("SMOKE OK")
