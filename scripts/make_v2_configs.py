import copy, yaml
b1 = yaml.safe_load(open("configs/phase1_baseline.yaml"))
b2 = yaml.safe_load(open("configs/phase2_boundary.yaml"))
OFF = dict(official_repo_dir="/workspace/SegFormer3D-official", in_channels=4, num_classes=3,
           embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], sr_ratios=[4, 2, 1, 1], depths=[2, 2, 2, 2],
           mlp_ratios=[4, 4, 4, 4], patch_kernel_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2],
           patch_padding=[3, 1, 1, 1], decoder_head_embedding_dim=256, decoder_dropout=0.0)

def write(name, cfg, ck):
    cfg["experiment_name"] = name
    cfg["train"]["checkpoint_dir"] = f"checkpoints/{ck}"
    cfg["train"]["log_dir"] = f"logs/{ck}"
    yaml.safe_dump(cfg, open(f"configs/{name}.yaml", "w"), sort_keys=False)

c = copy.deepcopy(b1); c["model"] = {"type": "segformer3d_official", **OFF}
write("v2_phase1_baseline", c, "v2_phase1")

c = copy.deepcopy(b2)   # control: plain baseline continued with the Phase 2 recipe
c["model"] = {"type": "segformer3d_official", "init_from_phase1_checkpoint": "checkpoints/v2_phase1/best.pt", **OFF}
c["train"]["loss"] = {"type": "dice_bce", "dice_weight": 1.0, "bce_weight": 1.0}
c["data"]["return_boundary_target"] = False
c["eval"].pop("eval_output", None)
write("v2_phase1_continued", c, "v2_continued")

BND = {"type": "segformer3d_official_boundary", "init_from_phase1_checkpoint": "checkpoints/v2_phase1/best.pt",
       "boundary_dim": 128, "voxel_spacing": [1.0, 1.0, 1.0], **OFF}
c = copy.deepcopy(b2); c["model"] = copy.deepcopy(BND)
write("v2_phase2_boundary", c, "v2_phase2")
c = copy.deepcopy(b2); c["model"] = copy.deepcopy(BND); c["train"]["loss"]["boundary_weight"] = 0.0
write("v2_abl_noloss", c, "v2_abl_noloss")
print("configs written")
