import os, re, shutil, glob

def patch(path, edits):
    if not os.path.exists(path):
        print(f"!! {path} not found (run from project root)"); return
    src = open(path).read(); out = src
    for name, old, new, marker in edits:
        if marker in out:
            print(f"   already applied : {path} :: {name}"); continue
        if out.count(old) != 1:
            print(f"   MISSING ({out.count(old)} matches): {path} :: {name}  -> edit by hand"); continue
        out = out.replace(old, new); print(f"   applied         : {path} :: {name}")
    if out != src:
        if not os.path.exists(path + ".bak"): shutil.copy(path, path + ".bak")
        open(path, "w").write(out)

patch("src/models/segformer3d.py", [
 ("decoder signature",
  "    def forward(self, feats: List[torch.Tensor], out_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:",
  "    def forward(self, feats: List[torch.Tensor], out_shape: Tuple[int, int, int], feature_mode: str = \"full\") -> Tuple[torch.Tensor, torch.Tensor]:",
  "feature_mode: str"),
 ("decoder native return",
  "        fused_up = F.interpolate(fused, size=out_shape, mode=\"trilinear\", align_corners=False)\n        return logits, fused_up",
  "        if feature_mode == \"native\":      # decoder's own (stride-4) grid, no full-res upsampling\n            return logits, fused\n        fused_up = F.interpolate(fused, size=out_shape, mode=\"trilinear\", align_corners=False)\n        return logits, fused_up",
  "feature_mode == \"native\""),
 ("model signature",
  "    def forward(self, x: torch.Tensor, return_decoder_features: bool = False):",
  "    def forward(self, x: torch.Tensor, return_decoder_features: bool = False, feature_mode: str = \"full\"):",
  "return_decoder_features: bool = False, feature_mode"),
 ("model decoder call",
  "logits, decoder_feat = self.decoder(feats, out_shape=in_shape)",
  "logits, decoder_feat = self.decoder(feats, out_shape=in_shape, feature_mode=feature_mode)",
  "feature_mode=feature_mode"),
])

patch("src/models/boundary_refinement.py", [
 ("branch __init__",
  "    def __init__(self, decoder_dim: int, num_classes: int, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):\n        super().__init__()\n        self.sobel =",
  "    def __init__(self, decoder_dim: int, num_classes: int, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),\n                 mutually_exclusive: bool = False):\n        super().__init__()\n        self.mutually_exclusive = mutually_exclusive\n        self.sobel =",
  "self.mutually_exclusive = mutually_exclusive"),
 ("softmax -> sigmoid",
  "        region_prob = torch.softmax(region_logits, dim=1)\n        edge_prior = self.sobel(region_prob)  # (B, num_classes, D, H, W)",
  "        # WT/TC/ET overlap and are trained with sigmoid, so the soft prediction must be sigmoid\n        if self.mutually_exclusive:\n            region_prob = torch.softmax(region_logits.float(), dim=1)\n        else:\n            region_prob = torch.sigmoid(region_logits.float())\n        edge_prior = self.sobel(region_prob)  # (B, num_classes, D, H, W), full-res\n        if edge_prior.shape[2:] != decoder_feat.shape[2:]:   # native-resolution branch\n            edge_prior = F.adaptive_max_pool3d(edge_prior, decoder_feat.shape[2:])\n        edge_prior = edge_prior.to(decoder_feat.dtype)",
  "torch.sigmoid(region_logits.float())"),
 ("wrapper __init__",
  "    def __init__(self, base_model: nn.Module, decoder_dim: int, num_classes: int, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):\n        super().__init__()\n        self.base_model = base_model\n        self.boundary_branch = BoundaryBranch3D(decoder_dim, num_classes, voxel_spacing)",
  "    def __init__(self, base_model: nn.Module, decoder_dim: int, num_classes: int, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),\n                 mutually_exclusive: bool = False, branch_resolution: str = \"native\"):\n        super().__init__()\n        assert branch_resolution in (\"native\", \"full\"), branch_resolution\n        self.branch_resolution = branch_resolution\n        self.base_model = base_model\n        self.boundary_branch = BoundaryBranch3D(decoder_dim, num_classes, voxel_spacing, mutually_exclusive)",
  "branch_resolution: str"),
 ("wrapper forward",
  "        region_logits, decoder_feat = self.base_model(x, return_decoder_features=True)\n        boundary_logits, boundary_feat = self.boundary_branch(decoder_feat, region_logits)\n        refined_logits = self.fusion(decoder_feat, boundary_feat)\n",
  "        region_logits, decoder_feat = self.base_model(\n            x, return_decoder_features=True, feature_mode=self.branch_resolution)\n        boundary_logits, boundary_feat = self.boundary_branch(decoder_feat, region_logits)\n        refined_logits = self.fusion(decoder_feat, boundary_feat)\n        out_shape = x.shape[2:]\n        if boundary_logits.shape[2:] != out_shape:\n            boundary_logits = F.interpolate(boundary_logits, size=out_shape, mode=\"trilinear\", align_corners=False)\n        if refined_logits.shape[2:] != out_shape:\n            refined_logits = F.interpolate(refined_logits, size=out_shape, mode=\"trilinear\", align_corners=False)\n",
  "feature_mode=self.branch_resolution"),
])

patch("src/train.py", [
 ("pass new model args",
  "            voxel_spacing=tuple(m.get(\"voxel_spacing\", (1.0, 1.0, 1.0))),\n        )",
  "            voxel_spacing=tuple(m.get(\"voxel_spacing\", (1.0, 1.0, 1.0))),\n            mutually_exclusive=cfg[\"data\"].get(\"mutually_exclusive\", False),\n            branch_resolution=m.get(\"branch_resolution\", \"native\"),\n        )",
  "branch_resolution=m.get("),
 ("--manifest/--data_dir args",
  "    parser.add_argument(\"--resume\", default=None)\n",
  "    parser.add_argument(\"--resume\", default=None)\n    parser.add_argument(\"--manifest\", default=None, help=\"override data.manifest\")\n    parser.add_argument(\"--data_dir\", default=None, help=\"override data.data_dir\")\n",
  "\"--manifest\""),
 ("apply overrides",
  "        cfg = yaml.safe_load(f)\n",
  "        cfg = yaml.safe_load(f)\n    if args.manifest: cfg[\"data\"][\"manifest\"] = args.manifest\n    if args.data_dir: cfg[\"data\"][\"data_dir\"] = args.data_dir\n",
  "cfg[\"data\"][\"manifest\"] = args.manifest"),
])
patch("src/evaluate.py", [
 ("--manifest/--data_dir args",
  "    parser.add_argument(\"--output_json\", default=None)\n",
  "    parser.add_argument(\"--output_json\", default=None)\n    parser.add_argument(\"--manifest\", default=None, help=\"override data.manifest\")\n    parser.add_argument(\"--data_dir\", default=None, help=\"override data.data_dir\")\n",
  "\"--manifest\""),
 ("apply overrides",
  "        cfg = yaml.safe_load(f)\n",
  "        cfg = yaml.safe_load(f)\n    if args.manifest: cfg[\"data\"][\"manifest\"] = args.manifest\n    if args.data_dir: cfg[\"data\"][\"data_dir\"] = args.data_dir\n",
  "cfg[\"data\"][\"manifest\"] = args.manifest"),
])

patch("scripts/prepare_brats.py", [
 ("recursive case discovery (HGG/LGG)",
  "    case_dirs = sorted([d for d in glob(os.path.join(args.raw_dir, \"*\")) if os.path.isdir(d)])",
  "    # 2017 nests cases inside HGG/ and LGG/: a case = any folder holding a *seg.nii* file\n    case_dirs = sorted({os.path.dirname(p) for p in glob(os.path.join(args.raw_dir, \"**\", \"*seg.nii*\"), recursive=True)})",
  "recursive=True"),
 ("skip no-op resample",
  "    zoom_factors = [o / t for o, t in zip(orig_spacing, target_spacing)]\n",
  "    zoom_factors = [o / t for o, t in zip(orig_spacing, target_spacing)]\n    if all(abs(f - 1.0) < 1e-3 for f in zoom_factors):   # BraTS is already 1 mm\n        return volume\n",
  "BraTS is already 1 mm"),
 ("json-safe spacing",
  "\"orig_spacing\": list(orig_spacing)",
  "\"orig_spacing\": [float(s) for s in orig_spacing]",
  "[float(s) for s in orig_spacing]"),
 ("label sanity check",
  "    # raw BraTS labels {0,1,2,4} -> contiguous {0,1,2,3}\n    out = label.copy()",
  "    # raw BraTS labels {0,1,2,4} -> contiguous {0,1,2,3}\n    _vals = set(np.unique(label).tolist())\n    if not _vals <= {0, 1, 2, 4}:\n        raise ValueError(f\"unexpected BraTS label values {sorted(_vals)} (wrong dataset/format?)\")\n    out = label.copy()",
  "unexpected BraTS label values"),
])

for cfgp in sorted(glob.glob("configs/*.yaml")):
    s = open(cfgp).read(); o = s
    s = re.sub(r"data/processed/BraTS\b(?!2017)", "data/processed/BraTS2017", s)
    if "type: segformer3d_boundary" in s and "branch_resolution" not in s:
        s = re.sub(r"(\n  voxel_spacing: [^\n]*\n)", lambda m: m.group(1) + "  # native = boundary branch on the decoder grid (cheap) | full = full-res branch (~50x compute)\n  branch_resolution: native\n", s, count=1)
    if s != o:
        if not os.path.exists(cfgp + ".bak"): shutil.copy(cfgp, cfgp + ".bak")
        open(cfgp, "w").write(s); print(f"   updated config  : {cfgp}")
print("done")
