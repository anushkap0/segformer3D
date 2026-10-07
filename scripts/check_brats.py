from pathlib import Path
import nibabel as nib

ROOT = Path(r"data\raw\archive\BraTS2021_Training_Data")

bad = 0

for case_dir in sorted(ROOT.iterdir()):
    if not case_dir.is_dir():
        continue

    print(f"\nChecking {case_dir.name}")

    for nii_file in sorted(case_dir.glob("*.nii.gz")):
        try:
            img = nib.load(str(nii_file))

            # Force actual data reading.
            data = img.get_fdata(dtype="float32")

            print(f"  OK   {nii_file.name}  shape={data.shape}")

        except Exception as e:
            bad += 1
            print(f"  BAD  {nii_file.name}")
            print(f"       {type(e).__name__}: {e}")

print("\n================================")
print(f"Bad files found: {bad}")
print("================================")