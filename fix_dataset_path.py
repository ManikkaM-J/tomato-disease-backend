import re

path = r"D:\PROJECT\tomato_leaf_backend_v2\train_stage1_validator.py"

with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip().startswith("STAGE1_DIR") and "=" in line:
        line = 'STAGE1_DIR = r"D:\\PROJECT\\tomato_leaf_backend_v2\\Validator_Dataset"\n'
    elif line.strip().startswith("DATASET_DIR") and "=" in line:
        line = 'DATASET_DIR = r"D:\\PROJECT\\tomato_leaf_backend_v2\\Validator_Dataset"\n'
    new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print("Done! Verifying changes...")

# Verify
with open(path, "r") as f:
    for i, line in enumerate(f, 1):
        if "STAGE1_DIR" in line or "DATASET_DIR" in line:
            if "=" in line and "os.path" not in line:
                print(f"  Line {i}: {line.strip()}")

print("Update complete!")
