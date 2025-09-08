import torch

# Paths
OLD_CKPT = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\models\vit_allergen_epoch10.pth"
NEW_CKPT = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\models\vit_allergen_epoch10_fixed.pth"

# Load checkpoint
state_dict = torch.load(OLD_CKPT, map_location="cpu")

# Rename keys
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model.heads.weight"):
        new_state_dict["model.heads.head.weight"] = v
    elif k.startswith("model.heads.bias"):
        new_state_dict["model.heads.head.bias"] = v
    else:
        new_state_dict[k] = v

# Save fixed checkpoint
torch.save(new_state_dict, NEW_CKPT)
print(f"✅ Fixed checkpoint saved to {NEW_CKPT}")
