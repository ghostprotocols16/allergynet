import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ===========================
# Paths
# ===========================
MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\models\vit_epoch1.pth"
TOP_5000_JSON = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\top_5000_ingredients.json"
DATA_JSON = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\clean_core_ingredients.json"

# ===========================
# Device
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# Load JSONs
# ===========================
with open(TOP_5000_JSON, 'r') as f:
    top_ingredients = json.load(f)
label_map = {ingredient: idx for idx, ingredient in enumerate(top_ingredients)}

with open(DATA_JSON, 'r') as f:
    data = json.load(f)

# ===========================
# Model
# ===========================
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
num_labels = len(top_ingredients)
model.heads.head = nn.Linear(model.heads.head.in_features, num_labels)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ===========================
# Transform
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# Prediction Function
# ===========================
def predict_image(image_path, allergen_name):
    if allergen_name not in label_map:
        print(f"Allergen '{allergen_name}' not in top 5000 list.")
        return

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).squeeze(0)

    allergen_idx = label_map[allergen_name]
    confidence = probs[allergen_idx].item()

    # Find nutrition info if allergen exists in any recipe
    matched_recipes = []
    for img_name, info in data.items():
        core_ingredients = info.get("core_ingredients", [])
        if allergen_name in core_ingredients:
            matched_recipes.append({
                "title": info.get("title"),
                "yield": info.get("yield"),
                "calories": info.get("calories"),
                "fat_g": info.get("fat_g"),
                "protein_g": info.get("protein_g"),
                "carbs_g": info.get("carbs_g"),
                "sugar_g": info.get("sugar_g")
            })

    print(f"\nPrediction for allergen '{allergen_name}':")
    print(f"Confidence: {confidence:.4f}")
    if matched_recipes:
        print("\nNutrition info from recipes containing this allergen:")
        for r in matched_recipes[:3]:  # show max 3 recipes
            print(f"- {r['title']} | Calories: {r['calories']} kcal | Protein: {r['protein_g']} g | Carbs: {r['carbs_g']} g | Fat: {r['fat_g']} g | Sugar: {r['sugar_g']} g")
    else:
        print("No recipes with this allergen found in dataset.")

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    allergen_name = input("Enter allergen (from top 5000 list): ").strip().lower()
    predict_image(img_path, allergen_name)
