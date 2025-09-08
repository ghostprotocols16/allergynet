import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pillow_heif
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

# ===========================
# HEIC Support
# ===========================
pillow_heif.register_heif_opener()

# ===========================
# Paths
# ===========================
DATA_JSON = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\clean_core_ingredients.json"
TOP_5000_JSON = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\top_5000_ingredients.json"
IMAGE_FOLDER = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\images"
SAVE_MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\models"

# ===========================
# Dataset
# ===========================
class AllergenDataset(Dataset):
    def __init__(self, data_json, image_folder, top_ingredients_json, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        with open(data_json, 'r') as f:
            self.data = json.load(f)
        with open(top_ingredients_json, 'r') as f:
            self.top_ingredients = json.load(f)

        # Map ingredient to index
        self.label_map = {ingredient: idx for idx, ingredient in enumerate(self.top_ingredients)}

        self.samples = []
        for img_name, info in self.data.items():
            img_path = os.path.join(self.image_folder, img_name)
            if not os.path.exists(img_path):
                continue

            # Convert core ingredients to multi-hot vector
            labels = torch.zeros(len(self.top_ingredients), dtype=torch.float32)
            for ingredient in info.get("core_ingredients", []):
                if ingredient in self.label_map:
                    labels[self.label_map[ingredient]] = 1.0

            self.samples.append((img_name, labels))

        print(f"Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, labels = self.samples[idx]
        img_path = os.path.join(self.image_folder, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: skipping unreadable image {img_name}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # black image placeholder
        if self.transform:
            image = self.transform(image)
        return image, labels

# ===========================
# Transforms
# ===========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ===========================
# Device
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# ===========================
# Dataset & Loader
# ===========================
dataset = AllergenDataset(DATA_JSON, IMAGE_FOLDER, TOP_5000_JSON, transform=train_transform)
train_loader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True if device.type=='cuda' else False
)

# ===========================
# Model
# ===========================
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
num_labels = len(dataset[0][1])
model.heads.head = nn.Linear(model.heads.head.in_features, num_labels)
model = model.to(device)

# ===========================
# Loss & Optimizer
# ===========================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===========================
# Training Loop
# ===========================
num_epochs = 10

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            if i % 50 == 0:
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # Save checkpoint
        os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"vit_epoch{epoch+1}.pth"))

    print("Training complete!")
