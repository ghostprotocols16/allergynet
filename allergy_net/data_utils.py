import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class AllergenDataset(Dataset):
    def __init__(self, data_json, image_folder, label_map_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Load JSONs
        with open(data_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(label_map_file, 'r', encoding='utf-8') as f:
            self.label_list = json.load(f)

        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_list)}

        # Build samples list
        self.samples = []
        for img_name, info in tqdm(self.data.items(), desc="Building dataset"):
            img_path = os.path.join(image_folder, img_name)
            if os.path.exists(img_path):
                labels = torch.zeros(len(self.label_list))
                for ing in info.get("core_ingredients", []):
                    if ing in self.label_to_idx:
                        labels[self.label_to_idx[ing]] = 1
                self.samples.append((img_path, labels))
            else:
                # skip missing images
                pass

        print(f"Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, labels
