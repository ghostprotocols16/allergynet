import json
import random

# Paths
input_json = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\train_ready_normalized.json"
train_json = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\train_split.json"
virtual_val_json = r"C:\Users\harsh\OneDrive\Desktop\allergy_Net\allergy_net\dataset\virtual_val.json"

# Load dataset
with open(input_json, 'r') as f:
    data = json.load(f)

# Shuffle data
random.shuffle(data)

# Keep only 2% for virtual validation
split_ratio = 0.02
split_index = int(len(data) * split_ratio)

virtual_val_data = data[:split_index]
train_data = data[split_index:]

# Set image to None for virtual validation
for item in virtual_val_data:
    item['image'] = None

# Save splits
with open(train_json, 'w') as f:
    json.dump(train_data, f, indent=2)
with open(virtual_val_json, 'w') as f:
    json.dump(virtual_val_data, f, indent=2)

print(f"Training data: {len(train_data)} images")
print(f"Virtual validation data: {len(virtual_val_data)} entries")
