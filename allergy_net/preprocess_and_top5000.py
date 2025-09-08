import json
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -------------------------
# File paths
# -------------------------
INPUT_JSON = "C:/Users/harsh/OneDrive/Desktop/allergy_Net/allergy_net/dataset/image_linked_dataset.json"
PROCESSED_JSON = "C:/Users/harsh/OneDrive/Desktop/allergy_Net/allergy_net/dataset/processed_dataset.json"
TOP5000_JSON = "C:/Users/harsh/OneDrive/Desktop/allergy_Net/allergy_net/dataset/top_5000_ingredients.json"


# -------------------------
# Preprocessing helpers
# -------------------------
STOPWORDS = set(stopwords.words("english"))
UNITS = ["cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
         "g", "kg", "ml", "l", "oz", "pinch", "dash", "slice", "slices", "lb", "lbs"]

lemmatizer = WordNetLemmatizer()

def clean_ingredient(text):
    text = text.lower()
    # Remove digits and punctuation
    text = re.sub(r"[\d/]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Remove units
    words = [w for w in text.split() if w not in UNITS and w not in STOPWORDS]
    # Lemmatize
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

# -------------------------
# Load original dataset
# -------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------
# Preprocess ingredients
# -------------------------
for img_id, details in data.items():
    core = []
    for ingredient in details["ingredients"].split("|"):
        core += clean_ingredient(ingredient)
    # Optional: unify variations (e.g., green chili -> chili)
    core = [w.split()[-1] for w in core if w]  # keep last word as main ingredient
    data[img_id]["core_ingredients"] = list(set(core))  # remove duplicates

# Save processed dataset
with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Preprocessing complete. Saved to {PROCESSED_JSON}")

# -------------------------
# Generate top 5000 ingredients
# -------------------------
ingredient_counter = Counter()
for details in data.values():
    ingredient_counter.update(details["core_ingredients"])

top_5000 = [ing for ing, _ in ingredient_counter.most_common(5000)]

with open(TOP5000_JSON, "w", encoding="utf-8") as f:
    json.dump(top_5000, f, ensure_ascii=False, indent=2)

print(f"✅ Top 5000 ingredients saved to {TOP5000_JSON}")
