# File: src/normalize_all_core_ingredients.py
import json
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

JSON_PATH = "../dataset/processed_dataset.json"
OUTPUT_PATH = "../dataset/normalized_core_ingredients.json"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Common descriptors/colors to remove
DESCRIPTORS = ["green", "red", "yellow", "fresh", "dried", "chopped", "sliced", "large", "small", "medium", "lightly", "peeled", "grated", "minced"]

# Units already removed in previous step, but double-check
UNITS = ["cup", "tablespoon", "teaspoon", "g", "kg", "ml", "l", "pinch", "oz"]

def normalize_word(word):
    word = word.lower().strip()
    # Remove descriptors
    for desc in DESCRIPTORS:
        word = re.sub(rf"\b{desc}\b", "", word)
    # Remove units
    for unit in UNITS:
        word = re.sub(rf"\b{unit}\b", "", word)
    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)
    # Remove stopwords
    if word in stop_words or len(word) == 0:
        return None
    # Lemmatize to singular form
    word = lemmatizer.lemmatize(word)
    return word

# Load dataset
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Normalize core ingredients
for key, value in data.items():
    core_ings = value.get("core_ingredients", [])
    normalized = [normalize_word(w) for w in core_ings]
    # Remove None and duplicates
    value["core_ingredients"] = list(set([w for w in normalized if w]))

# Save normalized dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ All core ingredients normalized and saved.")
