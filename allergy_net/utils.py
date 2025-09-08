from transformers import ViTFeatureExtractor

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def preprocess_images(images):
    """
    images: list of PIL images
    Returns PyTorch tensors ready for ViT
    """
    return feature_extractor(images=images, return_tensors="pt")['pixel_values']
