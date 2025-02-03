import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import timm
import os
# Load models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder.eval()
tile_encoder = tile_encoder.to(device)

print("param #", sum(p.numel() for p in tile_encoder.parameters()))


class LinearProbe(nn.Module):

    def __init__(self, embed_dim: int = 1536, num_classes: int = 10):
        super(LinearProbe, self).__init__()

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def load_models(model, linear_probe_path):
    # Load embedding model
    # embedding_model = torch.load(embedding_model_path)
    # embedding_model.eval()
    
    # Load linear probe
    model.load_state_dict(torch.load(linear_probe_path))
    # linear_probe = torch.load(linear_probe_path)
    model.eval()
    model = model.to(device)
    
    return model
# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose(
    [
        transforms.Resize(962, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Generate embeddings for an image
def generate_embedding(image_path, embedding_model):

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = embedding_model(image_tensor)
    return embedding

# Classify tiles and compute confidence score
# Classify tiles and compute confidence score
def classify_tiles(main_directory_path, linear_probe):
    # Locate the `tiles_files/17` subfolder
    target_folder = os.path.join(main_directory_path, 'pyramid', 'tiles_files', '17')
    if not os.path.exists(target_folder):
        print(f"Target folder not found: {target_folder}")
        return [], 0
    
    # Get all tile files inside the `17` subfolder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    tile_paths = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not tile_paths:
        print("No valid image files found in the '17' folder.")
        return [], 0
    
    positive_count = 0
    negative_count = 0
    results = []
    
    for tile_path in tile_paths:
        # tile_path = tile_path.to(device)
        # Generate embedding
        embedding = generate_embedding(tile_path, tile_encoder)
        
        # Classify using the linear probe
        with torch.no_grad():
            prediction = linear_probe(embedding)
            predicted_label = torch.argmax(prediction, dim=1).item()
        
        # Count positive and negative tiles
        if predicted_label == 1:  # Assuming '1' is positive
            positive_count += 1
        else:
            negative_count += 1
        
        # Save the result for this tile
        results.append({"tile_path": tile_path, "prediction": predicted_label})
    
    # Compute confidence score
    total = positive_count + negative_count
    confidence_score = positive_count / total if total > 0 else 0
    
    return results, confidence_score

# Main function
def main():
    # Paths to model checkpoints
    # embedding_model_path = "path/to/embedding_model.pth"
    linear_probe_path = "/home/aindra/prov-gigapath/outputs/LBC/best_model.pth"
    model = LinearProbe(1536, 2)
    # Input CSV file
    # csv_file = "path/to/input.csv"
    
    # Load models
    linear_probe = load_models(model, linear_probe_path)
    
    # Classify tiles
    results, confidence_score = classify_tiles('/home/aindra/marketing_cases_cyto/4_AINDRAAS0002C00-2779CS20',linear_probe)
    
    # Print results
    print("Classification Results:")
    for result in results:
        print(result)
    
    print(f"\nConfidence Score: {confidence_score:.2f}")

if __name__ == "__main__":
    main()
