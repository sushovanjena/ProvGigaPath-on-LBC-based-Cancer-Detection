import timm
from PIL import Image
from torchvision import transforms
import torch
import os
import pandas as pd
from pathlib import Path

sftp_path = "/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra"

# assert "HF_TOKEN" in os.environ, "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"

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder.eval()

print("param #", sum(p.numel() for p in tile_encoder.parameters()))

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# img_path = "images/prov_normal_000_1.png"
# sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

# Paths and directories
csv_file = os.path.join(sftp_path,"media/Data/2d_annot_model_lbc/dataset/final_test_data_otsu_rectfd_1.csv")
print('csv_path', csv_file)
output_folder = "GigaPath_embeddings_LBC_test"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load CSV file
data = pd.read_csv(csv_file)

# Ensure columns exist
assert "WebP_Image_Path" in data.columns, "The CSV file must have an 'image_path' column."
assert "Label" in data.columns, "The CSV file must have a 'label' column."

# data_iterator = iter(data.iterrows())
# index, row = next(data_iterator)  # Get the first row
# sftp_path = "/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra"
# image_path = f'/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra{row["WebP_Image_Path"]}'
# print('image_path', image_path)
# label = row["Label"]
# try:
#     # Transform the image
#     img = Image.open(image_path).convert("RGB")
#     input_tensor = transform(img).unsqueeze(0)

#     # Run inference
#     with torch.no_grad():
#         output = tile_encoder(input_tensor).squeeze()

#     # Determine output subfolder based on label
#     label_folder = os.path.join(output_folder, f"label_{label}")
#     os.makedirs(label_folder, exist_ok=True)

#     # Save the output tensor with 'train_' prefix
#     output_file = os.path.join(label_folder, f"train_{Path(image_path).stem}.pt")
#     torch.save(output, output_file)
#     print(f"Processed {image_path}, saved to {output_file}")

# except Exception as e:
#     print(f"Error processing {image_path}: {e}")

# print("Processed one row from the CSV.")



# Iterate over rows in the CSV
for index, row in data.iterrows():
    sftp_path = "/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra"
    image_path = f'/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra{row["WebP_Image_Path"]}'
    print('image_path', image_path)
    label = row["Label"]

    try:
        # Transform the image
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = tile_encoder(input_tensor).squeeze()
            print("Model output:", output.shape)
            print(output)

        # Determine output subfolder based on label
        label_folder = os.path.join(output_folder, f"{label}")
        os.makedirs(label_folder, exist_ok=True)

        # Save the output tensor
        # Extract the last three parts of the path
        path_parts = Path(image_path).parts[-5:]

        # Replace the last part with its stem (without extension)
        path_parts = list(path_parts)
        path_parts[-1] = Path(path_parts[-1]).stem
        # Join parts with underscores and add the .pt extension
        new_filename = "_".join(path_parts) + ".pt"
        output_file = os.path.join(label_folder, f"{Path(new_filename).stem}.pt")
        torch.save(output, output_file)
        print(f"Processed {image_path}, saved to {output_file}")

        expected_output = torch.load(output_file)
        print("Expected output:", expected_output.shape)
        print(expected_output)

        assert torch.allclose(output, expected_output, atol=1e-2)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print(f"Inference completed. Outputs saved in {output_folder}.")



# with torch.no_grad():
#     output = tile_encoder(sample_input).squeeze()
#     print("Model output:", output.shape)
#     print(output)

# expected_output = torch.load("images/prov_normal_000_1.pt")
# print("Expected output:", expected_output.shape)
# print(expected_output)

# assert torch.allclose(output, expected_output, atol=1e-2)
