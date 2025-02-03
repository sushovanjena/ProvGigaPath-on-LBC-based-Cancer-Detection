import os
import csv

# Define the input folders
base_folders = [
    "GigaPath_embeddings_LBC_train",
    "GigaPath_embeddings_LBC_test",
    "GigaPath_embeddings_LBC_valid"
]

# Output CSV file
output_csv = "merged_embeddings.csv"

# Prepare the data
data = []

for folder in base_folders:
    # Extract the split from the folder name (last word)
    split = folder.split('_')[-1]

    # Traverse the subfolders (e.g., '0' and '1')
    for label in ['0', '1']:
        subfolder_path = os.path.join(folder, label)
        
        if not os.path.exists(subfolder_path):
            print(f"Skipping non-existent folder: {subfolder_path}")
            continue

        # List all files in the subfolder
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)

            # Add entry to the data list
            if os.path.isfile(file_path):
                data.append({
                    'path': file_path,
                    'label': label,
                    'split': split
                })

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['path', 'label', 'split'])
    writer.writeheader()
    writer.writerows(data)

print(f"CSV file '{output_csv}' has been created successfully.")
