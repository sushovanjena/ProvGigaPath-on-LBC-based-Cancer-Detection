import pandas as pd
import os

# SFTP Path
sftp_path = "/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra"

# File paths
train_file = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/train_data_Prov.csv")
test_file = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/final_test_data_otsu_rectfd_1.csv")
valid_file = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/valid_data_Prov.csv")

# Read the CSV files
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
valid_df = pd.read_csv(valid_file)

# Add the 'split' column to each DataFrame
train_df['split'] = 'train'
test_df['split'] = 'test'
valid_df['split'] = 'valid'

# Rename columns to match the required format
train_df = train_df.rename(columns={
    'Folder_Name': 'input',
    'Label': 'label',
    'WebP_Image_Path': 'path'
})

test_df = test_df.rename(columns={
    'Folder_Name': 'input',
    'Label': 'label',
    'WebP_Image_Path': 'path'
})

valid_df = valid_df.rename(columns={
    'Folder_Name': 'input',
    'Label': 'label',
    'WebP_Image_Path': 'path'
})

# Merge all DataFrames
merged_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

# Select required columns
merged_df = merged_df[['input', 'label', 'split', 'path']]

# Output file path
output_file = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/merged_data_train_test_valid_Prov.csv")

# Save the merged CSV
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved at: {output_file}")
