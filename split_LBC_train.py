import pandas as pd
import os

# Define the SFTP path
sftp_path = "/run/user/1000/gvfs/sftp:host=192.168.1.11,user=aindra"
csv_file = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/final_train_data_otsu_rectfd_1.csv")

# Load the CSV file
df = pd.read_csv(csv_file)

# Split into 80% train and 20% test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save the splits
train_csv_path = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/train_data_Prov.csv")
test_csv_path = os.path.join(sftp_path, "media/Data/2d_annot_model_lbc/dataset/test_data_Prov.csv")

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train CSV saved at: {train_csv_path}")
print(f"Test CSV saved at: {test_csv_path}")
