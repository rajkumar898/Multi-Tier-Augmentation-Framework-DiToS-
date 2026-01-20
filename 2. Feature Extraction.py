#####################
# FEATURE EXTRACTION
#####################

import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Paths
clean_path = './Detect_solar_dust/Clean'
dusty_path = './Detect_solar_dust/Dusty'
generated_path = r'D:\Raj Kumar\Hybrid Augmentation Approach\generated_solar_faults'

# Collect selected image paths
# 150 Clean images
clean_files = [os.path.join(clean_path, f) for f in os.listdir(clean_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:750]
# 30 original Dusty images
dusty_files = [os.path.join(dusty_path, f) for f in os.listdir(dusty_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:150]
# 60 generated images
generated_files = [os.path.join(generated_path, f) for f in os.listdir(generated_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:300]

# Combine Dusty and Generated files for Dusty category
dusty_all_files = dusty_files + generated_files

# Load pretrained DenseNet121
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()
model.to(device)

# Remove classifier, keep only feature extractor
feature_extractor = model.features

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Feature extraction function
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)  # Flatten to (1, 1024)
    return features.cpu().numpy()

# Extract features for Clean images
clean_features_list = []
clean_file_names = []
for img_path in tqdm(clean_files, desc='Extracting features (Clean)'):
    features = extract_features(img_path)
    clean_features_list.append(features[0])
    clean_file_names.append(os.path.basename(img_path))

# Convert Clean features to numpy array → shape (150, 1024)
clean_features_array = np.array(clean_features_list)

# Save Clean features directly in current directory
np.save('CLEAN_FEATURES.npy', clean_features_array)
print("✅ Saved to CLEAN_FEATURES.npy with shape:", clean_features_array.shape)

# Save Clean features to .csv
clean_df = pd.DataFrame(clean_features_array)
clean_df.insert(0, 'filename', clean_file_names)
clean_df.insert(1, 'label', [0] * len(clean_files))  # Label: 0 for Clean
clean_df.to_csv('CLEAN_FEATURES.csv', index=False)
print("✅ Saved to CLEAN_FEATURES.csv with shape:", clean_df.shape)

# Extract features for Dusty images (original + generated)
dusty_features_list = []
dusty_file_names = []
for img_path in tqdm(dusty_all_files, desc='Extracting features (Dusty)'):
    features = extract_features(img_path)
    dusty_features_list.append(features[0])
    dusty_file_names.append(os.path.basename(img_path))

# Convert Dusty features to numpy array → shape (90, 1024)
dusty_features_array = np.array(dusty_features_list)

# Save Dusty features directly in current directory
np.save('DUSTY_FEATURES.npy', dusty_features_array)
print("✅ Saved to DUSTY_FEATURES.npy with shape:", dusty_features_array.shape)

# Save Dusty features to .csv
dusty_df = pd.DataFrame(dusty_features_array)
dusty_df.insert(0, 'filename', dusty_file_names)
dusty_df.insert(1, 'label', [1] * len(dusty_all_files))  # Label: 1 for Dusty/Generated
dusty_df.to_csv('DUSTY_FEATURES.csv', index=False)
print("✅ Saved to DUSTY_FEATURES.csv with shape:", dusty_df.shape)


