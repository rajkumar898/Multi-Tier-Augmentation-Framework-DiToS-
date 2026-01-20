##################
# LOADING DATASET
##################

import os
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = './Detect_solar_dust'

# Define class folders
classes = ['Clean', 'Dusty']

# Count images
image_counts = {}
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_counts[cls] = len(image_files)

# Print counts
print("Image counts per class:")
for cls, count in image_counts.items():
    print(f"{cls}: {count}")

# Assign colors (one per class)
colors = ['#1f77b4', '#ff7f0e']  # Clean → blue, Dusty → orange

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(image_counts.keys(), image_counts.values(), color=colors)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.show()

#####################################
# TAKING ONLY 150 DIRTY/DUSTY SAMPLE
#####################################

import os
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = './Detect_solar_dust'

# Define class folders
classes = ['Clean', 'Dusty']

# Initialize image count dictionary
image_counts = {}

for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if cls == 'Dusty':
        # Take 30 images from Dusty
        selected_files = image_files[:150]
        image_counts[cls] = len(selected_files)
    else:
        # Take 150 images from Clean
        selected_files = image_files[:750]
        image_counts[cls] = len(selected_files)

# Print counts
print("Selected image counts per class:")
for cls, count in image_counts.items():
    print(f"{cls}: {count}")

# Assign colors (unique for each class)
colors = ['#1f77b4', '#ff7f0e']  # Clean → blue, Dusty → orange

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(image_counts.keys(), image_counts.values(), color=colors)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
#plt.savefig('image_counts_per_class.png')
plt.show()

###########################################################
# ADDING SAMPLES FROM STABLE DIFFUSION FOR PARTIAL BALANCE
###########################################################

import os
import matplotlib.pyplot as plt

# Define the dataset paths
dataset_path = './Detect_solar_dust'
generated_path = r'D:\Raj Kumar\Hybrid Augmentation Approach\generated_solar_faults'

# Define class folders
classes = ['Clean', 'Dusty']

# Initialize image count dictionary
image_counts = {}

# Process Clean and Dusty classes
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if cls == 'Dusty':
        # Take 30 original Dusty images
        selected_dusty_files = image_files[:150]
                
        # Count 60 generated images
        generated_files = [f for f in os.listdir(generated_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        selected_generated_files = generated_files[:300]
        
        total_dusty = len(selected_dusty_files) + len(selected_generated_files)
        image_counts[cls] = total_dusty
    else:
        # Take 150 Clean images
        selected_clean_files = image_files[:750]
        image_counts[cls] = len(selected_clean_files)

# Print counts
print("Final image counts per class:")
for cls, count in image_counts.items():
    print(f"{cls}: {count}")
    
# Assign colors (unique for each class)
colors = ['#1f77b4', '#ff7f0e']  # Clean → blue, Dusty → orange

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(image_counts.keys(), image_counts.values(), color=colors)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class (Dusty: 30 Original + 60 Generated)')
#plt.savefig('image_counts_per_class.png')
plt.show()

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

######################
# APPLYING TOMEK LINK
######################

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import TomekLinks

# ==============================
# 🔥 Load your features
# ==============================
clean_df = pd.read_csv('CLEAN_FEATURES.csv')
dusty_df = pd.read_csv('DUSTY_FEATURES.csv')

full_df = pd.concat([clean_df, dusty_df], ignore_index=True)
X = full_df.drop(columns=['filename', 'label']).values
y = full_df['label'].values

print("Original dataset shape:", Counter(y))

# ==============================
# ✂️ Apply Tomek Links
# ==============================
tomek = TomekLinks(sampling_strategy='auto')
X_resampled, y_resampled = tomek.fit_resample(X, y)

print("After Tomek Links:", Counter(y_resampled))

# ✅ Save cleaned dataset after Tomek to use in SMOTE
cleaned_df = pd.DataFrame(X_resampled)
cleaned_df.insert(0, 'label', y_resampled)
cleaned_df.to_csv('CLEANED_DATASET_AFTER_TOMEK.csv', index=False)
print("✅ Saved cleaned dataset to CLEANED_DATASET_AFTER_TOMEK.csv")

# ==============================
# 🎨 Run t-SNE on original data
# ==============================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_original = tsne.fit_transform(X)

# ==============================
# 🎨 Run t-SNE on cleaned data
# ==============================
X_tsne_cleaned = tsne.fit_transform(X_resampled)

# ==============================
# 🖼️ Plot before vs after
# ==============================
plt.figure(figsize=(12,5))

# Before
plt.subplot(1, 2, 1)
plt.scatter(X_tsne_original[y==0,0], X_tsne_original[y==0,1], alpha=0.5, label='Clean', s=10)
plt.scatter(X_tsne_original[y==1,0], X_tsne_original[y==1,1], alpha=0.5, label='Dusty', s=10)
plt.title("Before Tomek Links")
plt.legend()

# After
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_cleaned[y_resampled==0,0], X_tsne_cleaned[y_resampled==0,1], alpha=0.5, label='Clean', s=10)
plt.scatter(X_tsne_cleaned[y_resampled==1,0], X_tsne_cleaned[y_resampled==1,1], alpha=0.5, label='Dusty', s=10)
plt.title("After Tomek Links")
plt.legend()

plt.suptitle("t-SNE Visualization of Feature Space")
plt.tight_layout()
plt.show()

#################
# APPLYING SMOTE
#################

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# ==============================
# 🔥 Load your cleaned dataset after Tomek Links
# ==============================
cleaned_df = pd.read_csv('CLEANED_DATASET_AFTER_TOMEK.csv')

X_cleaned = cleaned_df.drop(columns=['label']).values
y_cleaned = cleaned_df['label'].values

print("Shape before DeepSMOTE:", Counter(y_cleaned))

# ==============================
# 🚀 Apply DeepSMOTE on cleaned features
# ==============================
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_cleaned, y_cleaned)

print("Shape after DeepSMOTE:", Counter(y_balanced))

# ==============================
# 📝 Build final DataFrame for training
# ==============================
balanced_df = pd.DataFrame(X_balanced)
balanced_df.insert(0, 'label', y_balanced)

# Save to CSV for easy reuse
balanced_df.to_csv('FINAL_BALANCED_FEATURES.csv', index=False)
print("✅ Saved FINAL_BALANCED_FEATURES.csv with shape:", balanced_df.shape)

######################
# t-SNE VISUALIZATION
#####################

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

print("Balanced dataset shape:", Counter(y_balanced))

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_balanced = tsne.fit_transform(X_balanced)

# Plot
plt.figure(figsize=(6,5))
plt.scatter(X_tsne_balanced[y_balanced==0,0], X_tsne_balanced[y_balanced==0,1], 
            alpha=0.5, label='Clean', s=10)
plt.scatter(X_tsne_balanced[y_balanced==1,0], X_tsne_balanced[y_balanced==1,1], 
            alpha=0.5, label='Dusty', s=10)
plt.title("t-SNE Visualization After DeepSMOTE (Balanced Features)")
plt.legend()
plt.show()

############################
# MODEL PERFORMANCE 
# XGBOOST
############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
from collections import Counter

# ======================
# 🔥 Load data
# ======================
df = pd.read_csv('FINAL_BALANCED_FEATURES.csv')
X = df.drop(columns=['label']).values
y = df['label'].values

print("Dataset shape:", Counter(y))

# ======================
# 🚀 Prepare 10-fold stratified CV
# ======================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_fold_histories = []
all_fold_auc_data = []
all_y_true = []
all_y_pred = []

fold = 1

for train_index, val_index in skf.split(X, y):
    print(f"\n🔥 Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_lambda=2,
        random_state=42,
        eval_metric=["logloss"]
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    evals_result = model.evals_result()
    all_fold_histories.append(evals_result)

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix (Fold {fold}):")
    print(cm)

    print(f"\nClassification Report (Fold {fold}):")
    print(classification_report(y_val, y_pred, digits=4))

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(evals_result['validation_0']['logloss'], label='Train Logloss', linestyle='--')
    plt.plot(evals_result['validation_1']['logloss'], label='Val Logloss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(1-np.array(evals_result['validation_0']['logloss']), label='Train Accuracy', linestyle='--')
    plt.plot(1-np.array(evals_result['validation_1']['logloss']), label='Val Accuracy')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Pseudo Accuracy')
    plt.title(f'Pseudo Accuracy Curve - Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    all_fold_auc_data.append((fpr, tpr, roc_auc))

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.show()

    fold +=1

plt.figure(figsize=(12,5))
for i, hist in enumerate(all_fold_histories, 1):
    plt.subplot(1,2,1)
    plt.plot(hist['validation_1']['logloss'], label=f'Val Fold {i}', alpha=0.7)
    plt.subplot(1,2,2)
    plt.plot(1-np.array(hist['validation_1']['logloss']), label=f'Val Fold {i}', alpha=0.7)
plt.subplot(1,2,1)
plt.title('Combined Loss Curves (Validation Logloss)')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.legend()
plt.subplot(1,2,2)
plt.title('Combined Pseudo Accuracy Curves')
plt.xlabel('Boosting Rounds')
plt.ylabel('Pseudo Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_auc_data, 1):
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves across 10 Folds')
plt.legend()
plt.show()

print("\n🔥 Final Combined Classification Report across all folds:")
print(classification_report(all_y_true, all_y_pred, digits=4))

final_cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Combined Confusion Matrix')
plt.show()

# RUN THE SAME CODE FOR ALL EXPERIMENTS

######################
# TABNET
######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report, confusion_matrix, roc_curve, auc
from pytorch_tabnet.tab_model import TabNetClassifier
from collections import Counter
import torch

# ======================
# 🔥 Load data
# ======================
df = pd.read_csv('FINAL_BALANCED_FEATURES.csv')
X = df.drop(columns=['label']).values.astype(np.float32)
y = df['label'].values.astype(np.int64)  # important for TabNet

print("Dataset shape:", Counter(y))

# ======================
# 🚀 Prepare 10-fold stratified CV
# ======================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_fold_histories = []
all_fold_auc_data = []
all_y_true = []
all_y_pred = []
metrics_per_fold = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fold = 1

for train_index, val_index in skf.split(X, y):
    print(f"\n🔥 Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # ======================
    # ⚙️ TabNet Classifier
    # ======================
    model = TabNetClassifier(
        n_d=64, n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
        seed=42
    )

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["logloss"],
        max_epochs=50,
        patience=10,
        batch_size=256,
        virtual_batch_size=128
    )

    history = model.history
    train_logloss = history['loss']
    val_logloss = history['val_logloss']
    evals_result = {
        'validation_0': {'logloss': train_logloss},
        'validation_1': {'logloss': val_logloss}
    }
    all_fold_histories.append(evals_result)

    y_proba = model.predict_proba(X_val)[:,1]
    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix (Fold {fold}):\n{cm}")
    print(f"\nClassification Report (Fold {fold}):\n{classification_report(y_val, y_pred, digits=4)}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.show()

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    metrics_per_fold.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Log Loss": loss
    })

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_logloss, label='Train Logloss', linestyle='--')
    plt.plot(val_logloss, label='Val Logloss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(1-np.array(train_logloss), label='Train Accuracy', linestyle='--')
    plt.plot(1-np.array(val_logloss), label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pseudo Accuracy')
    plt.title(f'Pseudo Accuracy Curve - Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    all_fold_auc_data.append((fpr, tpr, roc_auc))

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.show()

    fold += 1

# ======================
# 🚀 Combined plots & summary
# ======================
plt.figure(figsize=(12,5))
for i, hist in enumerate(all_fold_histories, 1):
    plt.subplot(1,2,1)
    plt.plot(hist['validation_1']['logloss'], label=f'Val Fold {i}', alpha=0.7)
    plt.subplot(1,2,2)
    plt.plot(1-np.array(hist['validation_1']['logloss']), label=f'Val Fold {i}', alpha=0.7)
plt.subplot(1,2,1)
plt.title('Combined Loss Curves (Validation Logloss)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.subplot(1,2,2)
plt.title('Combined Pseudo Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Pseudo Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_auc_data, 1):
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves across 10 Folds')
plt.legend()
plt.show()

print("\n🔥 Final Combined Classification Report across all folds:")
print(classification_report(all_y_true, all_y_pred, digits=4))

final_cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Combined Confusion Matrix')
plt.show()

summary_df = pd.DataFrame(metrics_per_fold)
summary_df.loc['Mean'] = summary_df.mean(numeric_only=True)
summary_df.loc['Std'] = summary_df.std(numeric_only=True)

print("\n📊 Cross-Fold Metrics Summary Table:")
print(summary_df.round(4))

###################################
# ViT-HEAD
###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report, confusion_matrix, roc_curve, auc
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 🔥 Load data
# ======================
df = pd.read_csv('FINAL_BALANCED_FEATURES.csv')
X = df.drop(columns=['label']).values.astype(np.float32)
y = df['label'].values.astype(np.float32)

print("Dataset shape:", Counter(y))

# ======================
# 🚀 Prepare 10-fold stratified CV
# ======================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

all_fold_histories = []
all_fold_auc_data = []
all_y_true = []
all_y_pred = []
metrics_per_fold = []

# ======================
# ⚡ Improved ViT-Head Model
# ======================
class ImprovedViTHead(nn.Module):
    def __init__(self, input_dim, num_heads=8, d_model=64, num_layers=4):
        super(ImprovedViTHead, self).__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm([input_dim, d_model])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.embed(x)
        x = self.norm(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,0,2)
        return self.fc(x)

fold = 1

for train_index, val_index in skf.split(X, y):
    print(f"\n🔥 Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ImprovedViTHead(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    history = {"train_loss":[], "val_loss":[]}
    for epoch in range(50):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        history["train_loss"].append(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        history["val_loss"].append(np.mean(val_losses))

    all_fold_histories.append({
        "validation_0": {"logloss": history["train_loss"]},
        "validation_1": {"logloss": history["val_loss"]}
    })

    model.eval()
    with torch.no_grad():
        y_proba = model(torch.tensor(X_val).to(device)).cpu().numpy().flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # ✅ Check balanced predictions
    print(f"Predicted counts: 0 = {np.sum(y_pred==0)}, 1 = {np.sum(y_pred==1)}")

    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix (Fold {fold}):\n{cm}")
    print(f"\nClassification Report (Fold {fold}):\n{classification_report(y_val, y_pred, digits=4)}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.show()

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    metrics_per_fold.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Log Loss": loss
    })

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label='Train Logloss', linestyle='--')
    plt.plot(history["val_loss"], label='Val Logloss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(1-np.array(history["train_loss"]), label='Train Accuracy', linestyle='--')
    plt.plot(1-np.array(history["val_loss"]), label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pseudo Accuracy')
    plt.title(f'Pseudo Accuracy Curve - Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    all_fold_auc_data.append((fpr, tpr, roc_auc))

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.show()

    fold += 1

# ======================
# 🚀 Combined plots & summary
# ======================
plt.figure(figsize=(12,5))
for i, hist in enumerate(all_fold_histories, 1):
    plt.subplot(1,2,1)
    plt.plot(hist['validation_1']['logloss'], label=f'Val Fold {i}', alpha=0.7)
    plt.subplot(1,2,2)
    plt.plot(1-np.array(hist['validation_1']['logloss']), label=f'Val Fold {i}', alpha=0.7)
plt.subplot(1,2,1)
plt.title('Combined Loss Curves (Validation Logloss)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.subplot(1,2,2)
plt.title('Combined Pseudo Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Pseudo Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_auc_data, 1):
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves across 10 Folds')
plt.legend()
plt.show()

print("\n🔥 Final Combined Classification Report across all folds:")
print(classification_report(all_y_true, all_y_pred, digits=4))

final_cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Combined Confusion Matrix')
plt.show()

summary_df = pd.DataFrame(metrics_per_fold)
summary_df.loc['Mean'] = summary_df.mean(numeric_only=True)
summary_df.loc['Std'] = summary_df.std(numeric_only=True)

print("\n📊 Cross-Fold Metrics Summary Table:")
print(summary_df.round(4))

#########################
# ViT-HEAD FINE TUNNED
#########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report, confusion_matrix, roc_curve, auc
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 🔥 Load data
# ======================
df = pd.read_csv('FINAL_BALANCED_FEATURES.csv')
X = df.drop(columns=['label']).values.astype(np.float32)
y = df['label'].values.astype(np.float32)

print("Dataset shape:", Counter(y))

# ======================
# 🚀 Prepare 10-fold stratified CV
# ======================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

all_fold_histories = []
all_fold_auc_data = []
all_y_true = []
all_y_pred = []
metrics_per_fold = []

# ======================
# ⚡ Improved ViT-Head with clipping, scheduler
# ======================
class BetterViTHead(nn.Module):
    def __init__(self, input_dim, num_heads=8, d_model=64, num_layers=4):
        super(BetterViTHead, self).__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm([input_dim, d_model])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.embed(x)
        x = self.norm(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,0,2)
        return self.fc(x)

fold = 1

for train_index, val_index in skf.split(X, y):
    print(f"\n🔥 Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = BetterViTHead(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    history = {"train_loss":[], "val_loss":[]}
    for epoch in range(100):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

    print(f"✅ Min Val Loss: {min(history['val_loss']):.4f} | Final Val Loss: {history['val_loss'][-1]:.4f}")
    all_fold_histories.append({
        "validation_0": {"logloss": history["train_loss"]},
        "validation_1": {"logloss": history["val_loss"]}
    })

    model.eval()
    with torch.no_grad():
        y_proba = model(torch.tensor(X_val).to(device)).cpu().numpy().flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    print(f"Predicted counts: 0 = {np.sum(y_pred==0)}, 1 = {np.sum(y_pred==1)}")
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix (Fold {fold}):\n{cm}")
    print(f"\nClassification Report (Fold {fold}):\n{classification_report(y_val, y_pred, digits=4)}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.show()

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    metrics_per_fold.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Log Loss": loss
    })

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label='Train Logloss', linestyle='--')
    plt.plot(history["val_loss"], label='Val Logloss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(1-np.array(history["train_loss"]), label='Train Accuracy', linestyle='--')
    plt.plot(1-np.array(history["val_loss"]), label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pseudo Accuracy')
    plt.title(f'Pseudo Accuracy Curve - Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    all_fold_auc_data.append((fpr, tpr, roc_auc))

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.show()

    fold += 1

# ======================
# 🚀 Combined plots & summary
# ======================
plt.figure(figsize=(12,5))
for i, hist in enumerate(all_fold_histories, 1):
    plt.subplot(1,2,1)
    plt.plot(hist['validation_1']['logloss'], label=f'Val Fold {i}', alpha=0.7)
    plt.subplot(1,2,2)
    plt.plot(1-np.array(hist['validation_1']['logloss']), label=f'Val Fold {i}', alpha=0.7)
plt.subplot(1,2,1)
plt.title('Combined Loss Curves (Validation Logloss)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.subplot(1,2,2)
plt.title('Combined Pseudo Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Pseudo Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_auc_data, 1):
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves across 10 Folds')
plt.legend()
plt.show()

print("\n🔥 Final Combined Classification Report across all folds:")
print(classification_report(all_y_true, all_y_pred, digits=4))

final_cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Combined Confusion Matrix')
plt.show()

summary_df = pd.DataFrame(metrics_per_fold)
summary_df.loc['Mean'] = summary_df.mean(numeric_only=True)
summary_df.loc['Std'] = summary_df.std(numeric_only=True)

print("\n📊 Cross-Fold Metrics Summary Table:")
print(summary_df.round(4))

###################################
#  
###################################
