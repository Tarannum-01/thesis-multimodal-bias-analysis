#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

import mediapipe as mp; print('mediapipe imported successfully')
from transformers import pipeline; print('transformers imported successfully')
from datasets import load_dataset; print('datasets imported successfully')
import torchattacks; print('torchattacks imported successfully')
import pandas as pd; print('pandas imported successfully')
import PIL; print('PIL imported successfully')


# In[2]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torchattacks
from facenet_pytorch import MTCNN


# In[3]:


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} on {torch.cuda.get_device_name(0)}")


# In[4]:


import pandas as pd
import os
root_dir = '/home/h703276408/projects/bias_llms/UTKFace/utkface_aligned_cropped/UTKFace'
files = os.listdir(root_dir)
print(f"Total files: {len(files)}")
print(f"Sample files: {files[:10]}")  # First 10 files
valid_files = [f for f in files if f.endswith('.chip.jpg') and len(f.replace('.chip.jpg', '').split('_')) == 4]
print(f"Valid files count: {len(valid_files)}")
print(f"Valid samples: {valid_files[:10]}")  # First 10 valid files
df = pd.DataFrame({
    'image': valid_files,
    'full_path': [os.path.join(root_dir, f) for f in valid_files]
})
# Split and remove .chip.jpg
df[['age', 'gender', 'ethnicity', 'datetime']] = df['image'].str.replace('.chip.jpg', '').str.split('_', expand=True)
df['datetime'] = df['datetime'].astype(str)  # Ensure string type
df['age'] = df['age'].astype(int)
df['gender'] = df['gender'].astype(int)
df['ethnicity'] = df['ethnicity'].astype(int)
ethnicity_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
print(f"Loaded {len(df)} valid images")
print(f"Excluded {len(files) - len(valid_files)} invalid files")


# In[17]:


import os
import pandas as pd
import re

# Step 4: Load and Preprocess the UTKFace Dataset
root_dir = '/home/h703276408/projects/bias_llms/UTKFace/utkface_aligned_cropped/UTKFace'  # Exact subdirectory

# Create a DataFrame from the file names with validation
files = os.listdir(root_dir)
print(f"Total files: {len(files)}")
print(f"Sample files: {files[:10]}")  # First 10 files for debug

valid_files = [f for f in files if f.endswith('.chip.jpg') and len(f.replace('.chip.jpg', '').split('_')) == 4]
print(f"Valid files count: {len(valid_files)}")
print(f"Valid samples: {valid_files[:10]}")  # First 10 valid files

df = pd.DataFrame({
    'image': valid_files,
    'full_path': [os.path.join(root_dir, f) for f in valid_files]
})

# Ensure all entries are strings
df['image'] = df['image'].astype(str)

# Parse labels with error handling
try:
    df[['age', 'gender', 'ethnicity', 'datetime']] = df['image'].str.replace('.chip.jpg', '').str.split('_', expand=True)
    # Clean datetime by removing any residual .jpg
    df['datetime'] = df['datetime'].str.replace('.jpg', '')
except AttributeError as e:
    print(f"❌ Parsing error: {e}. Check 'image' column data.")
    raise

# Convert columns to appropriate types
df['datetime'] = df['datetime'].astype(str)
df['age'] = pd.to_numeric(df['age'], errors='coerce').astype(int)
df['gender'] = pd.to_numeric(df['gender'], errors='coerce').astype(int)
df['ethnicity'] = pd.to_numeric(df['ethnicity'], errors='coerce').astype(int)

# Drop rows with invalid numeric conversions
df = df.dropna(subset=['age', 'gender', 'ethnicity'])

# Ethnicity map
ethnicity_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}

print(f"Loaded {len(df)} valid images")
print(f"Excluded {len(files) - len(valid_files)} invalid files")
print(f"Sample rows:\n{df.head()}")


# In[18]:


# Step 5: Optional Face Detection and Cropping Validation (Using MTCNN)
import os
import torch  # Added for torch.device
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image

# Define device locally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use a writable directory under your home directory
cropped_dir = '/home/h703276408/projects/bias_llms/UTKFace_cropped'  # User-writable path
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)
    print(f"✅ Created directory: {cropped_dir}")

# Initialize MTCNN for validation
mtcnn = MTCNN(
    image_size=224,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)

# Validate cropping on a random sample (e.g., 100 images)
detection_count = 0
sample_size = 100
for idx, row in df.sample(n=sample_size).iterrows():
    img_path = row['full_path']
    try:
        image = Image.open(img_path).convert('RGB')
        face = mtcnn(image)
        if face is not None:
            detection_count += 1
            # Optional: Save re-cropped image for comparison
            cropped_path = os.path.join(cropped_dir, row['image'])
            face_pil = transforms.ToPILImage()(face)
            face_pil.save(cropped_path)
            print(f"✅ Valid face detected in {row['image']}")
        else:
            print(f"⚠️ No face detected in {row['image']}")
    except Exception as e:
        print(f"❌ Error validating {row['image']}: {e}")

print(f"✅ Validation completed: {detection_count}/{sample_size} samples contain valid faces ({detection_count/sample_size*100:.1f}%)")
print(f"✅ Validation output saved to {cropped_dir}")


# In[19]:


# Step 6: Create Dataset and DataLoader
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class
class UTKFaceDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, train=True):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df['image'].iloc[idx])
        ethnicity = self.df['ethnicity'].iloc[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = img.resize((224, 224))
            if self.transform:
                input_tensor = self.transform(input_tensor)
            else:
                input_tensor = transforms.ToTensor()(input_tensor)
            return input_tensor, torch.tensor(ethnicity, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(0, dtype=torch.long)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create datasets with stratified split
root_dir = '/home/h703276408/projects/bias_llms/UTKFace/utkface_aligned_cropped/UTKFace'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ethnicity'])
train_dataset = UTKFaceDataset(train_df, root_dir, transform=train_transform, train=True)
val_dataset = UTKFaceDataset(val_df, root_dir, transform=val_transform, train=False)

# Create DataLoaders with reduced batch size and pin_memory
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

print(f"✅ Training set: {len(train_dataset)} samples")
print(f"✅ Validation set: {len(val_dataset)} samples")
print("\nTesting DataLoader...")
try:
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"✅ Batch shape: {images.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Labels range: {labels.min().item()} to {labels.max().item()}")
except Exception as e:
    print(f"❌ DataLoader error: {e}")


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Assume df is available from Step 4
# Select 9 random samples for a 3x3 grid
sample_indices = np.random.choice(df.index, size=9, replace=False)
sample_df = df.iloc[sample_indices]

# Create a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for idx, ax in enumerate(axes):
    img_path = sample_df.iloc[idx]['full_path']
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        age = sample_df.iloc[idx]['age']
        gender = 'Male' if sample_df.iloc[idx]['gender'] == 0 else 'Female'
        ethnicity = ethnicity_map[sample_df.iloc[idx]['ethnicity']]
        ax.set_title(f"Age: {age}\nGender: {gender}\nEthnicity: {ethnicity}")
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
plt.show()
print("✅ Sample images displayed from UTKFace dataset.")


# In[21]:


# Step 7: Load Model
import torch  
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ResNet50 with pre-trained weights
model = models.resnet50(weights='IMAGENET1K_V1').to(device)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 ethnicity classes: White, Black, Asian, Indian, Other

# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Print model info
print(f"✅ Model loaded: {model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"✅ Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)")
print(f"✅ Loss: CrossEntropyLoss")
print(f"✅ Scheduler: StepLR (step_size=10, gamma=0.1)")


# In[22]:


# Step 8: Training and Validation Functions
import torch.nn.functional as F

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"Starting epoch training...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)  
        labels = labels.to(device)  
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if batch_idx % 50 == 49:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/50:.3f}, Acc: {100 * correct / total:.1f}%")
            running_loss = 0.0
            correct = 0
            total = 0
    epoch_acc = 100 * correct / len(train_loader.dataset)
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)  
            labels = labels.to(device)  
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(val_loader)
    return epoch_loss, epoch_acc


# In[26]:


# Step 9: Execute Training and Validation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model = model.to(device)

# Training parameters
num_epochs = 3
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print(f"\n🚀 Starting training for {num_epochs} epochs on GPU...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: 16")
print(f"Classes: {list(ethnicity_map.values())}")

# Training loop with explicit GPU reassignment
for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step()
    print(f"  Training Loss: {train_loss:.3f}, Accuracy: {train_acc:.1f}%")
    print(f"  Validation Loss: {val_loss:.3f}, Accuracy: {val_acc:.1f}%")
    print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save checkpoint
    checkpoint_path = f'/home/h703276408/projects/bias_llms/utkface_checkpoint_epoch_{epoch+1}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'val_accuracy': val_acc,
        'ethnicity_map': ethnicity_map
    }, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")

print(f"\n✅ Training completed successfully!")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-o', label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, 'r-o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final evaluation
final_correct = 0
final_total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()
final_acc = 100 * final_correct / final_total
print(f"\n🎯 FINAL VALIDATION ACCURACY: {final_acc:.2f}% ({final_correct}/{final_total})")

# Save final model
model_save_path = '/home/h703276408/projects/bias_llms/utkface_ethnicity_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'val_accuracy': final_acc,
    'ethnicity_map': ethnicity_map
}, model_save_path)
print(f"✅ Model saved to {model_save_path}")

# Training summary
print("\n=== TRAINING SUMMARY ===")
print(f"Dataset: {len(df)} images")
print(f"Training split: {len(train_dataset)} images")
print(f"Validation split: {len(val_dataset)} images")
print(f"Model: ResNet50 (fine-tuned for ethnicity)")
print(f"Classes: {list(ethnicity_map.values())}")
print(f"Final validation accuracy: {final_acc:.2f}%")
print(f"Training completed in {num_epochs} epochs")


# In[27]:


# Step 10: Evaluate and Visualize Results (Enhanced)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("🎯 Starting model evaluation and visualization...")

# Ensure model is on correct device and in evaluation mode
model = model.to(device)
model.eval()

# Compute predictions and true labels
all_preds = []
all_labels = []
all_confidences = []

print("📊 Computing predictions on validation set...")
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
        
        # Progress indicator
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

print(f"✅ Processed {len(all_preds)} validation samples")

# Calculate overall accuracy
accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"📈 Overall Validation Accuracy: {accuracy:.2f}%")
print(f"📈 Average Confidence: {np.mean(all_confidences):.3f}")

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n📋 Confusion Matrix:")
print(cm)

# Plot confusion matrix with better formatting
plt.figure(figsize=(10, 8))
class_names = list(ethnicity_map.values())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix for Ethnicity Classification', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print detailed classification report
print("\n📊 DETAILED CLASSIFICATION REPORT:")
print("=" * 60)
report = classification_report(all_labels, all_preds, 
                             target_names=class_names, 
                             digits=3)
print(report)

# Calculate per-class accuracy
print("\n📈 PER-CLASS ACCURACY:")
print("-" * 40)
for i, class_name in enumerate(class_names):
    class_mask = np.array(all_labels) == i
    if np.sum(class_mask) > 0:
        class_accuracy = 100 * np.sum((np.array(all_preds)[class_mask] == i)) / np.sum(class_mask)
        class_count = np.sum(class_mask)
        print(f"{class_name:10}: {class_accuracy:6.2f}% ({class_count:4d} samples)")

# Visualize sample predictions with confidence scores
print("\n🖼️  Visualizing sample predictions...")
sample_indices = np.random.choice(len(val_dataset), 9, replace=False)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for idx, ax in enumerate(axes):
    sample_idx = sample_indices[idx]
    image, label = val_dataset[sample_idx]
    
    # Get prediction with confidence
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    
    # Convert image for display (handle normalization)
    img_display = image.permute(1, 2, 0).numpy()
    
    # Denormalize if needed (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    true_label = ethnicity_map[label.item()]
    pred_label = ethnicity_map[pred.item()]
    conf_score = confidence.item()
    
    # Color code: green for correct, red for incorrect
    title_color = 'green' if true_label == pred_label else 'red'
    
    ax.imshow(img_display)
    ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf_score:.3f}", 
                color=title_color, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample Predictions with Confidence Scores\n(Green=Correct, Red=Incorrect)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Additional analysis: confidence distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(all_confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidences')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
correct_mask = np.array(all_preds) == np.array(all_labels)
correct_conf = np.array(all_confidences)[correct_mask]
incorrect_conf = np.array(all_confidences)[~correct_mask]

plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution: Correct vs Incorrect')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Evaluation and visualization completed successfully!")
print(f"📊 Final Summary:")
print(f"   • Total samples evaluated: {len(all_preds)}")
print(f"   • Overall accuracy: {accuracy:.2f}%")
print(f"   • Average confidence: {np.mean(all_confidences):.3f}")
print(f"   • High confidence predictions (>0.8): {np.sum(np.array(all_confidences) > 0.8)}/{len(all_confidences)} ({100*np.sum(np.array(all_confidences) > 0.8)/len(all_confidences):.1f}%)")


# In[29]:


import torch

# 1. Detect and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Move model to the selected device
model = model.to(device)

# 3. Example inference loop on validation loader
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        # Move a batch of data to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        print(f"Batch predictions: {preds.cpu().numpy()}")
        print(f"True labels:      {labels.cpu().numpy()}")
        break  # Remove this break to run on the full dataset


# In[33]:


# Step 11: Adversarial Attacks on Validation Set (Updated)

import torch
import torch.nn as nn
import torchattacks
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

# 1. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load model checkpoint
checkpoint = torch.load(
    '/home/h703276408/projects/bias_llms/utkface_ethnicity_model.pth',
    map_location=device
)
# Use weights=None to avoid downloading ImageNet weights
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("✅ Model loaded and set to eval mode")

# 3. (Optional) Compute clean accuracy for comparison
def eval_clean(model, loader):
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print(f"Clean Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

clean_acc = eval_clean(model, val_loader)

# 4. Define attacks
fgsm = torchattacks.FGSM(model, eps=0.1)
pgd  = torchattacks.PGD(model, eps=0.1, alpha=0.01, steps=10)

# 5. Adversarial evaluation function
def eval_attack(model, loader, attack, name):
    correct = total = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Attack generation must be outside no_grad
        adv_images = attack(images, labels)
        with torch.no_grad():
            outputs = model(adv_images)
            _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()
    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}% ({correct}/{total})")
    return all_preds, all_labels

# 6. Evaluate FGSM and PGD
print("\n=== Adversarial Attack Results ===")
fgsm_preds, fgsm_labels = eval_attack(model, val_loader, fgsm, "FGSM")
pgd_preds, pgd_labels   = eval_attack(model, val_loader, pgd,  "PGD")

# 7. Plot confusion matrices
def plot_cm(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=list(ethnicity_map.values()),
        yticklabels=list(ethnicity_map.values())
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_cm(fgsm_labels, fgsm_preds, f"FGSM Attack Confusion Matrix (Acc {fgsm_preds and 100*sum(p==t for p,t in zip(fgsm_preds, fgsm_labels))/len(fgsm_labels):.2f}%)")
plot_cm(pgd_labels, pgd_preds,   f"PGD Attack Confusion Matrix  (Acc {pgd_preds and 100*sum(p==t for p,t in zip(pgd_preds, pgd_labels))/len(pgd_labels):.2f}%)")


# In[38]:


# Visualize Adversarial Examples and Model Predictions

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# 1. Helper to unnormalize and convert tensor to image
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def tensor_to_image(tensor):
    img = inv_normalize(tensor.cpu())
    img = torch.clamp(img, 0, 1)
    return np.transpose(img.numpy(), (1,2,0))

# 2. Collect a batch of clean and adversarial images with labels
model.eval()
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)

# Generate FGSM adversarial examples
adv_images = fgsm(images, labels)

# 3. Get predictions
with torch.no_grad():
    clean_outputs = model(images)
    adv_outputs   = model(adv_images)
_, clean_preds = clean_outputs.max(1)
_, adv_preds   = adv_outputs.max(1)

# 4. Plot side-by-side comparisons
num_display = 6  # number of examples to show
plt.figure(figsize=(12, 6))
for i in range(num_display):
    # Clean image
    plt.subplot(2, num_display, i+1)
    plt.imshow(tensor_to_image(images[i]))
    plt.title(f"Clean: {ethnicity_map[clean_preds[i].item()]}")
    plt.axis('off')
    # Adversarial image
    plt.subplot(2, num_display, num_display + i+1)
    plt.imshow(tensor_to_image(adv_images[i]))
    plt.title(f"Adv:   {ethnicity_map[adv_preds[i].item()]}")
    plt.axis('off')
plt.suptitle("Clean vs. FGSM Adversarial Examples and Predictions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[ ]:




