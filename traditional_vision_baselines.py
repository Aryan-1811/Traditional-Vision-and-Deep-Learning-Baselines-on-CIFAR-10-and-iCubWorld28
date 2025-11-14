import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import tarfile
# Extract the archive
with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
    tar.extractall()

# %%
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        # Reshape to (num_images, 3, 32, 32) then transpose to (H, W, C)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels

# Load and extract CIFAR-10 dataset (first batch only)
# Images will be used for traditional CV (SIFT + BoW)
images, labels = load_cifar_batch("cifar-10-batches-py/data_batch_1")
print("Shape:", images.shape)  # (10000, 32, 32, 3)

# %%
sift_images = []
sift_labels = []

for i in range(10000):  # Use a smaller subset for traditional CV
    img = images[i]
    label = labels[i]

    # Resize each image to 160x160 and convert to grayscale
    img_resized = cv2.resize(img, (160, 160))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) # These larger grayscale images are better suited for SIFT feature extraction

    sift_images.append(img_gray)
    sift_labels.append(label)

# %%
sift = cv2.SIFT_create()
descriptor_list = []
valid_labels = []

# Detect keypoints and compute descriptors using SIFT
for img, label in zip(sift_images, sift_labels):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None: # Store descriptors only for images where detection was successful
        descriptor_list.append(descriptors)
        valid_labels.append(label)

print("Images with valid descriptors:", len(descriptor_list))

# %%
# Stack all descriptors into one large array
all_descriptors = np.vstack(descriptor_list)
print("Total descriptors:", all_descriptors.shape)

# Run KMeans clustering to create visual vocabulary
k = 200  # Number of visual words 
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(all_descriptors)

# Save visual words (cluster centers)
vocab = kmeans.cluster_centers_

# %%
def compute_bow_histogram(descriptors, kmeans_model, k):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k)

    # Assign each descriptor to the closest cluster center
    word_assignments = pairwise_distances_argmin(descriptors, kmeans_model.cluster_centers_)
    
    # Build normalized histogram
    hist, _ = np.histogram(word_assignments, bins=np.arange(k + 1), density=True)
    # Add small constant to avoid zeros
    hist += 1e-6
    hist = hist / np.sum(hist)
    return hist

# Create BoW vectors
X = []
y = []

for descriptors, label in zip(descriptor_list, valid_labels): 
    hist = compute_bow_histogram(descriptors, kmeans, k)
    X.append(hist)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("BoW dataset shape:", X.shape) 

# %%
# Apply TF-IDF to reweight the BoW histograms
# This reduces the influence of common words and highlights discriminative features
tfidf = TfidfTransformer(norm='l2')  # Use L2-normalization
X_tfidf = tfidf.fit_transform(X).toarray()

# %%
# Train-test split (e.g., 80-20)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01],
    'kernel': ['rbf', 'linear']
}

# Use grid search to tune SVM hyperparameters: C, gamma, and kernel type
grid = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# Best model
print("Best Parameters:", grid.best_params_)

# Train SVM and evaluate on the test set
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Tuned SVM Accuracy:", acc)

# %%
# Classification report
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SIFT + BoW + SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# Draw SIFT keypoints on one of the processed grayscale images
img = cv2.drawKeypoints(sift_images[9999], keypoints, None)
plt.imshow(img, cmap='gray')
plt.title("SIFT Keypoints")

# %%
# Define transformations: flipping, cropping, normalization
transform_cnn = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 mean/std
])

# Load train and test datasets
train_dataset_cnn = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cnn)
test_dataset_cnn = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cnn)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64)

# %%
# Basic CNN with two convolutional layers, pooling, and two fully connected layers
# Suitable for small datasets like CIFAR-10
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 → 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CIFAR10CNN()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Train the custom CNN for 10 epochs using Adam optimizer
    model.train()
    running_loss = 0.0
    for images, labels in train_loader_cnn:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader_cnn):.4f}")

# %%
model.eval()
all_preds = []
all_labels = []

# Evaluate the trained CNN on the test set
with torch.no_grad():
    for images, labels in test_loader_cnn:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc_cnn = accuracy_score(all_labels, all_preds)
print("CNN Accuracy:", acc_cnn)

# %%
# Resize images to 224x224 and normalize for ResNet
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset again with these transformations
train_dataset_rn18 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_rn18 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader_rn18 = DataLoader(train_dataset_rn18, batch_size=64, shuffle=True)
test_loader_rn18 = DataLoader(test_dataset_rn18, batch_size=64)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = models.resnet18(pretrained=True)

# Replace the final layer to match CIFAR-10's 10 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Use learning rate scheduler to decay LR after 10 epochs

# %%
# Train for 20 epochs using SGD with momentum
for epoch in range(20):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader_rn18, desc=f"Epoch {epoch+1}", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=avg_loss)

    scheduler.step()
    print(f"Epoch {epoch+1} finished. Avg Loss: {running_loss/len(train_loader_rn18):.4f}")

# %%
model.eval()
all_preds = []
all_labels = []

# Run inference on test set
with torch.no_grad():
    for images, labels in test_loader_rn18:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print("ResNet18 (Transfer Learning) Accuracy:", acc)

# %%
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
plt.title("Confusion Matrix - ResNet18 on CIFAR-10")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class-wise precision/recall/f1
print(classification_report(all_labels, all_preds, target_names=train_dataset_rn18.classes))





Code for iCubWorld28
# %%
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import shutil

dataset_root = r"\iCubWorld28\iCubWorld28_128x128\test"
days = ["day1", "day2"]

grayscale_images = []
image_labels = []
label_map = {}
label_id = 0

# Load images from iCubWorld28 (day1 and day2)
for day in days:
    day_path = os.path.join(dataset_root, day)
    for class_name in sorted(os.listdir(day_path)):
        class_path = os.path.join(day_path, class_name)
        if not os.path.isdir(class_path):
            continue

        if class_name not in label_map:
            label_map[class_name] = label_id
            label_id += 1

        for instance_name in os.listdir(class_path):
            instance_path = os.path.join(class_path, instance_name)
            if not os.path.isdir(instance_path):
                continue

            for img_name in os.listdir(instance_path):
                if not img_name.endswith(".ppm"):
                    continue
                img_path = os.path.join(instance_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert each RGB image to grayscale for traditional SIFT processing
                grayscale_images.append(gray)
                image_labels.append(label_map[class_name]) # Create label mapping for each object class

# %%
# Initialize SIFT detector
sift = cv2.SIFT_create()

# List to hold all descriptor arrays
descriptor_list = []
valid_labels = []  # Corresponding labels for images with descriptors

# Store descriptors and their corresponding labels
for img, label in zip(grayscale_images, image_labels):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None and len(descriptors) > 0:
        descriptor_list.append(descriptors)
        valid_labels.append(label)

# %%
print(f"Total images processed: {len(grayscale_images)}")
print(f"Images with SIFT descriptors: {len(descriptor_list)}")

# %%
# Stack all descriptors into one large array (N x 128)
all_descriptors = np.vstack(descriptor_list)
print("All descriptors shape:", all_descriptors.shape)

# Set vocabulary size 
k = 200  
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(all_descriptors)

# Visual vocabulary = cluster centers
vocab = kmeans.cluster_centers_

# %%
def compute_bow_histogram(descriptors, kmeans_model, k):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k)

    # Assign each descriptor to the nearest cluster center
    word_indices = pairwise_distances_argmin(descriptors, kmeans_model.cluster_centers_)

    # Build normalized histogram
    hist, _ = np.histogram(word_indices, bins=np.arange(k + 1), density=True)
    hist += 1e-6  # Smoothing to avoid zero bins
    hist /= np.sum(hist)  # Normalize

    return hist

# %%
X = []  # feature vectors (histograms)
y = []  # labels

for descriptors, label in zip(descriptor_list, valid_labels):
    hist = compute_bow_histogram(descriptors, kmeans, k)
    X.append(hist)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("BoW feature matrix shape:", X.shape)

# %%
# Apply TF-IDF to BoW histograms to reduce common word impact
tfidf = TfidfTransformer(norm='l2')  # Use L2-normalization
X_tfidf = tfidf.fit_transform(X).toarray()

# %%
# Use train-test split (stratified) and perform GridSearchCV on SVM
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01],
    'kernel': ['rbf', 'linear']
}

# Use 3-fold cross-validation
grid = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# Best model
print("Best Parameters:", grid.best_params_)

# Evaluate on test set
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Tuned SVM Accuracy:", acc)

# %%
# Classification report
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SIFT + BoW + SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# Display keypoints detected by SIFT on a sample grayscale image
img_gray = grayscale_images[42]
img = cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img, cmap='gray')
plt.title("SIFT Keypoints on iCubWorld28")
plt.axis('off')
plt.show()

# %%
# Split the original data into train/test folders for ImageFolder loading
# This ensures compatibility with PyTorch DataLoader
source_root = r"\iCubWorld28\iCubWorld28_128x128\test"
output_root = r"\iCubWorld28\iCubWorld28_128x128\split"

# Get class names only from valid directories in day1
classes = [
    name for name in os.listdir(os.path.join(source_root, "day1"))
    if os.path.isdir(os.path.join(source_root, "day1", name))
]

# Create train/test folders
for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# Loop through each class
for cls in classes:
    all_images = []

    # Look inside day1 and day2
    for day in ["day1", "day2"]:
        class_dir = os.path.join(source_root, day, cls)

        # Make sure it's a valid class folder
        if not os.path.isdir(class_dir):
            continue

        # Go through all instance folders
        for instance in os.listdir(class_dir):
            instance_path = os.path.join(class_dir, instance)
            if not os.path.isdir(instance_path):
                continue

            # Collect all .ppm image paths
            for img_name in os.listdir(instance_path):
                if img_name.endswith(".ppm"):
                    img_path = os.path.join(instance_path, img_name)
                    all_images.append(img_path)

    # Train/test split
    train_imgs, test_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

    # Copy images
    for img_path in train_imgs:
        dest_path = os.path.join(output_root, "train", cls, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

    for img_path in test_imgs:
        dest_path = os.path.join(output_root, "test", cls, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

print("Train/test split completed.")

# %%# Resize and normalize RGB images for training with CNN 
transform_icub = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
])

train_path = r"\iCubWorld28\iCubWorld28_128x128\split\train"
test_path = r"\iCubWorld28\iCubWorld28_128x128\split\test"

train_data = datasets.ImageFolder(root=train_path, transform=transform_icub)
test_data = datasets.ImageFolder(root=test_path, transform=transform_icub)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# %%
# Simple CNN with 2 convolutional layers + 2 fully connected layers
class ICubCNN(nn.Module):
    def __init__(self):
        super(ICubCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # RGB: 3 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, len(train_data.classes))  # Output: number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 → 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 → 32x32
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ICubCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10): # Train using Adam optimizer for 10 epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# %%
model.eval()
all_preds, all_labels = [], []

# Predict and evaluate CNN on test set
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print("CNN Accuracy on iCubWorld28:", acc)
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# %%
# Resize to 224x224 and normalize for ResNet18 input
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Use standard RGB normalization
])

train_data = datasets.ImageFolder(root=train_path, transform=transform_resnet)
test_data = datasets.ImageFolder(root=test_path, transform=transform_resnet)

# Load datasets again with new transforms
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer to match iCubWorld28 class count
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))  # Match number of iCub classes

model = model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# %%
for epoch in range(20):
    model.train() # Train with SGD optimizer and learning rate scheduler over 20 epochs
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=avg_loss)

    scheduler.step()
    print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss/len(train_loader):.4f}")

# %%
model.eval()
all_preds = []
all_labels = []

# Evaluate performance of fine-tuned ResNet18
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print("ResNet18 Accuracy on iCubWorld28:", acc)
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# %%
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
plt.title("Confusion Matrix - ResNet18 on CIFAR-10")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
