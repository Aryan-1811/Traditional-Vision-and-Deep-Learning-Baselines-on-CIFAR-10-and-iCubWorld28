# Traditional-Vision-and-Deep-Learning-Baselines-on-CIFAR-10-and-iCubWorld28

This project compares traditional computer vision methods (SIFT + Bag-of-Words + SVM) with deep learning models (Custom CNN and ResNet18) on CIFAR-10 and iCubWorld28. It highlights the performance gap between hand-engineered features and modern neural networks.

## Datasets

### CIFAR-10
- 10 classes, 32x32 RGB images  
- Used for both traditional CV and deep learning  
- Requires `cifar-10-python.tar.gz` placed in the repo root
- Download it from "https://www.cs.toronto.edu/~kriz/cifar.html"

### iCubWorld28
- Real-world robotic vision dataset with multiple object classes  
- Used for both SIFT + BoW + SVM and CNN/ResNet models  
- User must update dataset paths in the notebook
- Download it from "https://robotology.github.io/iCubWorld/#icubworld-28-modal"

## Methods

### 1. Traditional Vision: SIFT + Bag of Visual Words + SVM
- Convert images to grayscale  
- Extract SIFT descriptors  
- Construct a 200-word visual vocabulary using KMeans  
- Build histograms for each image and apply TF-IDF  
- Train SVM classifier using GridSearchCV  
- Evaluate accuracy, confusion matrix, classification report  

### 2. CNN Baseline
- Two convolutional layers with max pooling  
- Two fully connected layers  
- Trained for 10 epochs with Adam  
- Evaluated on CIFAR-10 and iCubWorld28

### 3. ResNet18 Transfer Learning
- Resize images to 224x224  
- Load pretrained ResNet18 and replace final FC layer  
- Train for 20 epochs using SGD with momentum and StepLR  
- Evaluate class-wise metrics and confusion matrix  

## Results Summary
- **ResNet18** gives the highest accuracy on both datasets  
- **CNN** outperforms traditional CV  
- **SIFT + BoW + SVM** is the strongest non-deep baseline  
- Metrics and confusion matrices are displayed in the notebook

## Repository Structure
```
project/
├── traditional_vision_baselines.py
└── README.md
```

## How to Run
1. Clone the repository.
2. Download CIFAR-10 (python version) and place `cifar-10-python.tar.gz` in the project root.
3. Download iCubWorld28 128x128 and update the following paths in the notebook:
   - `dataset_root`
   - `train_path`
   - `test_path`
4. Install dependencies:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn opencv-python torch torchvision tqdm
   ```
5. Run the notebook:
   - `traditional_vision_baselines.py`  
   Run all cells in order.

## Dependencies
- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- opencv-python  
- torch  
- torchvision  
- tqdm  
