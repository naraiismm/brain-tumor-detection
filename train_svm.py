import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import joblib

# ---- SETTINGS ----
IMG_SIZE = (224, 224)   # Input image size
BATCH_SIZE = 32          # Number of images per batch

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"   # Primary dataset
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"  # Secondary dataset

# ---- DATA PREPARATION ----
# No augmentation for SVM - only normalization
datagen = ImageDataGenerator(rescale=1./255)

# Training data - shuffle=False to keep labels aligned with features
train_data = datagen.flow_from_directory(
    DATASET1 + r"\Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Dataset 1 - Test set
test_data1 = datagen.flow_from_directory(
    DATASET1 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Dataset 2 - External test set (different source)
test_data2 = datagen.flow_from_directory(
    DATASET2 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("✅ Data loaded successfully!")

# ---- FEATURE EXTRACTION WITH MobileNetV2 ----
# Use MobileNetV2 as a feature extractor (without classification layer)
print("\n🔍 Extracting features...")
base_model = MobileNetV2(
    weights='imagenet',    # Pre-trained ImageNet weights
    include_top=False,     # Remove classification layer
    pooling='avg',         # Global average pooling
    input_shape=(224, 224, 3)
)

def extract_features(data):
    # Extract deep features from images using MobileNetV2
    features = base_model.predict(data, verbose=1)
    labels = data.classes  # Get true class labels
    return features, labels

# Extract features from all datasets
X_train, y_train = extract_features(train_data)
X_test1, y_test1 = extract_features(test_data1)
X_test2, y_test2 = extract_features(test_data2)

print("✅ Features extracted successfully!")

# ---- TRAIN SVM CLASSIFIER ----
# RBF kernel SVM works well for high-dimensional feature spaces
print("\n🚀 Training SVM...")
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)  # Train on extracted features

# ---- EVALUATE RESULTS ----
acc1 = accuracy_score(y_test1, svm.predict(X_test1))  # Dataset 1 accuracy
acc2 = accuracy_score(y_test2, svm.predict(X_test2))  # Dataset 2 accuracy

print(f"\n📊 SVM Dataset 1 Accuracy: {acc1*100:.2f}%")
print(f"📊 SVM Dataset 2 Accuracy: {acc2*100:.2f}%")

# ---- SAVE MODEL ----
joblib.dump(svm, r'models\svm_model.pkl')  # Save trained SVM model
print("✅ SVM model saved!")