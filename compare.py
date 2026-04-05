import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ---- SETTINGS ----
IMG_SIZE = (224, 224)   # Input image size
BATCH_SIZE = 32          # Number of images per batch
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Tumor classes

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"   # Primary dataset
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"  # Secondary dataset

# ---- DATA PREPARATION ----
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

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

# ---- LOAD TRAINED MODELS ----
print("📦 Loading models...")
mobilenet = load_model(r'models\mobilenet_model.keras')  # Load MobileNetV2
cnn = load_model(r'models\cnn_model.keras')              # Load CNN
svm = joblib.load(r'models\svm_model.pkl')               # Load SVM

# Feature extractor for SVM (MobileNetV2 without classification layer)
base = MobileNetV2(weights='imagenet', include_top=False,
                   pooling='avg', input_shape=(224,224,3))

print("✅ All models loaded!")

# ---- CALCULATE ACCURACY ----
print("\n📊 Calculating results...")

# MobileNetV2 accuracy on both datasets
_, acc_mb1 = mobilenet.evaluate(test_data1, verbose=0)
_, acc_mb2 = mobilenet.evaluate(test_data2, verbose=0)

# CNN accuracy on both datasets
_, acc_cnn1 = cnn.evaluate(test_data1, verbose=0)
_, acc_cnn2 = cnn.evaluate(test_data2, verbose=0)

# SVM - extract features first, then evaluate
test_data1.reset()  # Reset generator to start from beginning
test_data2.reset()
X_test1 = base.predict(test_data1, verbose=0)  # Extract features from dataset 1
X_test2 = base.predict(test_data2, verbose=0)  # Extract features from dataset 2
acc_svm1 = joblib.load(r'models\svm_model.pkl').score(X_test1, test_data1.classes)
acc_svm2 = joblib.load(r'models\svm_model.pkl').score(X_test2, test_data2.classes)

# ---- PRINT RESULTS ----
print("\n" + "="*50)
print("📊 COMPARISON RESULTS")
print("="*50)
print(f"{'Algorithm':<15} {'Dataset 1':>10} {'Dataset 2':>10}")
print("-"*50)
print(f"{'MobileNetV2':<15} {acc_mb1*100:>9.2f}% {acc_mb2*100:>9.2f}%")
print(f"{'CNN':<15} {acc_cnn1*100:>9.2f}% {acc_cnn2*100:>9.2f}%")
print(f"{'SVM':<15} {acc_svm1*100:>9.2f}% {acc_svm2*100:>9.2f}%")
print("="*50)

# ---- PLOT COMPARISON CHART ----
algorithms = ['MobileNetV2', 'CNN', 'SVM']
dataset1_scores = [acc_mb1*100, acc_cnn1*100, acc_svm1*100]
dataset2_scores = [acc_mb2*100, acc_cnn2*100, acc_svm2*100]

x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each dataset
bars1 = ax.bar(x - width/2, dataset1_scores, width, label='Dataset 1', color='steelblue')
bars2 = ax.bar(x + width/2, dataset2_scores, width, label='Dataset 2', color='orange')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Algorithm Comparison: MobileNetV2 vs CNN vs SVM')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()
ax.set_ylim(0, 100)

# Add accuracy labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(r'results\comparison.png')
print("\n✅ Comparison chart saved: results/comparison.png")