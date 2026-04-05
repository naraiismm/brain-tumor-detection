import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ---- AYARLAR ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"

# ---- DATA ----
datagen = ImageDataGenerator(rescale=1./255)

test_data1 = datagen.flow_from_directory(
    DATASET1 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_data2 = datagen.flow_from_directory(
    DATASET2 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ---- MODELLƏRİ YÜKLƏ ----
print("📦 Modellər yüklənir...")
mobilenet = load_model(r'models\mobilenet_model.keras')
cnn = load_model(r'models\cnn_model.keras')
svm = joblib.load(r'models\svm_model.pkl')

# SVM üçün feature extractor
base = MobileNetV2(weights='imagenet', include_top=False,
                   pooling='avg', input_shape=(224,224,3))

print("✅ Modellər hazırdır!")

# ---- ACCURACY HESABLA ----
print("\n📊 Nəticələr hesablanır...")

# MobileNetV2
_, acc_mb1 = mobilenet.evaluate(test_data1, verbose=0)
_, acc_mb2 = mobilenet.evaluate(test_data2, verbose=0)

# CNN
_, acc_cnn1 = cnn.evaluate(test_data1, verbose=0)
_, acc_cnn2 = cnn.evaluate(test_data2, verbose=0)

# SVM
test_data1.reset()
test_data2.reset()
X_test1 = base.predict(test_data1, verbose=0)
X_test2 = base.predict(test_data2, verbose=0)
acc_svm1 = joblib.load(r'models\svm_model.pkl').score(X_test1, test_data1.classes)
acc_svm2 = joblib.load(r'models\svm_model.pkl').score(X_test2, test_data2.classes)

# ---- NƏTİCƏLƏR ----
print("\n" + "="*50)
print("📊 MÜQAYİSƏ NƏTİCƏLƏRİ")
print("="*50)
print(f"{'Algoritma':<15} {'Dataset 1':>10} {'Dataset 2':>10}")
print("-"*50)
print(f"{'MobileNetV2':<15} {acc_mb1*100:>9.2f}% {acc_mb2*100:>9.2f}%")
print(f"{'CNN':<15} {acc_cnn1*100:>9.2f}% {acc_cnn2*100:>9.2f}%")
print(f"{'SVM':<15} {acc_svm1*100:>9.2f}% {acc_svm2*100:>9.2f}%")
print("="*50)

# ---- MÜQAYİSƏ QRAFİKİ ----
algorithms = ['MobileNetV2', 'CNN', 'SVM']
dataset1_scores = [acc_mb1*100, acc_cnn1*100, acc_svm1*100]
dataset2_scores = [acc_mb2*100, acc_cnn2*100, acc_svm2*100]

x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, dataset1_scores, width, label='Dataset 1', color='steelblue')
bars2 = ax.bar(x + width/2, dataset2_scores, width, label='Dataset 2', color='orange')

ax.set_ylabel('Accuracy (%)')
ax.set_title('3 Algoritmanın Müqayisəsi')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()
ax.set_ylim(0, 100)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(r'results\comparison.png')
print("\n✅ Müqayisə qrafiki saxlanıldı: results/comparison.png")