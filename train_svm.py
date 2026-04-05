import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import joblib

# ---- AYARLAR ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"

# ---- DATA ----
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    DATASET1 + r"\Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

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

print("✅ Data hazırdır!")

# ---- MobileNetV2 ilə feature çıxar ----
print("\n🔍 Xüsusiyyətlər çıxarılır...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

def extract_features(data):
    features = base_model.predict(data, verbose=1)
    labels = data.classes
    return features, labels

X_train, y_train = extract_features(train_data)
X_test1, y_test1 = extract_features(test_data1)
X_test2, y_test2 = extract_features(test_data2)

print("✅ Xüsusiyyətlər hazırdır!")

# ---- SVM ----
print("\n🚀 SVM öyrənir...")
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

# ---- NƏTİCƏLƏR ----
acc1 = accuracy_score(y_test1, svm.predict(X_test1))
acc2 = accuracy_score(y_test2, svm.predict(X_test2))

print(f"\n📊 SVM Dataset 1 Accuracy: %{acc1*100:.2f}")
print(f"📊 SVM Dataset 2 Accuracy: %{acc2*100:.2f}")

# ---- SAXLA ----
joblib.dump(svm, r'models\svm_model.pkl')
print("✅ SVM Model saxlanıldı!")