import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# ---- AYARLAR ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"

# ---- DATA ----
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    DATASET1 + r"\Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data1 = test_datagen.flow_from_directory(
    DATASET1 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data2 = test_datagen.flow_from_directory(
    DATASET2 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("✅ Data hazırdır!")

# ---- CNN MODELİ ----
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ CNN Model hazırdır!")

# ---- ÖYRƏT ----
print("\n🚀 CNN öyrənir...")
history = model.fit(
    train_data,
    validation_data=test_data1,
    epochs=EPOCHS
)

model.save(r'models\cnn_model.keras')
print("✅ CNN Model saxlanıldı!")

# ---- DATASET 2 TEST ----
print("\n📊 Dataset 2 ilə test edilir...")
loss2, acc2 = model.evaluate(test_data2)
print(f"CNN Dataset 2 Accuracy: %{acc2*100:.2f}")

# ---- QRAFİK ----
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('CNN Loss')
plt.legend()

plt.savefig(r'results\cnn_results.png')
print("✅ Qrafik saxlanıldı: results/cnn_results.png")