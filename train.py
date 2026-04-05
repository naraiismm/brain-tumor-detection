import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# ---- AYARLAR ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"

# ---- DATA HAZIRLIĞI ----
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Dataset 1 - Train
train_data = datagen.flow_from_directory(
    DATASET1 + r"\Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Dataset 1 - Test
test_data1 = test_datagen.flow_from_directory(
    DATASET1 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Dataset 2 - Test
test_data2 = test_datagen.flow_from_directory(
    DATASET2 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("✅ Hər iki dataset hazırdır!")
print("Siniflər:", train_data.class_indices)

# ---- MODELİ QUR ----
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- ÖYRƏT ----
print("\n🚀 Model öyrənir...")
history = model.fit(
    train_data,
    validation_data=test_data1,
    epochs=EPOCHS
)

model.save('mobilenet_model.keras')
print("✅ Model saxlanıldı!")

# ---- DATASET 2 İLƏ TEST ----
print("\n📊 Dataset 2 ilə test edilir...")
loss2, acc2 = model.evaluate(test_data2)
print(f"Dataset 2 Accuracy: %{acc2*100:.2f}")

# ---- QRAFİK ----
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig('results.png')
print("✅ Qrafik saxlanıldı: results.png")