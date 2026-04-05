import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# ---- SETTINGS ----
IMG_SIZE = (224, 224)      # Input image size
BATCH_SIZE = 32             # Number of images per batch
EPOCHS = 10                 # Number of training epochs

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"   # Primary dataset
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"  # Secondary dataset

# ---- DATA PREPARATION ----
# Training data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0,1]
    rotation_range=10,       # Random rotation up to 10 degrees
    horizontal_flip=True     # Random horizontal flip
)

# Test data generator without augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Dataset 1 - Training set
train_data = datagen.flow_from_directory(
    DATASET1 + r"\Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Dataset 1 - Validation set
test_data1 = test_datagen.flow_from_directory(
    DATASET1 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Dataset 2 - External test set (different source)
test_data2 = test_datagen.flow_from_directory(
    DATASET2 + r"\Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("✅ Both datasets loaded successfully!")
print("Classes:", train_data.class_indices)

# ---- BUILD MODEL ----
# Load MobileNetV2 pre-trained on ImageNet (Transfer Learning)
base_model = MobileNetV2(
    weights='imagenet',      # Use pre-trained ImageNet weights
    include_top=False,       # Exclude original classification layer
    input_shape=(224, 224, 3)
)
base_model.trainable = False # Freeze base model weights

# Add custom classification layers on top
x = GlobalAveragePooling2D()(base_model.output)  # Reduce spatial dimensions
x = Dropout(0.3)(x)                               # Prevent overfitting
x = Dense(128, activation='relu')(x)              # Fully connected layer
output = Dense(4, activation='softmax')(x)         # Output layer for 4 classes

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- TRAIN MODEL ----
print("\n🚀 Training started...")
history = model.fit(
    train_data,
    validation_data=test_data1,
    epochs=EPOCHS
)

model.save('mobilenet_model.keras')
print("✅ Model saved!")

# ---- EVALUATE ON DATASET 2 ----
# Test on unseen data from a different source
print("\n📊 Evaluating on Dataset 2...")
loss2, acc2 = model.evaluate(test_data2)
print(f"Dataset 2 Accuracy: {acc2*100:.2f}%")

# ---- PLOT RESULTS ----
plt.figure(figsize=(12,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()

plt.savefig('results.png')
print("✅ Results chart saved: results.png")