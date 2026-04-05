import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# ---- SETTINGS ----
IMG_SIZE = (224, 224)   # Input image size
BATCH_SIZE = 32          # Number of images per batch
EPOCHS = 10              # Number of training epochs

DATASET1 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset"   # Primary dataset
DATASET2 = r"C:\Users\THE SULEYMANOVS\brain_tumor\dataset2"  # Secondary dataset

# ---- DATA PREPARATION ----
# Training data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values to [0,1]
    rotation_range=10,     # Random rotation up to 10 degrees
    horizontal_flip=True   # Random horizontal flip
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

print("✅ Data loaded successfully!")

# ---- BUILD CNN MODEL ----
# Custom CNN architecture built from scratch
model = Sequential([
    # First convolutional block - detect basic features (edges, textures)
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),  # Normalize activations for stable training
    MaxPooling2D(2,2),     # Reduce spatial dimensions by half

    # Second convolutional block - detect complex features
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Third convolutional block - detect high-level features
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),             # Convert 2D feature maps to 1D vector
    Dropout(0.5),          # Randomly drop 50% of neurons to prevent overfitting
    Dense(256, activation='relu'),  # Fully connected layer
    Dense(4, activation='softmax')  # Output layer for 4 tumor classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ CNN Model built successfully!")

# ---- TRAIN MODEL ----
print("\n🚀 CNN training started...")
history = model.fit(
    train_data,
    validation_data=test_data1,
    epochs=EPOCHS
)

model.save(r'models\cnn_model.keras')
print("✅ CNN Model saved!")

# ---- EVALUATE ON DATASET 2 ----
# Test on unseen data from a different source
print("\n📊 Evaluating on Dataset 2...")
loss2, acc2 = model.evaluate(test_data2)
print(f"CNN Dataset 2 Accuracy: {acc2*100:.2f}%")

# ---- PLOT RESULTS ----
plt.figure(figsize=(12,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('CNN Loss')
plt.legend()

plt.savefig(r'results\cnn_results.png')
print("✅ Results chart saved: results/cnn_results.png")