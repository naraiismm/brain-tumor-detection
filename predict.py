import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
import os

# ---- SETTINGS ----
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # 4 tumor classes
IMG_SIZE = (224, 224)  # Input image size for the models

# ---- LOAD TRAINED MODELS ----
print("📦 Loading models...")
mobilenet = load_model(r'models\mobilenet_model.keras')  # Load MobileNetV2 model
cnn = load_model(r'models\cnn_model.keras')              # Load CNN model
print("✅ Models loaded successfully!")

def predict(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    mb_pred = mobilenet.predict(img_array, verbose=0)
    cnn_pred = cnn.predict(img_array, verbose=0)

    mb_class = CLASSES[np.argmax(mb_pred)]
    cnn_class = CLASSES[np.argmax(cnn_pred)]
    mb_conf = np.max(mb_pred) * 100
    cnn_conf = np.max(cnn_pred) * 100

    # Voting system
    if mb_class == cnn_class:
        final = f"✅ Both agree: {mb_class}"
    else:
        final = f"⚠️ Uncertain - Please consult a doctor"

    return mb_class, mb_conf, cnn_class, cnn_conf, final

    # Visualize the input image with prediction results
    plt.figure(figsize=(6,6))
    plt.imshow(image.load_img(img_path))
    plt.title(f"MobileNetV2: {mb_class} ({mb_conf:.1f}%)\n"
              f"CNN: {cnn_class} ({cnn_conf:.1f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---- GET IMAGE PATH FROM USER ----
if len(sys.argv) > 1:
    img_path = sys.argv[1]  # Get path from command line argument
else:
    img_path = input("Enter image path: ").strip('"')  # Ask user for image path

# Check if file exists and run prediction
if os.path.exists(img_path):
    predict_image(img_path)
else:
    print("❌ Image not found!")