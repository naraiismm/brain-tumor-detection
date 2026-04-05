import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ---- SETTINGS ----
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = (224, 224)
COLORS = {
    'Glioma': '#FF4444',
    'Meningioma': '#FF8800',
    'No Tumor': '#00CC44',
    'Pituitary': '#4488FF'
}

# ---- LOAD MODEL ----
print("Loading model...")
mobilenet = load_model(r'models\mobilenet_model.keras')
print("Model loaded!")

# ---- PREDICT FUNCTION ----
def predict(img_path):
    # Load and preprocess image
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction using MobileNetV2
    mb_pred = mobilenet.predict(img_array, verbose=0)
    mb_class = CLASSES[np.argmax(mb_pred)]
    mb_conf = np.max(mb_pred) * 100

    return mb_class, mb_conf

# ---- UPLOAD IMAGE ----
def upload_image():
    # Open file dialog
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # Show selected image
    img = Image.open(file_path).resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Show loading text
    result_label.config(text="Analyzing...", fg='gray')
    root.update()

    # Run prediction
    mb_class, mb_conf = predict(file_path)

    # Display result
    result_label.config(
        text=f"Result: {mb_class} ({mb_conf:.1f}%)",
        fg=COLORS[mb_class]
    )

# ---- UI ----
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("500x700")
root.configure(bg='#1a1a2e')
root.resizable(False, False)

# Title
tk.Label(root,
         text="🧠 Brain Tumor Detection",
         font=('Arial', 20, 'bold'),
         bg='#1a1a2e', fg='white').pack(pady=20)

# Subtitle
tk.Label(root,
         text="Upload an MRI scan to detect tumor type",
         font=('Arial', 11),
         bg='#1a1a2e', fg='gray').pack()

# Image display area
image_label = tk.Label(root, bg='#16213e', width=45, height=20)
image_label.pack(pady=20)

# Upload button
tk.Button(root,
          text="📁 Upload MRI Image",
          font=('Arial', 13, 'bold'),
          bg='#0f3460', fg='white',
          padx=20, pady=10,
          cursor='hand2',
          command=upload_image).pack(pady=15)

# Result label
result_label = tk.Label(root,
                         text="Upload an image to see results",
                         font=('Arial', 16, 'bold'),
                         bg='#1a1a2e', fg='gray',
                         justify='center')
result_label.pack(pady=20)

# Footer
tk.Label(root,
         text="Classes: Glioma | Meningioma | No Tumor | Pituitary",
         font=('Arial', 9),
         bg='#1a1a2e', fg='#444466').pack(side='bottom', pady=10)

root.mainloop()