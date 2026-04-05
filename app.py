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

# ---- LOAD MODELS ----
print("Loading models...")
mobilenet = load_model(r'models\mobilenet_model.keras')
cnn = load_model(r'models\cnn_model.keras')
print("Models loaded!")

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
    return mb_class, mb_conf, cnn_class, cnn_conf

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    result_label.config(text="Analyzing...", fg='gray')
    root.update()
    mb_class, mb_conf, cnn_class, cnn_conf = predict(file_path)
    result_label.config(
        text=f"MobileNetV2: {mb_class} ({mb_conf:.1f}%)\nCNN: {cnn_class} ({cnn_conf:.1f}%)",
        fg=COLORS[mb_class]
    )

# ---- UI ----
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("500x700")
root.configure(bg='#1a1a2e')

tk.Label(root,
         text="Brain Tumor Detection",
         font=('Arial', 20, 'bold'),
         bg='#1a1a2e', fg='white').pack(pady=20)

tk.Label(root,
         text="Upload an MRI scan to detect tumor type",
         font=('Arial', 11),
         bg='#1a1a2e', fg='gray').pack()

image_label = tk.Label(root, bg='#16213e', width=40, height=15)
image_label.pack(pady=20)

tk.Button(root,
          text="Upload MRI Image",
          font=('Arial', 13, 'bold'),
          bg='#0f3460', fg='white',
          padx=20, pady=10,
          command=upload_image).pack(pady=15)

result_label = tk.Label(root,
                         text="Upload an image to see results",
                         font=('Arial', 14, 'bold'),
                         bg='#1a1a2e', fg='gray',
                         justify='center')
result_label.pack(pady=20)

root.mainloop()