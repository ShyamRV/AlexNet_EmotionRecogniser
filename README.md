
# Emotion Recognition using AlexNet on FER-2013 with Real-Time Computer Vision

This project implements a complete **emotion recognition pipeline** using the **AlexNet deep CNN architecture**, trained on the **FER-2013 (Facial Expression Recognition)** dataset.
The system performs end-to-end processing: **data loading, preprocessing, model training, evaluation, and real-time emotion detection using OpenCV**.

With real-time integration, the model can classify emotions directly from a webcam feed, making it suitable for interactive and applied AI systems.

---

## 📌 Highlights – Full Work Done in This Project

### ✔ Dataset: FER-2013

You used the FER-2013 dataset, which contains:

* **35,887** 48×48 grayscale images
* **7 emotion classes:**

  * Angry
  * Disgust
  * Fear
  * Happy
  * Sad
  * Surprise
  * Neutral
* Training, Public Test, Private Test splits

You processed this raw dataset for AlexNet's requirements.

---

## ✔ Data Preprocessing

* Read the FER-2013 CSV dataset
* Converted pixel strings to numpy arrays
* Resized grayscale 48×48 images to **227×227×3** (AlexNet format)
* Normalized pixel values
* One-hot encoded emotion labels
* Created train/validation/test splits

---

## ✔ Data Augmentation

Used real-time augmentation to improve performance:

* Rotation
* Horizontal Flip
* Zoom
* Width/Height Shift
* Brightness Variations
* Rescale + Batch Generation

---

## ✔ AlexNet Architecture Implementation

You successfully recreated a full AlexNet model:

* Conv + ReLU layers
* MaxPooling
* Deep convolution blocks
* Fully connected 4096 × 4096 layers
* Dropout regularization
* Softmax output for **7 emotion classes**

Optimized and compiled with:

* Loss: Categorical Crossentropy
* Optimizer: Adam/SGD
* Metric: Accuracy

---

## ✔ Model Training

* Trained the model on augmented FER-2013 images
* Visualized training/validation accuracy & loss curves
* Tuned epoch count, learning rate, batch size
* Saved the model for later use in real-time testing

---

## ✔ Model Evaluation

* Evaluated on FER-2013 test set
* Calculated classification accuracy
* Displayed confusion matrix
* Shown correctly/incorrectly predicted images

---

# 🔥 Real-Time Emotion Recognition (Your Key Feature)

You integrated OpenCV with your trained AlexNet model to enable **live emotion detection**.

### ✔ Real-Time Pipeline

Using webcam input:

1. Capture frame using OpenCV
2. Detect face (Haar Cascade or DNN)
3. Crop face region
4. Resize to 227×227
5. Normalize
6. Predict emotion using trained AlexNet model
7. Display predicted emotion above face with bounding box

### ✔ Real-Time Capabilities

* Real-time frame-by-frame prediction
* Multi-face emotion detection
* Smooth and optimized processing
* Option to test videos or images

This transforms the project from a research model into an **interactive, production-ready AI emotion recognition system**.

---

## 📂 Project Structure

```
├── Alexnet_Emotion.ipynb         # Main training notebook
├── real_time_emotion.py          # Real-time CV script
├── README.md                     # Documentation
├── fer2013.csv                   # FER-2013 dataset (optional)
└── model/                        # Saved model files
```

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* OpenCV
* Matplotlib
* scikit-learn
* TQDM


---

## 📬 Contact

**Shyamji Pandey**

LinkedIn: *https://www.linkedin.com/in/shyamji-pandey/*

Portfolio: *https://shyamjipandey.vercel.app/*
---
