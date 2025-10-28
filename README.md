# Hindi Handwritten Character Recognition ✍️

This repository contains the code for a project that builds, trains, and evaluates a deep learning model to recognize handwritten Hindi characters (Devanagari script). The notebook experiments with two different custom-built Convolutional Neural Network (CNN) architectures to classify images into 36 distinct character classes.

## 🤖 Models & Architectures

This project does not use a pre-trained model. Instead, it defines and compares two separate **custom CNN architectures** built from scratch using the TensorFlow/Keras functional API. Both models are designed for image classification on `(32, 32, 3)` images and output a classification for 36 classes.

---

### Architecture 1
A deep CNN model designed to capture intricate features using multiple convolutional blocks.

* **Key Layers:**
    * **Block 1:** `Conv2D(64)` -> `BatchNormalization` -> `Conv2D(64)` -> `BatchNormalization` -> `MaxPool2D`
    * **Block 2:** `Conv2D(128)` -> `BatchNormalization` -> `Conv2D(128)` -> `BatchNormalization` -> `MaxPool2D`
    * **Block 3:** `Conv2D(256)` -> `BatchNormalization` -> `MaxPool2D`
    * **Classifier Head:** `Flatten` -> `Dropout(0.4)` -> `Dense(512)` -> `BatchNormalization` -> `Dropout(0.4)` -> `Dense(36, 'softmax')`
* **Optimizer:** `adam`
* **Loss Function:** `categorical_crossentropy`

---

### Architecture 2
A more streamlined CNN that uses strided convolutions for down-sampling, representing a different approach to feature extraction.

* **Key Layers:**
    * `Conv2D(32, strides=2)` -> `BatchNormalization`
    * `Conv2D(64, strides=2)` -> `BatchNormalization` -> `MaxPool2D`
    * `Conv2D(128, strides=2)` -> `BatchNormalization` -> `MaxPool2D`
    * **Classifier Head:** `Flatten` -> `Dropout(0.3)` -> `Dense(512)` -> `BatchNormalization` -> `Dropout(0.3)` -> `Dense(36, 'softmax')`
* **Optimizer:** `SGD` (Stochastic Gradient Descent) with momentum.
* **Loss Function:** `categorical_crossentropy`

---

## 🛠️ Technology Used

This project is built entirely in Python, leveraging the **TensorFlow** and **Keras** ecosystems for deep learning.

* **Core Framework:** **TensorFlow 2.x** (with `tf.keras`) is used for all aspects of model creation, training, and evaluation.
* **Data Pipeline:** `tf.keras.preprocessing.image_dataset_from_directory` is used to efficiently load and create training and validation `tf.data.Dataset` objects directly from image directories.
* **Model Building:** The Keras Functional API (`Input`, `Model`) is used to construct the flexible CNN architectures.
* **Core Layers:** The models are built with standard Keras layers, including `Conv2D`, `MaxPool2D`, `BatchNormalization`, `Flatten`, `Dense`, and `Dropout`.
* **Data Visualization:**
    * **`matplotlib.pyplot`**: Used to plot the training and validation accuracy/loss curves for model evaluation.
    * **`seaborn`**: Used along with `sklearn.metrics.confusion_matrix` to generate and display a detailed confusion matrix, showing model performance on a per-class basis.
* **Inference & Utilities:**
    * **`numpy`**: Used for numerical operations and processing prediction outputs.
    * **`cv2` (OpenCV)**: Used in the final "Image Prediction" step to load, resize, and preprocess a single test image from disk.
