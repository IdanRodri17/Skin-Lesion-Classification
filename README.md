# Skin Lesion Classification using Deep Learning 🩺🔬

A **Medical AI research project** focused on classifying dermatological skin lesions from clinical images using **Deep Learning and Computer Vision** techniques. The project demonstrates the full pipeline — from data preprocessing and model design to evaluation and explainability — with an emphasis on real-world medical applicability.

---

## 🚀 Overview

Early detection of skin cancer plays a critical role in improving patient outcomes. This project implements a robust **Computer Vision pipeline** that analyzes dermoscopic images and classifies skin lesions into diagnostic categories, supporting early diagnosis and informed clinical decision-making.

The project was developed as part of advanced academic training in **Artificial Intelligence and Computer Vision**, bridging theoretical Deep Learning concepts with a real-world, high-impact medical problem.

---

## ✨ Key Features

* **Deep Learning Architecture**: CNN-based model (ResNet-based architecture) optimized for medical image feature extraction.
* **Data Preprocessing**: Image normalization, resizing, and augmentation techniques to improve generalization on limited medical datasets.
* **Model Evaluation**: Performance measured using Accuracy, Precision, Recall, AUC, and Confusion Matrices.
* **Model Explainability**: **Grad-CAM heatmaps** to visualize regions of interest influencing model predictions.
* **End-to-End Pipeline**: Dataset loading → preprocessing → training → evaluation → inference.

---

## 🛠️ Tech Stack

* **Frameworks**: TensorFlow, Keras
* **Libraries**: NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn
* **Domain**: Medical Imaging, Computer Vision, Deep Learning

---

## 📊 Dataset Information

The model was trained and evaluated on dermoscopic skin lesion images obtained from publicly available medical datasets (e.g., **ISIC Archive / HAM10000**), widely used in academic research for skin cancer classification.

All data was used strictly for **research and educational purposes**.

---

## 🏗️ Model Architecture

1. **Input Layer**: Preprocessed dermoscopic images.
2. **Convolutional Blocks**: Feature extraction of color, texture, and lesion patterns using a ResNet-based CNN.
3. **Fully Connected Layers**: High-level feature aggregation and classification.
4. **Output Layer**: Probability distribution across lesion classes.

---

## 📈 Results

* Achieved strong classification performance on the validation set.
* Demonstrated effective feature localization using Grad-CAM visualizations.

*(Exact metrics may vary depending on training configuration and dataset split.)*

---

## 💻 How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/IdanRodri17/Skin-Lesion-Classification.git
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run inference on an image**:

```bash
python classify.py --image path/to/image.jpg
```

---

## ⚠️ Disclaimer

This project is intended **for research and educational purposes only**. It is **not approved for clinical use** and should not be used as a diagnostic tool in medical practice.

---

## 👨‍💻 Author

**Idan Rodrigez**
Computer Science Graduate | Junior Software Developer

**Skills**: Artificial Intelligence, Computer Vision, Deep Learning, Cloud Architecture
**Interests**: Real-world AI systems

---

If you found this project interesting, feel free to ⭐ the repository or connect with me on LinkedIn.
