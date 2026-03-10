# 🎗️ Cancer Detection & Stage Prediction using Machine Learning

A collection of machine learning projects focused on cancer diagnosis and stage classification using classical ML algorithms, built with Python and scikit-learn.

---

## 📁 Project Structure

```
├── Untitled.ipynb               # Breast cancer diagnosis using SVM (Wisconsin dataset)
├── SVM_from_scratch.ipynb       # SVM implementation from scratch (no libraries)
├── cancer_stage_prediction.py   # Cancer stage classification (Low / Medium / High)
└── breast-cancer-wisconsin.data # UCI Breast Cancer Wisconsin dataset
```

---

## 📌 Projects Overview

### 1. Breast Cancer Diagnosis — `Untitled.ipynb`
Classifies breast tumors as **benign (2)** or **malignant (4)** using the UCI Wisconsin dataset.

- **Dataset:** `breast-cancer-wisconsin.data` — 699 samples, 9 cytology features
- **Model:** Support Vector Machine (`sklearn.svm.SVC`)
- **Workflow:** Load data → define features/labels → train-test split (80/20) → train SVM → evaluate accuracy → predict on new samples
- **Result:** ~96–97% accuracy on test data

**Features used:**
| Feature | Description |
|---|---|
| `clump_thickness` | Clump Thickness |
| `unif_cell_size` | Uniformity of Cell Size |
| `unif_cell_shape` | Uniformity of Cell Shape |
| `marg_adhesion` | Marginal Adhesion |
| `single_epith_cell_size` | Single Epithelial Cell Size |
| `bare_nuclie` | Bare Nuclei |
| `bland_chrom` | Bland Chromatin |
| `norm_nucleoli` | Normal Nucleoli |
| `mitoses` | Mitoses |

---

### 2. SVM from Scratch — `SVM_from_scratch.ipynb`
A ground-up implementation of a Support Vector Machine **without using scikit-learn**, to understand the math behind SVMs.

- Implements the `Support_Vector_Machine` class with `fit()` and `predict()` methods
- Uses the sign of `X·W + b` for classification
- Applies geometric transforms to explore the optimal hyperplane
- Visualizes class separation with matplotlib

---

### 3. Cancer Stage Prediction — `cancer_stage_prediction.py`
Multi-class classification to predict cancer severity level: **Low (0)**, **Medium (1)**, or **High (2)**.

- **Dataset:** `cancer_patients.csv` (external, not included)
- **Preprocessing:** Label encoding, MinMaxScaler normalization, missing value checks
- **Models trained & compared:**
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM with RBF kernel)
  - Random Forest (1000 estimators)
  - Ridge Regression
  - Multi-Layer Perceptron (MLP Neural Network)
- **Visualization:** K-Means clustering plot + confusion matrix heatmaps for all models

---

## 🛠️ Tech Stack

- Python 3.10+
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cancer-ml-detection.git
cd cancer-ml-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3. Run the notebooks
```bash
jupyter notebook
```
Open `Untitled.ipynb` or `SVM_from_scratch.ipynb` in your browser.

### 4. Run the stage prediction script
```bash
python cancer_stage_prediction.py
```
> ⚠️ Make sure `cancer_patients.csv` is present in the same directory.

---

## 📊 Dataset

**UCI Breast Cancer Wisconsin Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))
- 699 instances, 9 features, binary target (benign / malignant)
- Missing values handled by dropping or replacing with `-99999`

---

## 📈 Results Summary

| Model | Task | Accuracy |
|---|---|---|
| SVM (sklearn) | Breast Cancer Diagnosis | ~96–97% |
| KNN | Cancer Stage Prediction | High |
| SVM (RBF) | Cancer Stage Prediction | High |
| Random Forest | Cancer Stage Prediction | Very High |
| MLP Neural Network | Cancer Stage Prediction | High |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
