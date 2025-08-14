# 🍷 Wine Reviews Classification using TensorFlow & NLP

This project uses **Natural Language Processing (NLP)** and **deep learning** with TensorFlow to classify wine reviews as *high-quality* or *not*, based on textual descriptions. Reviews are labeled based on wine ratings, where wines scoring **90 points or more** are considered **high-quality** (label = 1).

---

## 🧠 Project Objective

To train a binary classification model that reads a wine's textual review (description) and predicts whether the wine is **highly rated (90+)** using:

- Pretrained text embeddings (TF Hub: `nnlm-en-dim50`)
- LSTM-based deep learning architecture
- TensorFlow and Keras

---

## 📁 Dataset

- **Source**: [`wine-reviews.csv`](https://www.kaggle.com/datasets/zynicide/wine-reviews) (Kaggle)
- **Features Used**:
  - `description`: The text review of the wine
  - `points`: Score between 80 and 100 (used to generate binary labels)

- **Label Creation**:
  ```python
  df["label"] = (df.points >= 9


---
🛠️ Tools & Libraries

Python

Pandas & NumPy

TensorFlow & Keras

TensorFlow Hub

scikit-learn

imbalanced-learn (for oversampling, if needed)


---
📊 Data Processing Pipeline

Load dataset

Drop missing values in description or points

Create binary labels (1 if points ≥ 90, else 0)

Train/Val/Test split (80/10/10)

Convert to TensorFlow datasets using tf.data

Use TF-Hub embeddings and later a custom LSTM model

Train, validate, and evaluate performance


---
performance

🧪 Model 1: TF-Hub Embedding + Dense Layers

Embedding: nnlm-en-dim50

Architecture:

TF-Hub Embedding Layer

Dense(16) + Dropout

Dense(16) + Dropout

Output Layer (Sigmoid)
---


📈 Results:

Validation Accuracy: ~69%

Test Accuracy: ~68%

---
🧠 Model 2: LSTM + Word Embedding

TextVectorization + Embedding Layer

LSTM Layer

Dense Layers + Dropout

📈 Results:

Validation Accuracy: ~84%

Test Accuracy: ~84%

Loss: 0.34

This model significantly outperforms the basic dense model, showing LSTM's power in sequence-based classification.


---
🙌 Acknowledgements

Wine Reviews Dataset on Kaggle

TensorFlow Hub

TensorFlow Tutorials

---
📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
📬 Contact

Author: Saniya Chhabra
