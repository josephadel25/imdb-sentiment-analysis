# 🎬 IMDB Movie Review Sentiment Analysis

This project uses machine learning to classify IMDB movie reviews as **positive** or **negative**.

---

## 📦 Dataset

- **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Please place `IMDB Dataset.csv` in the project folder.

---

## 🔄 Workflow

1. Load and clean data
2. Preprocess text (lowercase, remove stopwords & punctuation)
3. Convert text to numbers using **TF-IDF**
4. Train two models:
   - Logistic Regression
   - Naive Bayes
5. Compare accuracy

---

## 🧪 Model Accuracy

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~89%     |
| Naive Bayes         | ~85%     |

---

## 📊 Visuals

- Bar chart comparing model accuracy  
- Confusion Matrix Comparison (visualized)
- Optional: Stylish plots using `mplcyberpunk`

---

## 🧰 Tech Stack

- Python
- NLTK
- Scikit-learn
- Pandas / Matplotlib
- TF-IDF Vectorizer

---

## ▶️ How to Run

1. Clone the repo
2. Install required libraries
3. Add the dataset
4. Run sentement.ipynb in Jupyter or VS Code

## 📁 Folder Structure
.
├── sentement.ipynb
├── IMDB Dataset.csv
└── README.md
