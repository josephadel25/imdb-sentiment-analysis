# 🎬 IMDB Movie Review Sentiment Analysis

This project applies Natural Language Processing (NLP) techniques to classify IMDB movie reviews as either **positive** or **negative** using machine learning models. The dataset contains 50,000 reviews split evenly between sentiments.

---

## 📂 Dataset

- **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Columns**:
  - `review`: The actual text of the movie review.
  - `sentiment`: Label (`positive` or `negative`), later mapped to 1 and 0.

---

## 📊 Workflow

1. **Mount Google Drive and Load Dataset**
2. **Text Preprocessing**:
   - Lowercasing
   - Removing stopwords
   - Removing punctuation
   - Tokenization using `nltk`
3. **Vectorization**:
   - Used **TF-IDF Vectorizer** to convert text into numerical features.
4. **Model Training**:
   - Trained both **Logistic Regression** and **Naive Bayes** classifiers.
5. **Evaluation**:
   - Compared models using **Accuracy Score** and **Bar Plots**
6. **Bonus**:
   - Visualized **most frequent positive and negative words** from the reviews.

---

## 📈 Results

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | ~89%     |
| Naive Bayes         | ~85%     |

✅ Logistic Regression performed slightly better on the test set.

---

## 📌 Visualizations

- **Bar chart** comparing model accuracies.
- **Top 15 most frequent words** in positive and negative reviews displayed using horizontal bar plots.

---

## 📚 Libraries Used

- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `nltk` for NLP tasks
- `sklearn` for ML models and metrics

---

## 🚀 How to Run

1. Open in [Google Colab](https://colab.research.google.com/)
2. Upload or mount the dataset via Google Drive.
3. Run all cells in the notebook.

---

## 📎 Credits

- Dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- NLP with `NLTK`
- Machine learning with `Scikit-learn`

---

## 📌 Future Improvements

- Add deep learning models (e.g., LSTM)
- Perform hyperparameter tuning
- Use WordClouds for word visualization
