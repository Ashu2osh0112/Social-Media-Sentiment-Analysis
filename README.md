# 💬 Social Media Sentiment Analysis

This project focuses on analyzing the sentiment of tweets (positive or negative) using Natural Language Processing (NLP) and machine learning techniques. It uses the **Sentiment140** dataset containing 1.6 million pre-labeled tweets and builds a classification model using **Logistic Regression**.

---

## 📌 Project Objective

To develop a machine learning model that can classify the sentiment of tweets as **positive** or **negative** by processing raw text data from social media.

---

## 🛠️ Technologies & Tools

- **Programming Language**: Python
- **Libraries Used**:
  - `pandas`, `numpy` – Data manipulation
  - `re` – Text cleaning using regular expressions
  - `nltk` – Stopwords removal & stemming
  - `scikit-learn` – TF-IDF, model training, evaluation
  - `pickle` – Model saving & loading
- **Dataset**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (Kaggle)
- **Environment**: Google Colab / Jupyter Notebook

---

## 📂 Dataset Description

The dataset contains 1.6 million tweets labeled for sentiment.

| Column | Description               |
|--------|---------------------------|
| target | Sentiment (0 = Negative, 4 = Positive) |
| ids    | Tweet ID                  |
| date   | Date of the tweet         |
| flag   | Query                     |
| user   | Username                  |
| text   | Tweet content             |

*Note: The target value `4` is remapped to `1` (positive) for binary classification.*

---

## 🔄 Project Workflow

### 1. Data Collection
- Used Kaggle API to download the dataset.
- Extracted the compressed `.zip` file and loaded data using pandas.

### 2. Data Preprocessing
- Removed unwanted characters using regex.
- Converted text to lowercase.
- Removed stopwords using NLTK.
- Applied stemming using **PorterStemmer**.

### 3. Feature Engineering
- Converted text data into numerical vectors using **TF-IDF Vectorizer**.

### 4. Model Training
- Split data into training and test sets (80/20).
- Trained a **Logistic Regression** classifier with `max_iter=1000`.

### 5. Evaluation
- Accuracy:
  - Training: ~78.4%
  - Testing: ~77.8%
- Verified model performance on unseen data.

### 6. Model Saving & Deployment
- Saved the model using `pickle` for future reuse.
- Demonstrated predictions on new tweet samples.

---

## 🧪 Example Usage

```python
import pickle
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
prediction = loaded_model.predict(X_new)
