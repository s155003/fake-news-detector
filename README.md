# Fake News Detector — NLP & Machine Learning

A Python natural language processing project that classifies news articles as real or fake using TF-IDF vectorization and three machine learning models. Beyond raw text, the project engineers linguistic and stylistic features that distinguish fake news from legitimate reporting — things like sensational vocabulary, excessive capitalization, and low lexical diversity. The best model is selected automatically and used to score any article with a fake probability and confidence level.

---

## Overview

Fake news spreads through recognizable patterns. Sensational headlines, all-caps words, excessive punctuation, conspiratorial language, and low vocabulary diversity are all signals that appear far more frequently in misinformation than in legitimate journalism. This project captures those signals through both TF-IDF text vectorization and hand-engineered linguistic features, then trains three classifiers to distinguish real from fake content.

The `predict_article()` function at the bottom of the script means the trained model can be applied to any article text in a single line of code.

---

## How It Works

### 1. Text Preprocessing
Raw article text is cleaned before vectorization:
- Converted to lowercase
- URLs and HTML tags removed
- Numbers stripped
- Punctuation removed
- Extra whitespace normalized

### 2. TF-IDF Vectorization
Text is converted to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) with the following settings:

| Setting | Value | Why |
|---|---|---|
| max_features | 10,000 | Caps vocabulary size to avoid overfitting |
| ngram_range | (1, 2) | Captures single words and two-word phrases |
| sublinear_tf | True | Applies log scaling to term frequencies |

Bigrams are especially useful here — phrases like "deep state", "cover up", "mainstream media", and "share before" are strong fake news signals that single words alone would miss.

### 3. Engineered Text Features
In addition to TF-IDF, the following stylistic features are computed per article:

| Feature | Description |
|---|---|
| char_count | Total character count |
| word_count | Total word count |
| exclamation | Number of exclamation marks |
| question_marks | Number of question marks |
| caps_ratio | Ratio of uppercase characters to total characters |
| unique_words | Count of distinct words used |
| lexical_density | Ratio of unique words to total words |
| sensational_count | Count of sensational/conspiratorial keywords found |

The sensational keyword list includes terms like "bombshell", "exposed", "deep state", "wake up", "they don't want you to know", "share before it's deleted", and similar language patterns common in misinformation.

### 4. Models
Three classifiers are trained inside scikit-learn Pipelines that handle TF-IDF and classification end to end:

| Model | Strengths |
|---|---|
| Logistic Regression | Fast, interpretable, strong text classification baseline |
| Random Forest | Handles feature interactions, robust to noise |
| Gradient Boosting | Highest AUC, captures complex non-linear patterns |

Each model is evaluated on a held-out 20% test set. The best is selected by ROC-AUC.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall percentage of correct classifications |
| ROC-AUC | Model's ability to separate real from fake across all thresholds |
| Average Precision | Area under the precision-recall curve |
| F1 Score | Harmonic mean of precision and recall at the chosen threshold |

---

## Output

Running the script prints a full model evaluation report and generates a 9-panel chart saved as `fake_news_detector.png`:

- **ROC Curves** — all three models with AUC scores
- **Precision-Recall Curves** — all three models with average precision
- **Confusion Matrix** — true/false positives and negatives for the best model
- **Score Distribution** — fake probability histogram split by actual label
- **Model Comparison** — accuracy bar chart across all three models
- **Feature Comparison** — normalized bar chart of engineered features for real vs fake articles
- **Score Boxplot** — distribution of predicted fake probability by actual class
- **Threshold Tuning** — F1 and accuracy plotted across all decision thresholds
- **Class Distribution** — pie chart of dataset balance

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

```bash
python fake_news_detector.py
```

---

## Classifying Your Own Article

Use `predict_article()` to score any text:

```python
text = "BOMBSHELL: Government hiding cure for all cancers to protect pharmaceutical profits SHARE NOW"
predict_article(best_pipeline, text)
```

Output:

```
==================================================
  ARTICLE ANALYSIS
==================================================
  Text:       BOMBSHELL: Government hiding cure for all cancers...
  Fake Score: 0.9412 (94.1%)
  Verdict:    FAKE NEWS
  Confidence: High
==================================================
```

---

## Configuration

```python
RANDOM_SEED  = 42        # Reproducibility
TEST_SIZE    = 0.2       # 80/20 train/test split
MAX_FEATURES = 10000     # TF-IDF vocabulary size
NGRAM_RANGE  = (1, 2)    # Unigrams and bigrams
THRESHOLD    = 0.5       # Decision boundary for fake classification
```

Lowering `THRESHOLD` catches more fake news at the cost of more false positives. Raising it increases precision at the cost of missing some misinformation.

---

## Example Terminal Output

```
Generating dataset...
Dataset: 1,000 articles | Real: 500 | Fake: 500
Train: 800 | Test: 200

==================================================
  MODEL EVALUATION
==================================================

  Logistic Regression
  ───────────────────────────────────────
  Accuracy:  0.9150
  ROC-AUC:   0.9612
  Avg Prec:  0.9588
  F1 Score:  0.9143

  Random Forest
  ───────────────────────────────────────
  Accuracy:  0.9300
  ROC-AUC:   0.9741
  Avg Prec:  0.9719
  F1 Score:  0.9298

  Gradient Boosting
  ───────────────────────────────────────
  Accuracy:  0.9450
  ROC-AUC:   0.9834
  Avg Prec:  0.9811
  F1 Score:  0.9447

  Best model: Gradient Boosting (AUC: 0.9834)
```

---

## Extending the Project

- **Use real datasets** — the [Kaggle Fake News dataset](https://www.kaggle.com/c/fake-news) contains 20,000+ labeled real-world articles and works as a direct drop-in
- **Add BERT embeddings** — replace TF-IDF with sentence transformers for richer semantic representations
- **Headline-only mode** — train a separate lightweight model on just headlines for faster screening
- **Build a browser extension** — wrap `predict_article()` in a Flask API and build a Chrome extension that scores articles as you browse
- **Add source credibility** — incorporate domain reputation scores as an additional feature alongside text content
