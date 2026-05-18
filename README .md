# Phishing Email Detection Model

A machine learning classifier that detects phishing emails using **Naive Bayes** and **TF-IDF** feature extraction. The model is trained on labelled email data and classifies incoming emails as **Phishing** or **Safe** with confidence scores.

---

## Overview

| Property | Detail |
|---|---|
| Algorithm | Multinomial Naive Bayes |
| Feature extraction | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Training set | 40 labelled emails (20 phishing, 20 safe) |
| Train/test split | 72% / 28% |
| Output | Binary classification: Phishing / Safe + confidence score |

---

## Features

- Tokenises email subject and body text
- Converts URLs to a `URLTOKEN` sentinel before feature extraction
- Builds a vocabulary from training data with TF-IDF weighting
- Trains a Naive Bayes model with Laplace (add-α) smoothing
- Evaluates on a held-out test set with accuracy, precision, recall, and F1
- Displays a confusion matrix and ROC curve
- Exposes a live inference interface for testing arbitrary emails
- Shows top phishing vs. safe word signals with importance scores

---

## How It Works

### 1. Preprocessing

Each email (subject + body) is lowercased, URLs are replaced with the token `URLTOKEN`, punctuation is stripped, and the text is split into tokens of length > 2.

```
"Click http://phish.xyz to verify" → ["click", "URLTOKEN", "verify"]
```

### 2. TF-IDF Feature Extraction

Each token gets a TF-IDF score:

```
TF(t, d)  = count(t in d) / total tokens in d
IDF(t)    = log((N + 1) / (df(t) + 1)) + 1
TF-IDF    = TF × IDF
```

where `N` is the number of documents and `df(t)` is the number of documents containing term `t`.

### 3. Naive Bayes Classifier

A generative model is trained per class (phishing = 1, safe = 0):

```
log P(class | email) ∝ log P(class) + Σ TF-IDF(t) × log P(t | class)
```

Laplace smoothing (`α = 0.5`) is applied to handle unseen terms. At inference, softmax is applied over the two class log-scores to produce calibrated probabilities.

### 4. Prediction

```
predicted class = argmax P(class | email)
confidence      = max(P(phishing), P(safe))
```

---

## Key Phishing Signals

The model learns to weight these features heavily for phishing detection:

| Indicator | Type |
|---|---|
| `URLTOKEN` (any URL in the email) | Structural |
| `verify`, `suspended`, `urgent` | Urgency keywords |
| `password`, `credit`, `bank`, `account` | Sensitive data requests |
| `click`, `immediately`, `login`, `confirm` | Action prompts |
| `free`, `won`, `claim`, `limited` | Lure keywords |

---

## Model Metrics

Evaluated on a 28% held-out test set:

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of emails classified correctly |
| **Precision** | Of emails flagged as phishing, how many actually were |
| **Recall** | Of all phishing emails, how many were caught |
| **F1 Score** | Harmonic mean of precision and recall |

A confusion matrix and ROC curve are displayed in the interactive UI.

---

## Project Structure

```
phishing-detector/
├── index.html          # Interactive browser-based UI
├── README.md           # This file
└── data/
    └── emails.js       # Labelled training dataset (40 emails)
```

---

## Scaling to Production (Python / Scikit-learn)

The browser implementation mirrors a standard Scikit-learn pipeline. To scale it up:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
X = [email['subject'] + ' ' + email['body'] for email in emails]
y = [email['label'] for email in emails]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),       # unigrams + bigrams
        max_features=10000,
        stop_words='english',
        sublinear_tf=True
    )),
    ('clf', MultinomialNB(alpha=0.5))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds, target_names=['Safe', 'Phishing']))
print(confusion_matrix(y_test, preds))
```

### Recommended datasets for production training

| Dataset | Size | Source |
|---|---|---|
| SpamAssassin Public Corpus | ~6,000 emails | spamassassin.apache.org |
| Enron Email Dataset | ~500,000 emails | cs.cmu.edu/~enron |
| TREC Spam Corpus | ~75,000 emails | plg.uwaterloo.ca |

Training on these datasets typically pushes accuracy above **97%**.

### Additional features to extract

- Sender domain reputation (SPF/DKIM fail flags)
- Number of URLs per email
- Presence of IP-based URLs (e.g. `http://192.168.1.1/login`)
- HTML-to-text ratio
- Number of misspellings in the domain name
- Reply-to vs. from address mismatch

---

## Dependencies

### Browser version (this project)
- No dependencies — pure JavaScript, runs in any modern browser
- Chart.js 4.4.1 (loaded from CDN for visualisations)

### Python version (production)
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0   # for confusion matrix plots
seaborn>=0.12.0     # optional, for styled heatmaps
```

Install with:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Limitations

- The browser model trains on 40 emails — sufficient for demonstration but not production use
- Naive Bayes assumes feature independence, which is not strictly true for natural language
- Adversarial phishing emails that avoid known keywords may evade detection
- The model does not analyse email headers, sender reputation, or HTML structure

---

## License

MIT License — free to use, modify, and distribute.
