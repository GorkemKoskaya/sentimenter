# 🧠 Twitter Entity Sentiment Analysis with BERT

This project focuses on fine-tuning a BERT-based model for sentiment classification using Twitter data.  
The objective is to classify user-generated tweets into **Positive**, **Neutral**, or **Negative** sentiment classes.

---

## 📘 Table of Contents

- [Dataset Overview](#-dataset-overview)
- [Preprocessing](#-preprocessing)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [Evaluation Metrics](#-evaluation-metrics)
- [Confusion Matrix](#-confusion-matrix)
- [Sample Predictions](#-sample-predictions)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Author](#-author)

---

## 📂 Dataset Overview

- **Source:** [Kaggle - Twitter Entity Sentiment](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Size:** 75,000+ labeled tweets
- **Columns used:**
  - `text`: Tweet content
  - `sentiment`: Label (Positive, Neutral, Negative)

After filtering:
- Null entries removed from the `text` column
- Label mapping:
  - `Positive` → `0`
  - `Neutral` → `1`
  - `Negative` → `2`

---

## 🧹 Preprocessing

- Removed NaN values
- Used `bert-base-uncased` tokenizer
- Split into 80% training / 20% validation
- Encoded target sentiment classes into integers
- Converted text into token IDs and attention masks for BERT input

---

## 🧠 Model Architecture


- Base Model: `bert-base-uncased`
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Evaluation Metric: F1-Score

---

## 🏋️‍♀️ Training Details

- Library: Hugging Face Transformers
- Hardware: GPU recommended (e.g. Google Colab or CUDA-enabled machine)
- Epochs: 3 (can be tuned)
- Batch Size: 16
- Scheduler: Linear Warmup
- Evaluation Strategy: Epoch-based

---

## 📊 Evaluation Metrics

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Positive  | 0.94      | 0.91   | 0.93     | 4120    |
| Neutral   | 0.93      | 0.93   | 0.93     | 3678    |
| Negative  | 0.92      | 0.94   | 0.93     | 4427    |
| **Overall Accuracy** |        |        | **0.93** | **12225** |

---

## 📈 Confusion Matrix

🖼️ **Insert confusion matrix plot below:**  
📌
<img width="709" height="533" alt="Karmaşıklık Matrisi" src="https://github.com/user-attachments/assets/2ab92fb5-26ae-4d00-a4ed-cfbb9645b222" />

