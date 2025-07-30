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

## 🔮 Sample Predictions

```text
Text: Just saw the new Tesla update — absolutely love the self-driving improvements!
Predicted Sentiment: Positive ✅

Text: Apple's latest keynote was so disappointing, nothing innovative at all.
Predicted Sentiment: Negative ✅

Text: Microsoft Teams got some updates today. It’s decent but still a bit clunky to use.
Predicted Sentiment: Neutral ✅

Text: Amazon’s customer service handled my issue very efficiently and politely. Great job!
Predicted Sentiment: Positive ✅

Text: The new policy changes by Facebook are confusing and not well-communicated.
Predicted Sentiment: Negative ✅

Text: Google is launching something new again. Not sure what to expect this time.
Predicted Sentiment: Neutral ✅

Text: I had a really smooth flight with Delta today. Friendly staff and on-time arrival.
Predicted Sentiment: Positive ✅

Text: The new update from Spotify just ruined the whole interface, it’s a mess now.
Predicted Sentiment: Negative ✅

Text: Netflix's interface looks cleaner now. Let’s see if the streaming performance improves too.
Predicted Sentiment: Neutral ✅
```

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/gorkemkoskaya/twitter-sentiment-bert
cd twitter-sentiment-bert

# Install dependencies
pip install transformers[torch] accelerate pandas scikit-learn matplotlib

# Run training
python train.py
```

---

## 📦 Dependencies

```text
transformers >= 4.26.0
accelerate >= 0.26.0
torch
pandas
scikit-learn
matplotlib
```

---

## 🌱 Future Work

```text
- Experiment with other Transformer models (DistilBERT, RoBERTa)
- Add SHAP explanations for interpretability
- Deploy as a FastAPI/Flask API
- Extend to multilingual sentiment classification
```

---

## 🧾 Citation

```text
Dataset: Kaggle - Twitter Entity Sentiment (https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
Model: Hugging Face Transformers (https://huggingface.co/transformers/)
```

---

## 👨‍💻 Author

```text
Developed by [Your Name]

If you use this project, consider citing or starring the repo ⭐
Pull requests are welcome!
```

---

## 📈 Confusion Matrix

```text
# 🔽 Replace this with the actual confusion matrix plot
📌 Here
```
