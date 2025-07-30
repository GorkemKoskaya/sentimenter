import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# Define custom dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

# Load and prepare data
df = pd.read_csv("C:/Users/GorkemKoskaya/Desktop/Calısmalarım/Sentiment/twitter_training.csv", header=None)
df.columns = ["entity", "entity_type", "sentiment", "text"]
df = df[["text", "sentiment"]]
label_map = {"Positive": 0, "Neutral": 1, "Negative": 2}
df = df[df["sentiment"].isin(label_map.keys())]
df["label"] = df["sentiment"].map(label_map)
df = df.dropna()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset and DataLoader
train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
from sklearn.metrics import classification_report

model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, preds, target_names=["Positive", "Neutral", "Negative"]))

# Save the fine-tuned model and tokenizer
model.save_pretrained("bert-sentiment-model")
tokenizer.save_pretrained("bert-sentiment-model")

from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-sentiment-model")
tokenizer = BertTokenizer.from_pretrained("bert-sentiment-model")

import torch

# Sample sentences for prediction
texts = [
    "Just saw the new Tesla update — absolutely love the self-driving improvements!",
    "Apple's latest keynote was so disappointing, nothing innovative at all.",
    "Microsoft Teams got some updates today. It’s decent but still a bit clunky to use.",
    "Amazon’s customer service handled my issue very efficiently and politely. Great job!",
    "The new policy changes by Facebook are confusing and not well-communicated.",
    "Google is launching something new again. Not sure what to expect this time.",
    "I had a really smooth flight with Delta today. Friendly staff and on-time arrival.",
    "The new update from Spotify just ruined the whole interface, it’s a mess now.",
    "Netflix's interface looks cleaner now. Let’s see if the streaming performance improves too."
]

# Tokenize the inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1)

# Map numeric labels back to sentiment
label_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
predictions = [label_map[label.item()] for label in predicted_classes]

# Print results
for text, pred in zip(texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {pred}\n")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(val_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Neutral", "Negative"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Optional: ROC-AUC (only meaningful in binary or per-class in multi-class)
# You can use OneVsRestClassifier + roc_auc_score if needed (optional for now)



