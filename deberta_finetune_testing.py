import os
import ast
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Configuration
model_path = "./deberta-coursera-finetuned"
test_path = "M-ABSA-main/data/coursera/en/test.txt"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Parse Test Data
def load_mabsa_test_data(file_path):
    features = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '####' in line:
                parts = line.strip().split('####')
                text = parts[0]
                try:
                    labels = ast.literal_eval(parts[1])
                    if not labels:
                        continue
                    for aspect, category, sentiment in labels:
                        features.append({
                            "text": text,
                            "aspect": aspect,
                            "label": sentiment.lower()
                        })
                except:
                    continue
    return features

# 3. Load Model and Tokenizer
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

label_encoder = LabelEncoder()
label_encoder.fit(['conflict', 'negative', 'neutral', 'positive'])

# 4. Prepare and Clean Dataset
test_raw = load_mabsa_test_data(test_path)
test_dataset = Dataset.from_list(test_raw)

def preprocess_test(examples):
    return tokenizer(examples["text"], examples["aspect"], 
                     truncation=True, padding="max_length", max_length=128)

# Key Fix: Map and remove original text columns immediately
tokenized_test = test_dataset.map(preprocess_test, batched=True, remove_columns=test_dataset.column_names)

# Map labels separately (since we removed 'label' in the previous step, we use the raw data list)
test_labels = [int(label_encoder.transform([x["label"]])[0]) for x in test_raw]
tokenized_test = tokenized_test.add_column("labels", test_labels)

# 5. Define Evaluation Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

# 6. Run Evaluation using Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n--- Running Final Evaluation on test.txt ---")
# evaluate() automatically handles batching once columns are clean
results = trainer.evaluate(tokenized_test)

# 7. Detailed Report
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = tokenized_test["labels"]

print("\nFinal Metrics:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# DYNAMIC FIX: 
# Get the unique label indices actually present in your model's predictions and true labels
present_label_indices = np.unique(np.concatenate((true_labels, preds)))
# Filter the class names to match only the labels the model actually knows
target_names = [label_encoder.classes_[i] for i in present_label_indices]

print("\nDetailed Classification Report:")
print(classification_report(
    true_labels, 
    preds, 
    labels=present_label_indices, 
    target_names=target_names
))