import os
import ast
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# 1. Configuration
model_name = "yangheng/deberta-v3-base-absa-v1.1"
train_path = "M-ABSA-main/data/coursera/en/train.txt"
dev_path = "M-ABSA-main/data/coursera/en/dev.txt"
test_path = "M-ABSA-main/data/coursera/en/test.txt"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Parse M-ABSA format to Classification format
def load_mabsa_data(file_path):
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
                    for aspect, category, sentiment in labels:
                        features.append({
                            "text": text,
                            "aspect": aspect,
                            "label": sentiment.lower()
                        })
                except Exception as e:
                    continue
    return features

# 3. Data Preparation
train_raw = load_mabsa_data(train_path)
dev_raw = load_mabsa_data(dev_path)
test_raw = load_mabsa_data(test_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Automatically detect all labels in both sets to avoid "unseen label" errors
all_sentiment_labels = sorted(list(set([d['label'] for d in train_raw] + [d['label'] for d in dev_raw])))
# num_labels = len(all_sentiment_labels)
label_encoder = LabelEncoder()
label_encoder.fit(all_sentiment_labels)
num_labels = len(label_encoder.classes_)
print(f"Detected Labels: {label_encoder.classes_}")

def preprocess_function(examples):
    # This prepares the Sentence [SEP] Aspect pair for DeBERTa
    return tokenizer(examples["text"], examples["aspect"], 
                     truncation=True, padding="max_length", max_length=128)

def prepare_dataset(raw_data):
    ds = Dataset.from_list(raw_data)
    # Convert labels to integers using the encoder
    ds = ds.map(lambda x: {"labels": int(label_encoder.transform([x["label"]])[0])}, remove_columns=["label"])
    return ds.map(preprocess_function, batched=True)

train_dataset = prepare_dataset(train_raw)
eval_dataset = prepare_dataset(dev_raw)
test_dataset = prepare_dataset(test_raw)

# 4. Metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

# 5. Initialize Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels, 
    ignore_mismatched_sizes=True
)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta-coursera-finetuned",
    eval_strategy="epoch",            # FIXED: renamed from evaluation_strategy
    save_strategy="epoch",            
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,               
    weight_decay=0.01,
    load_best_model_at_end=True,      
    metric_for_best_model="f1_macro",
    push_to_hub=False,
    # use_mps_device is also deprecated in favor of auto-detection or 'device'
    logging_steps=10,
    report_to="none"                  # Prevents errors if wandb is not installed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,        
    tokenizer=tokenizer,
    compute_metrics=compute_metrics    
)

# 7. Run Training
print("\n--- Starting Fine-tuning ---")
trainer.train()

# Save final best model
model.save_pretrained("./deberta-coursera-finetuned")
tokenizer.save_pretrained("./deberta-coursera-finetuned")
print("\nTraining complete. Best model saved to ./deberta-coursera-finetuned")

# 7. Run Training
print("\n--- Starting Fine-tuning ---")
trainer.train()

# --- ADD THIS PART ---
print("\n--- Final Test Results ---")
test_results = trainer.evaluate(test_dataset)
print(test_results)
# --------------------

# Save final best model
model.save_pretrained("./deberta-coursera-finetuned")
tokenizer.save_pretrained("./deberta-coursera-finetuned")