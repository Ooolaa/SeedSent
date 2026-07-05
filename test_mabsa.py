import ast
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline
from setfit import AbsaModel

# 1. Load the Models
# DeBERTa ABSA Model
deberta_model = "yangheng/deberta-v3-base-absa-v1.1"
deberta_classifier = pipeline("text-classification", model=deberta_model)

# SetFit ABSA Model (Example paths - replace with your local or HF paths)
# For SetFit, you typically need both the aspect and polarity models
setfit_model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)

# 2. Parse M-ABSA Dataset
def parse_mabsa_data(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return []

    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} lines in file. Starting parse...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or "####" not in line:
            continue
            
        parts = line.split("####")
        text = parts[0].strip()
        
        try:
            # M-ABSA uses [[Aspect, Category, Polarity]]
            labels = ast.literal_eval(parts[1].strip())
            
            # Skip lines with empty label lists (like your snippet shows)
            if not labels:
                continue

            for label in labels:
                # format: [aspect, category, sentiment]
                parsed_data.append({
                    "text": text,
                    "aspect": label[0],
                    "true_sentiment": label[2].lower()
                })
        except Exception as e:
            print(f"Error parsing line {i}: {e}")
            
    print(f"Successfully parsed {len(parsed_data)} aspect-level samples.")
    return parsed_data

# --- UPDATE THIS PATH TO YOUR ACTUAL FILE LOCATION ---
file_path = "M-ABSA-main/data/coursera/en/train.txt"
test_data = parse_mabsa_data(file_path)

# 3. Evaluation Logic
def evaluate_models(data):
    y_true = []
    y_pred_deberta = []
    y_pred_setfit = []

    for item in tqdm(data):
        text = item['text']
        aspect = item['aspect']
        y_true.append(item['true_sentiment'])

        # DeBERTa Inference
        # Format: classifier(sentence, text_pair="aspect")
        deb_res = deberta_classifier(text, text_pair=aspect)
        y_pred_deberta.append(deb_res[0]['label'].lower())

        # SetFit Inference
        # predict() returns a list of dicts: [{'span': '...', 'polarity': '...'}]
        sf_res = setfit_model.predict([text])[0]
        
        pred_label = "neutral" # Default fallback
        for pred in sf_res:
            # CHECK: Using 'polarity' instead of 'label'
            # We also check if the predicted span matches our ground truth aspect
            if pred['span'].lower() in aspect.lower() or aspect.lower() in pred['span'].lower():
                pred_label = pred['polarity'].lower() 
                break
        y_pred_setfit.append(pred_label)

    return y_true, y_pred_deberta, y_pred_setfit

# Execution
raw_content = """M-ABSA-main/eval_baseline_mT5/data/coursera/en/train.txt"""
test_data = parse_mabsa_data(raw_content)
y_true, y_deb, y_sf = evaluate_models(test_data)

# 4. Display Results
def print_metrics(name, true, pred):
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy_score(true, pred):.4f}")
    print(f"F1-Score (Macro): {f1_score(true, pred, average='macro'):.4f}")

print_metrics("DeBERTa-v3-base-ABSA", y_true, y_deb)
print_metrics("SetFit-ABSA", y_true, y_sf)