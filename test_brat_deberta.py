import os
# Workaround for the protobuf error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

# 1. Load the Model using Transformers Pipeline
model_id = "yangheng/deberta-v3-base-absa-v1.1"
classifier = pipeline("text-classification", model=model_id)

def load_brat_data(txt_path, ann_path):
    if not os.path.exists(txt_path) or not os.path.exists(ann_path):
        print(f"ERROR: BRAT files not found at {txt_path} or {ann_path}")
        return [], {}, {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_content = f.read()
        
    terms = {}
    # Regex to capture Term ID and the text
    for match in re.finditer(r'(T\d+)\tTerm\s+(\d+)\s+(\d+)\t(.*)', ann_content):
        t_id = match.group(1)
        terms[t_id] = {'text': match.group(4)}
        
    polarities = {}
    # Regex to capture Polarity linked to Term ID
    for match in re.finditer(r'A\d+\tPolarity\s+(T\d+)\s+(.*)', ann_content):
        polarities[match.group(1)] = match.group(2)

    return sentences, terms, polarities

def evaluate_deberta_brat(sentences, terms, polarities):
    y_true = []
    y_pred = []

    print("\n--- Running DeBERTa-v3 (Pipeline) on BRAT Data ---")

    for t_id, info in tqdm(terms.items()):
        if t_id in polarities:
            gold_label = polarities[t_id].lower()
            aspect_text = info['text']
            
            # Simple context matching: find sentence containing the aspect
            target_sentence = next((s for s in sentences if aspect_text in s), None)
            
            if target_sentence:
                # Format: classifier("Sentence", text_pair="Aspect")
                result = classifier(target_sentence, text_pair=aspect_text)
                pred_label = result[0]['label'].lower()
                
                y_true.append(gold_label)
                y_pred.append(pred_label)

    return y_true, y_pred

# 3. Execution
# Point to your subfolder
txt_file = 'brat_format/foursquare_sentences.txt'
ann_file = 'brat_format/foursquare_sentences.ann'

# Load and Evaluate
sents, tms, pols = load_brat_data(txt_file, ann_file)
y_true, y_pred = evaluate_deberta_brat(sents, tms, pols)

# 4. Display Metrics
if y_true:
    print(f"\n--- DeBERTa-v3-base-ABSA Results ---")
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")