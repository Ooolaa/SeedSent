import os
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from setfit import AbsaModel

# 1. Load the Model
# You can use a pre-trained restaurant model since your data is Foursquare (food/places)
model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect",
    "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
)

def load_brat_data(txt_path, ann_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_content = f.read()
        
    terms, polarities = {}, {}
    for match in re.finditer(r'(T\d+)\tTerm\s+(\d+)\s+(\d+)\t(.*)', ann_content):
        terms[match.group(1)] = {'text': match.group(4)}
    for match in re.finditer(r'A\d+\tPolarity\s+(T\d+)\s+(.*)', ann_content):
        polarities[match.group(1)] = match.group(2)
    return sentences, terms, polarities

def evaluate_setfit_brat(sentences, terms, polarities):
    y_true, y_pred = [], []

    print("\n--- Running SetFit-ABSA on BRAT Data ---")
    for t_id, info in tqdm(terms.items()):
        if t_id in polarities:
            gold_label = polarities[t_id].lower()
            aspect_text = info['text']
            
            # Find context sentence
            target_sentence = next((s for s in sentences if aspect_text in s), None)
            
            if target_sentence:
                # SetFit predicts ALL aspects in a sentence
                preds = model.predict([target_sentence])[0]
                
                # Logic: Find the prediction that matches our gold aspect text
                found_match = False
                for p in preds:
                    if p['span'].lower() == aspect_text.lower():
                        y_pred.append(p['polarity'].lower())
                        y_true.append(gold_label)
                        found_match = True
                        break
                
                # If SetFit didn't find the aspect, we skip or mark as miss
                if not found_match:
                    continue 

    return y_true, y_pred

# 3. Execution
txt_file = 'brat_format/foursquare_sentences.txt'
ann_file = 'brat_format/foursquare_sentences.ann'

y_true, y_pred = evaluate_setfit_brat(*load_brat_data(txt_file, ann_file))

if y_true:
    print(f"\n--- SetFit-ABSA Results ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")