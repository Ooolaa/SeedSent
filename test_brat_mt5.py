import sys
import os
import re
import torch
from tqdm import tqdm
from transformers import T5Tokenizer

# Add path to the source code folder
sys.path.append("/Users/johnchen/Desktop/Programming/Sentiment Analysis/M-ABSA-main/eval_baseline_mT5")

from main import T5FineTuner
# Note: evaluate in main.py requires specific args, we will use a simplified inference loop here

def load_brat_as_mabsa(txt_path, ann_path):
    """Converts BRAT files into a format mT5 understands."""
    if not os.path.exists(txt_path):
        return []
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_content = f.read()

    terms = {}
    for m in re.finditer(r'(T\d+)\tTerm\s+\d+\s+\d+\t(.*)', ann_content):
        terms[m.group(1)] = m.group(2)
    pols = {}
    for m in re.finditer(r'A\d+\tPolarity\s+(T\d+)\s+(.*)', ann_content):
        pols[m.group(1)] = m.group(2)

    mabsa_data = []
    for t_id, aspect in terms.items():
        if t_id in pols:
            sentiment = pols[t_id].lower()
            for s in sentences:
                if aspect in s:
                    mabsa_data.append({"sent": s, "aspect": aspect, "gold": sentiment})
                    break
    return mabsa_data

# --- Execution ---
# 1. Load Model & Tokenizer
# Update this to your 'first.ckpt' path
ckpt_path = "/Users/johnchen/Desktop/Programming/Sentiment Analysis/M-ABSA-main/eval_baseline_mT5/outputs/uabsa/coursera/extraction/first.ckpt"

print(f"Loading mT5 Checkpoint...")
model = T5FineTuner.load_from_checkpoint(ckpt_path)
model.to("mps" if torch.backends.mps.is_available() else "cpu")
model.eval()

tokenizer = T5Tokenizer.from_pretrained(model.hparams.model_name_or_path)

# 2. Load BRAT Data
# Paths relative to where you run the script
data = load_brat_as_mabsa('brat_format/foursquare_sentences.txt', 'brat_format/foursquare_sentences.ann')

# 3. Inference Loop
print(f"Starting Evaluation on {len(data)} samples...")
correct = 0
total = 0

for item in tqdm(data):
    # mT5 for extraction usually expects the raw sentence
    inputs = tokenizer(item['sent'], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.model.generate(inputs["input_ids"], max_length=128)
    
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check if the gold sentiment is contained in the generated triplet (e.g., "(coffee, negative)")
    if item['gold'].lower() in pred_text.lower():
        correct += 1
    total += 1

if total > 0:
    print(f"\n--- mT5 Results on BRAT ---")
    print(f"Accuracy: {correct/total:.4f}")