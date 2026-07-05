import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# --- NEW: Added the missing data loader function ---
def load_twitter_data(file_path):
    """Parses the 3-line Twitter format: Text, Target, Label."""
    features = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
            # Iterate through the file 3 lines at a time
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                    
                sentence = lines[i]
                target = lines[i+1]
                label_num = lines[i+2]
                
                # Map Twitter numbers to your model's string labels
                # ACL format: -1=neg, 0=neu, 1=pos
                label_map = {"-1": "negative", "0": "neutral", "1": "positive"}
                
                # Replace $T$ with the actual target for the DeBERTa model
                processed_text = sentence.replace("$T$", target)
                
                features.append({
                    "text": processed_text,
                    "aspect": target,
                    "label": label_map.get(label_num, "neutral")
                })
        return features
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

# 1. Setup Model
model_path = "./deberta-coursera-finetuned/checkpoint-750"
tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Updated mapping based on your previous 'KeyError: 3'
# LabelEncoder sorts alphabetically: conflict (0), negative (1), neutral (2), positive (3)
id2label = {0: "conflict", 1: "negative", 2: "neutral", 3: "positive"}

# 2. Load the Twitter data
raw_data = load_twitter_data('acl_14.txt')

if not raw_data:
    print("No data loaded. Check if acl_14.txt exists in the correct folder.")
else:
    y_true = []
    y_pred = []

    print(f"{'Target':<15} | {'Ground Truth':<12} | {'Model Prediction'}")
    print("-" * 50)

    for item in raw_data:
        # DeBERTa expects (Text, Aspect) pair
        inputs = tokenizer(
            item["text"], 
            item["aspect"], 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        prediction = id2label[pred_idx]
        
        y_true.append(item["label"])
        y_pred.append(prediction)
        
        # Print the first 10 for verification
        if len(y_true) <= 10:
            print(f"{item['aspect'][:15]:<15} | {item['label']:<12} | {prediction}")

    # 3. Calculate Final Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy on Twitter Dataset: {accuracy:.2%}")