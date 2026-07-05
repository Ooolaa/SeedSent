import torch
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# 1. Load Model and Tokenizer
# Based on your error, you are using checkpoint-750. 
# Make sure this matches the path in your screenshot.
model_path = "./deberta-coursera-finetuned/checkpoint-750"

# Fix for the Mistral/DeBERTa tokenizer regex warning
tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2. Update Label Mapping
# Based on your 'KeyError: 3', you have 4 classes. 
# LabelEncoder sorts alphabetically. The mapping is almost certainly:
# 0: conflict, 1: negative, 2: neutral, 3: positive
id2label = {0: "conflict", 1: "negative", 2: "neutral", 3: "positive"}

def get_sentiment(text, aspect):
    # Added max_length=128 to match your training configuration 
    # and resolve the "no maximum length" warning.
    inputs = tokenizer(
        text, 
        aspect, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[prediction]

# 3. Parse XML and Predict
# Use the file you uploaded: 'Laptops Train.xml'
tree = ET.parse('Laptops Train.xml')
root = tree.getroot()

y_true = []
y_pred = []

print(f"{'Aspect':<20} | {'True':<10} | {'Predicted':<10}")
print("-" * 45)

for sentence in root.findall('sentence'):
    text_node = sentence.find('text')
    if text_node is None or text_node.text is None:
        continue
    text = text_node.text
    
    aspect_terms = sentence.find('aspectTerms')
    if aspect_terms is not None:
        for aspect in aspect_terms.findall('aspectTerm'):
            term = aspect.get('term')
            true_polarity = aspect.get('polarity')
            
            # Predict
            prediction = get_sentiment(text, term)
            
            y_true.append(true_polarity)
            y_pred.append(prediction)
            
            print(f"{term[:20]:<20} | {true_polarity:<10} | {prediction:<10}")

# 4. Calculate Accuracy
if y_true:
    acc = accuracy_score(y_true, y_pred)
    print(f"\nFinal Accuracy on XML Data: {acc:.2%}")
else:
    print("No aspect terms found to evaluate.")