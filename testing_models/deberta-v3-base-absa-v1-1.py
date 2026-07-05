import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm # Useful for progress tracking

def extract_data_and_label(line):
    """
    Separates the raw text from the sentiment label.
    Example label format: ####[['aspect', 'category', 'positive']]
    """
    # Extract raw text by removing source tags and the metadata block
    # clean_text = re.sub(r'\', '', line)
    clean_text = re.sub(r'\\', '', line)
    clean_text = re.sub(r'####\[\[.*\]\]', '', clean_text).strip()
    
    # Extract the sentiment label using regex to find the last element in the list
    # It looks for 'positive', 'negative', or 'neutral' inside the brackets
    label_match = re.findall(r"'(positive|negative|neutral)'", line)
    
    # If multiple aspects exist, we'll take the first one for overall sentence accuracy
    label = label_match[0] if label_match else None
    
    return clean_text, label

def run_absa_accuracy_test(file_path):
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    
    print(f"Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Map for comparison
    # Model output: 0: Negative, 1: Neutral, 2: Positive
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    
    correct_predictions = 0
    total_valid_entries = 0

    print(f"Processing entire dataset: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        # Skip JSON formatting and headers
        if line.strip().startswith(('{', '}', 'type:', 'fileName:', 'fullContent:')) or not line.strip():
            continue
            
        text, true_sentiment = extract_data_and_label(line)
        
        if not text or not true_sentiment:
            continue

        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Check accuracy
        if prediction == label_map[true_sentiment]:
            correct_predictions += 1
        
        total_valid_entries += 1

    if total_valid_entries > 0:
        accuracy = (correct_predictions / total_valid_entries) * 100
        print(f"\n--- Results ---")
        print(f"Total Sentences Tested: {total_valid_entries}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Average Accuracy: {accuracy:.2f}%")
    else:
        print("No valid data entries found to test.")

if __name__ == "__main__":
    # Adjusted path based on your folder structure
    DATA_PATH = '../M-ABSA-main/data/coursera/en/test.txt'
    run_absa_accuracy_test(DATA_PATH)