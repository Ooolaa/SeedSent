import pandas as pd
import numpy as np

# --- Define the file path for the restored data ---
# FIX: Using the deduplicated file path, as that is the final cleaned version
INPUT_FILEPATH = 'brooking_full_text_restored.csv' 
# Using the restored file path as requested in the context of the prompt

def count_and_list_blank_full_text(input_filepath):
    """
    Loads the full text CSV, counts rows where 'Full Text' is missing or empty,
    and lists the Titles of those records.
    """
    
    print(f"Loading data from: {input_filepath}")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 2. Identify Blank Full Text ---
    
    if 'Full Text' not in df.columns:
        print("\nError: The column 'Full Text' was not found in the file.")
        return

    # Fill NaN values with an empty string and check if the stripped length is zero.
    blank_full_text_mask = df['Full Text'].fillna('').str.strip().str.len() == 0
    
    df_missing_content = df[blank_full_text_mask]
    
    total_blank_count = len(df_missing_content)

    # --- 3. Report ---
    print("\n" + "="*50)
    print("BLANK FULL TEXT ANALYSIS")
    print("="*50)
    
    if total_blank_count == 0:
        print("🎉 Great news! All records have content in the 'Full Text' column.")
    else:
        print(f"Total records in file: {len(df)}")
        print(f"Total titles with BLANK or EMPTY 'Full Text': {total_blank_count}\n")
        
        print("--- Titles with Missing Content ---")
        # Print the relevant columns for the missing records
        print(df_missing_content[['Title', 'Authors', 'URL']].to_markdown(index=False))

# --- Run the Analysis ---
count_and_list_blank_full_text(INPUT_FILEPATH)