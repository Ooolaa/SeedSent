import pandas as pd
import os

# --- Define the file paths using the location from the previous step ---
INPUT_FILEPATH = '/Users/johnchen/Desktop/Programming/Sentiment Analysis/brooking_full_text_restored.csv'
OUTPUT_FILEPATH = '/Users/johnchen/Desktop/Programming/Sentiment Analysis/brooking_full_text_deduplicated.csv'

def remove_duplicates_and_report(input_filepath, output_filepath):
    """
    Loads a CSV, checks for duplicates based on Title and URL, removes them,
    and reports the count of deleted rows.
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

    initial_count = len(df)
    
    # --- 2. Check and Remove Duplicates ---
    
    # Define the columns that must be unique. URL is the most reliable.
    # We use 'Title' and 'URL' together for robust identification.
    deduplication_keys = ['Title', 'URL']
    
    # Drop duplicates, keeping the first occurrence found.
    # We use subset to target specific columns for the check.
    df_clean = df.drop_duplicates(subset=deduplication_keys, keep='first')
    
    final_count = len(df_clean)
    duplicates_removed = initial_count - final_count

    # --- 3. Save Cleaned Data ---
    df_clean.to_csv(output_filepath, index=False, encoding='utf-8-sig')

    # --- 4. Report ---
    print("\n" + "="*50)
    print("DEDUPLICATION COMPLETE")
    print("="*50)
    print(f"Initial total records: {initial_count}")
    print(f"Duplicates found and removed: {duplicates_removed}")
    print(f"Final unique records: {final_count}")
    print(f"\nCleaned data saved to: {output_filepath}")

# --- Run the Deduplication ---
remove_duplicates_and_report(INPUT_FILEPATH, OUTPUT_FILEPATH)