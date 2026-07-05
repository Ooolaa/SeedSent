import pandas as pd
import numpy as np

# --- Define the file paths using the locations you provided ---
REFERENCE_FILEPATH = '/Users/johnchen/Desktop/Programming/Sentiment Analysis/brooking_no_full_text.csv'
CORRUPTED_FILEPATH = '/Users/johnchen/Desktop/Programming/Sentiment Analysis/brooking_full_text.csv'
OUTPUT_FILEPATH = '/Users/johnchen/Desktop/Programming/Sentiment Analysis/brooking_full_text_restored.csv'

def restore_full_text_data(reference_filepath, corrupted_filepath, output_filepath):
    """
    Loads two CSV files, uses the accurate Title/Metadata from the reference file
    to restore/match records in the corrupted full text file, and saves the result.
    
    Args:
        reference_filepath (str): The summary file with the correct metadata.
        corrupted_filepath (str): The full text file with potentially truncated Titles.
        output_filepath (str): The path to save the new, corrected file.
    """
    
    print(f"Loading reference file: {reference_filepath}")
    print(f"Loading corrupted file: {corrupted_filepath}")

    # --- 1. Load Data ---
    try:
        df_reference = pd.read_csv(reference_filepath)
        df_corrupted = pd.read_csv(corrupted_filepath)
    except FileNotFoundError as e:
        print(f"Error: File not found at {e.filename}")
        return
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # --- 2. Normalize and Create Merge Keys ---
    
    # Define columns to use for matching (excluding the potentially corrupted 'Title')
    merge_keys = ['Authors', 'Date', 'URL']
    
    # Ensure all key columns are strings and strip whitespace/quotes
    for df in [df_reference, df_corrupted]:
        for col in ['Title', 'Authors', 'Date', 'URL']:
            if col in df.columns:
                # Note: We must handle potential NaN values by converting to string first
                df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False)

    # Convert all merge keys to lowercase for case-insensitive matching
    # This creates a unique identifier from the Authors, Date, and URL for matching rows.
    df_reference['merge_key'] = df_reference[merge_keys].apply(lambda row: '||'.join(row.values).lower(), axis=1)
    df_corrupted['merge_key'] = df_corrupted[merge_keys].apply(lambda row: '||'.join(row.values).lower(), axis=1)
    
    # --- 3. Perform Left Merge for Restoration ---
    
    # Select only the correct Title from the reference file and the merge key
    df_titles = df_reference[['Title', 'merge_key']].rename(columns={'Title': 'Correct_Title'})

    # Merge the corrected titles back into the corrupted full text data
    # We use 'left' merge to ensure we keep all rows from the original full text file.
    df_merged = pd.merge(
        df_corrupted,
        df_titles,
        on='merge_key',
        how='left'
    )

    # --- 4. Apply Restoration ---
    
    # Replace the existing (corrupted) 'Title' column with the 'Correct_Title' 
    # if a match was found (i.e., 'Correct_Title' is not NaN).
    df_merged['Title_Restored'] = np.where(
        df_merged['Correct_Title'].notna(),
        df_merged['Correct_Title'],
        df_merged['Title'] # Keep original corrupted title if no match was found (e.g., if the row was a complete mismatch)
    )
    
    # --- 5. Final Cleanup and Save ---

    # Drop temporary columns and the original 'Title' column
    df_final = df_merged.drop(columns=['Title', 'merge_key', 'Correct_Title']).rename(
        columns={'Title_Restored': 'Title'}
    )
    
    # Reorder columns to ensure 'Title' is the first column
    # Get all column names except 'Title'
    other_cols = [col for col in df_final.columns if col != 'Title']
    # Create the final column order: Title, then all others
    cols = ['Title'] + other_cols
    df_final = df_final[cols]

    # Save the new file
    df_final.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    
    # Report on changes
    # Count how many rows had a successful correction (i.e., the title actually changed)
    restored_count = (df_merged['Correct_Title'].notna() & (df_merged['Title'] != df_merged['Correct_Title'])).sum()
    
    print("\n" + "="*50)
    print("RESTORATION COMPLETE")
    print("="*50)
    print(f"Total records processed: {len(df_corrupted)}")
    print(f"Titles restored/corrected: {restored_count}")
    print(f"The new file is saved at: {output_filepath}")

# --- Run the Restoration using the user-provided paths ---
restore_full_text_data(
    REFERENCE_FILEPATH, 
    CORRUPTED_FILEPATH, 
    OUTPUT_FILEPATH
)