import pandas as pd
import re

# 1. Load the CSV file
df = pd.read_csv("Final_Hudson.csv")

# 2. Extract the content of the first article (first row, 'Content' column)
try:
    first_article_content = df.loc[0, 'Content']
except KeyError:
    # Fallback in case column name is different
    content_column = df.columns[-1]
    first_article_content = df.loc[0, content_column]

# 3. Split the content into sentences
# Using a regex-based splitter as it is a reliable method for sentence segmentation
# that works well on English text without external resource downloads (like NLTK's punkt).
# It splits on a period, question mark, or exclamation mark followed by whitespace,
# while attempting to retain the punctuation with the sentence.
sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', first_article_content)

# 4. Clean and save the sentences to a new file
output_filename = "preprocessed_first_article_sentences.txt"
cleaned_sentences = [s.strip() for s in sentences if s.strip()]

# Write the sentences to the output file, one sentence per line
with open(output_filename, 'w', encoding='utf-8') as f:
    for sentence in cleaned_sentences:
        f.write(sentence + '\n')