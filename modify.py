import os
input_file = 'test_hu.txt'
output_file = 'test_hu_new.txt' # Create a new file for safety

print(f"Formatting {input_file}...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        line = line.strip()
        if line:
            # Append '#### []' to represent an empty label list
            outfile.write(f"{line}#### []\n")

print(f"Formatting complete. Output saved to {output_file}.")

# Now, rename the formatted file to test.txt
os.replace(output_file, input_file)
print(f"Replaced {input_file} with the formatted content.")