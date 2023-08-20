import pandas as pd
import re
import matplotlib.pyplot as plt

# Load the input CSV file into a pandas DataFrame
input_csv = './generated_data/merged_data.csv'
df = pd.read_csv(input_csv).dropna()

# regex application
# Define the regex pattern for special characters
special_chars_pattern = r'[^a-zA-Z0-9\s\/-]'

# Apply str.replace() to each column
text_columns = ['x_title', 'cat_1', 'cat_2', 'cat_3']
for col in text_columns:
    df[col] = df[col].apply(lambda x: re.sub(special_chars_pattern, '', x))
    df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    


output_regex_csv = './generated_data/regex_cleaned.csv'
df.to_csv(output_regex_csv, index=False)    
    
# Create a new DataFrame for the output CSV
output_data = []

print(f"cleaning the dataset")

# Iterate through each row in the input DataFrame
for index, row in df.iterrows():
    # Extract values from the current row
    x_title = row['x_title']
    cat_1 = row['cat_1']
    cat_2 = row['cat_2']
    cat_3 = row['cat_3']
    
    # Add data to the output DataFrame
    output_data.append([x_title, cat_1])
    output_data.append([x_title, cat_2])
    output_data.append([x_title, cat_3])
    

# Create the output DataFrame
output_df = pd.DataFrame(output_data, columns=['x_title', 'cat'])

# Drop duplicate rows
output_df.drop_duplicates(inplace=True)


# Save the output DataFrame to a new CSV file
output_csv = './generated_data/merged_data_cleaned.csv'
output_df.to_csv(output_csv, index=False)

print(f"Indexing the unique category dataset")
# indexing unique categories unique_categeries

unique_cats = output_df['cat'].unique()

print(f"number of unique categories in the dataset: {len(unique_cats)}")


new_df = pd.DataFrame({'cat': unique_cats, 'index': range(len(unique_cats))})

new_df.to_csv('./generated_data/indexed_unique_cats.csv', index=False)



# plotting the freqnency of the each category

category_counts = output_df['cat'].value_counts()

# Create a bar plot
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Frequency of Categories')

# Save the plot as an image
plot_image_path = './generated_data/category_frequency_plot.png'
plt.savefig(plot_image_path)
plt.close() 
