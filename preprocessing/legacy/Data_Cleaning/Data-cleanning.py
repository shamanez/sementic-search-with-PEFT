#for merged_data file

import pandas as pd
# Load the input CSV file into a pandas DataFrame
input_csv = 'merged_data.csv'
df = pd.read_csv(input_csv)

# Create a new DataFrame for the output CSV
output_data = []

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
output_csv = 'merged_data_cleaned.csv'
output_df.to_csv(output_csv, index=False)

#................................................
#for unique_categeries

import pandas as pd
# Load the CSV file
csv_file_path = 'unique_categories.csv'
df = pd.read_csv(csv_file_path)

# Remove duplicate rows and keep the first occurrence
df_cleaned = df.drop_duplicates()

# Save the cleaned data to a new CSV file
cleaned_csv_file_path = 'unique_categories_cleaned.csv'
df_cleaned.to_csv(cleaned_csv_file_path, index=False)
