import pandas as pd
import random

# Load the merged_data_cleaned.csv and unique_categories_cleaned.csv files
merged_data = pd.read_csv('merged_data_cleaned.csv')
unique_categories = pd.read_csv('unique_categories_cleaned.csv')

# Add the 'Tag' column with all values as 1 to merged_data
merged_data['Tag'] = 1

# Save the modified merged_data as Truedata.csv
merged_data.to_csv('Truedata.csv', index=False)

# Count the occurrences of each unique x_title in merged_data
x_title_counts = merged_data['x_title'].value_counts()

# Create an empty DataFrame to store False data
false_data = pd.DataFrame(columns=['x_title', 'cat', 'Tag'])

# Generate False data for each x_title
for x_title, count in x_title_counts.items():
    # Select 2T unique categories randomly from unique_categories
    selected_categories = random.sample(list(unique_categories['Unique_Tag']), 2 * count)
    
    # Create rows for False data
    false_rows = pd.DataFrame({'x_title': [x_title] * (2 * count),
                               'cat': selected_categories,
                               'Tag': 0})
    
    # Append the rows to the false_data DataFrame
    false_data = false_data.append(false_rows, ignore_index=True)

# Save the False data as Falsedata.csv
false_data.to_csv('Falsedata.csv', index=False)

# Concatenate True and False data
true_and_false_data = pd.concat([merged_data[['x_title', 'cat', 'Tag']], false_data])

# Save the combined True and False data as True_and_False.csv
true_and_false_data.to_csv('True_and_False.csv', index=False)

# Display a sample set of the resulting DataFrame
print(true_and_false_data.sample(10))
