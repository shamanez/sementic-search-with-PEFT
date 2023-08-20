import pandas as pd

# Read the merged data
merged_data = pd.read_csv('merged_data_cleaned.csv')

# Add the 'Tag' column and set values to 1
merged_data['Tag'] = 1

# Save the modified data as 'Truedata.csv'
merged_data.to_csv('Truedata.csv', index=False)

#step 1 done
print('1/3 over')

T = 16

# Get unique x_title values
unique_x_titles = merged_data['x_title'].unique()

# Repeat each x_title T times
repeated_x_titles = list(unique_x_titles) * T

# Create a DataFrame with the repeated x_titles
repeated_data = pd.DataFrame({'x_title': repeated_x_titles})

# Read the unique_categories_cleaned.csv file
unique_categories = pd.read_csv('unique_categories_cleaned.csv')

# Initialize an empty DataFrame to store false data
false_data = pd.DataFrame(columns=['x_title', 'cat', 'Tag'])

# Iterate through each x_title
for x_title in unique_x_titles:
    # Get 16 unique categories for each x_title
    categories = unique_categories.sample(16)['Unique_Tag'].tolist()

    # Append the data to the false_data DataFrame
    false_data = false_data.append(pd.DataFrame({'x_title': [x_title] * 16, 'cat': categories, 'Tag': 0}), ignore_index=True)

# Save the false data as Falsedata.csv
false_data.to_csv('Falsedata.csv', index=False)

#step 2 done
print('2/3 over')

# Read the true data and false data
true_data = pd.read_csv('Truedata.csv')
false_data = pd.read_csv('Falsedata.csv')

# Concatenate both datasets
combined_data = pd.concat([true_data, false_data], ignore_index=True)

# Save the combined data as 'True_and_False.csv'
combined_data.to_csv('True_and_False.csv', index=False)


#step 3 done
print('over')
