import pandas as pd

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('Truedata.csv')

# Group by 'cat' and count the number of occurrences
grouped = data.groupby('cat')['x_title'].count().reset_index()
grouped.columns = ['cat', 'count']

# Save the grouped data to a new CSV file
grouped.to_csv('counts.csv', index=False)

print('done')
