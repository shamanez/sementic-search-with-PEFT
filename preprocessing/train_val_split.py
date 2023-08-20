import pandas as pd
from sklearn.model_selection import train_test_split

# to have a closer evaluation to the real world, 
# we split the dataset based on the unique titles

# Load your CSV into a pandas DataFrame
df = pd.read_csv('./generated_data/Pos_and_Neg.csv')

# to compute the precision and recall 
cleaned_df = pd.read_csv('./generated_data/regex_cleaned.csv')


print("total dataset shape:", df.shape)

# Get unique titles
unique_titles = df['x_title'].unique()


# Split unique titles into train, valid, and test sets
train_titles, remaining_titles = train_test_split(unique_titles, test_size=0.2, random_state=42)
valid_titles, test_titles = train_test_split(remaining_titles, test_size=0.5, random_state=42)

# Create train, valid, and test DataFrames based on selected titles
train_df = df[df['x_title'].isin(train_titles)]
valid_df = df[df['x_title'].isin(valid_titles)]

# we use the original df
test_df = cleaned_df[cleaned_df['x_title'].isin(test_titles)]

# Print the shapes of the splits
print("Train set shape:", train_df.shape)
print("Valid set shape:", valid_df.shape)
print("Test set shape:", test_df.shape)

# Save the splits as CSV files
train_df.to_csv('./training-dataset/train_split.csv', index=False)
valid_df.to_csv('./training-dataset/valid_split.csv', index=False)
test_df.to_csv('./training-dataset/test_split.csv', index=False)
