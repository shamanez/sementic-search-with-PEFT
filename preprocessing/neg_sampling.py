import pandas as pd
import random

# Load the merged_data_cleaned.csv and unique_categories_cleaned.csv files
positve_data = pd.read_csv('./generated_data/merged_data_cleaned.csv')

# will be used to sample negatives
unique_cat_list = positve_data['cat'].unique()


# Add the 'Tag' column with all values as 1 to merged_data
# label numbel one for all the postive examples
positve_data['label'] = 1

# Count the occurrences of each unique x_title in merged_data
x_title_counts = positve_data['x_title'].value_counts()

# here I want to balance the number of postives and negatives for each job title
# this way we train the model with some populariy metrics
NUM_NEGATIVES = 1

# Create an empty DataFrame to store False data
negative_data = pd.DataFrame(columns=['x_title', 'cat', 'label'])


# Generate False data for each x_title
for x_title, count in x_title_counts.items():
    

    unique_positive_cat_list = positve_data[positve_data['x_title'] == x_title]['cat']
    
    neg_cat_list = list(set(unique_cat_list) - set(unique_positive_cat_list))
        
    # Select 2T unique categories randomly from unique_categories
    sampled_negative_categories = random.sample(neg_cat_list, NUM_NEGATIVES * count)
    
    # Create rows for False data
    negative_rows = pd.DataFrame({'x_title': [x_title] * (NUM_NEGATIVES * count),
                               'cat': sampled_negative_categories,
                               'label': 0})
        
    # Append the rows to the false_data DataFrame
    negative_data = negative_data.append(negative_rows, ignore_index=True)
    

pos_neg_df = pd.concat([positve_data, negative_data], axis=0, ignore_index=True) \
                .sample(frac=1, random_state=42)  # frac=1 shuffles all rows


pos_neg_df = pos_neg_df.dropna()


# Save the combined True and False data as True_and_False.csv
pos_neg_df.to_csv('./generated_data/Pos_and_Neg.csv', index=False)

# Display a sample set of the resulting DataFrame
print(pos_neg_df.sample(10))
