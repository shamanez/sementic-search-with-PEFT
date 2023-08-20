import pandas as pd

# Load the True_data CSV file into a pandas DataFrame
true_data_df = pd.read_csv('Truedata.csv')

# Create a dictionary to store unique cat values and their corresponding index
cat_index_dict = {}
unique_cats = true_data_df['cat'].unique()
for index, cat in enumerate(unique_cats):
    cat_index_dict[cat] = index

# Create a new DataFrame to store cat_list and cat_list_index
cat_list_df = pd.DataFrame(columns=['x_title', 'cat_list', 'cat_list_index'])

# Group data by x_title and create cat_list and cat_list_index
for x_title, group in true_data_df.groupby('x_title'):
    cat_list = group['cat'].tolist()
    cat_list_index = [cat_index_dict[cat] for cat in cat_list]
    cat_list_df = cat_list_df.append({'x_title': x_title, 'cat_list': cat_list, 'cat_list_index': cat_list_index}, ignore_index=True)

# Save the resulting DataFrame to a new CSV file
cat_list_df.to_csv('cat_list.csv', index=False)

#completed task
print('Done')
