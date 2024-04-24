import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# initialize file names
f1 = './datasets/dataset1_pre.csv' # 
f2 = './datasets/test.ft.txt' 
f3 = './datasets/preprocessed_kindle_review .csv' #

# initialize dataframes: all empty
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

files_list = [f1, f2, f3]

#file_path = '/content/drive/MyDrive/Colab-Notebooks/ML-Data/' + f1
#df1 = pd.read_csv(file_path, names=['userId', 'productId','Rating','timestamp'])

def load_df(fname, df, name):
  #file_path = '/content/drive/MyDrive/Colab-Notebooks/ML-Data/' + fname
  file_path = fname

  # Read the CSV file
  new_data = pd.read_csv(file_path, low_memory=False)

  # If df is still empty, you can directly assign new_data to df
  if df.empty:
    df = new_data
  else:
    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

  print(name + ' has loaded in.')
  #print(df.shape)
  #print(df.head())
  #print('\n')
  return df

df1 = load_df(f1, df1, 'df1')
#df2 = load_df(f2, df2, 'df2')
df3 = load_df(f3, df3, 'df3')

df_list = [df1, df2, df3]

sample_size = 5000
# Split the data based on 'class_index' and sample 'sample_size' number of rows from each class 
index_1_df1 = df1[df1['class_index'] == 1]
index_2_df1 = df1[df1['class_index'] == 2]
sampled_df1 = pd.concat([index_1_df1.sample(sample_size, random_state=42), index_2_df1.sample(sample_size, random_state=42)], ignore_index=True)

# Define the filename
#filename = './dataset-csv-files/dataset1.csv'
filename = './dataset-csv-files/dataset1.csv'

sampled_df1.to_csv(filename, index=False)

df2 = pd.read_csv(f2, sep='\t', header=None,dtype=str,keep_default_na=False)

# split column 0 into two columns
df2 = df2.apply(lambda x: x[0].split(' ', 1), axis=1, result_type='expand')

# add column names
columns = ['label', 'text']
df2.columns = columns
# Split the data based on 'label' and sample 'sample_size' number of rows from each class 
sample_size = 5000
# Split the data based on 'class_index' and sample 'sample_size' number of rows from each class 
label_1_df2 = df2[df2['label'] == '__label__1']
label_2_df2 = df2[df2['label'] == '__label__2']
sampled_df2 = pd.concat([label_1_df2.sample(sample_size), label_2_df2.sample(sample_size)])
sampled_df2.to_csv('./dataset-csv-files/dataset2.csv', index=False, header=True)

# see distribution
df3_class_index = df3['rating'].value_counts()
print(df3_class_index)

# df3 processing
df3 = df3.replace({'rating': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}})

df3 = df3.drop(columns=['Unnamed: 0'])
df3.to_csv('./dataset-csv-files/dataset3.csv', index=False, header=True)

# Print the path where the CSV file was saved
print("CSV file saved at:", filename)